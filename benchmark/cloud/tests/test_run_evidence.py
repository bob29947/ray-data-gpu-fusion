from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from ray.autoscaler._private.util import prepare_config, validate_config

from benchmark.cloud import run_evidence as harness


def valid_config() -> dict[str, object]:
    return {
        "region": "us-west-2",
        "availability_zone": "us-west-2d",
        "ami_id": "ami-00000000000000000",
        "subnet_id": "subnet-00000000000000000",
        "security_group_ids": ["sg-00000000000000000"],
        "iam_instance_profile_arn": (
            "arn:aws:iam::000000000000:instance-profile/ray-evidence-test"
        ),
        "aws_profile": None,
        "ec2_key_name": None,
        "ssh_private_key_path": None,
        "ssh_user": "ubuntu",
        "docker_image": "rayproject/ray:2.55.1-py311",
        "gpu_instance_type": "g6.4xlarge",
        "cpu_head_instance_type": "m6i.4xlarge",
        "root_volume_gib": 100,
    }


def write_config(path: Path, document: dict[str, object] | None = None) -> Path:
    path.write_text(json.dumps(document or valid_config()))
    return path


@pytest.mark.parametrize(
    ("topology_name", "head_type", "worker_min", "worker_max", "head_options"),
    (
        ("fixed-1", "ray.head.gpu", 0, 0, ["--gpus=all"]),
        ("fixed-4", "ray.head.gpu", 3, 3, ["--gpus=all"]),
        ("autoscale-1-4", "ray.head.gpu", 0, 3, ["--gpus=all"]),
        ("autoscale-0-4-cold", "ray.head.cpu", 0, 4, []),
    ),
)
def test_render_cluster_topologies(
    tmp_path: Path,
    topology_name: str,
    head_type: str,
    worker_min: int,
    worker_max: int,
    head_options: list[str],
) -> None:
    run_id = "test-run"
    cluster_name = f"ray-gpu-adm-{run_id}-001-minimal-{topology_name}-incident-r1-n1"
    rendered = harness.render_cluster_config(
        harness.DEFAULT_TEMPLATE.read_text(),
        config=valid_config(),
        topology=harness.TOPOLOGIES[topology_name],
        cluster_name=cluster_name,
        run_id=run_id,
        arm=harness.ARMS["minimal"],
        harness_directory=tmp_path / "staged-harness",
        wheel_directory=harness.ROOT / "wheels" / "pr-candidate",
        wheel_file="ray-test-cp311-cp311-linux_x86_64.whl",
        expected_wheel_sha256="a" * 64,
        expected_ray_commit="b" * 40,
        source_provenance="unit-test-tree",
        expected_harness_sha256="c" * 64,
    )

    assert harness.TOKEN_RE.search(rendered) is None
    parsed = yaml.safe_load(rendered)
    prepared = prepare_config(parsed)
    validate_config(prepared)
    assert parsed["head_node_type"] == head_type
    assert parsed["file_mounts"]["/home/ray/evidence"] == str(
        (tmp_path / "staged-harness").resolve()
    )
    assert set(parsed["rsync_exclude"]) == {"__pycache__", "*.pyc"}
    assert parsed["docker"]["head_run_options"] == head_options
    worker = parsed["available_node_types"]["ray.worker.gpu"]
    assert worker["min_workers"] == worker_min
    assert worker["max_workers"] == worker_max
    tags = worker["node_config"]["TagSpecifications"][0]["Tags"]
    assert {item["Key"]: item["Value"] for item in tags}[
        harness.RUN_TAG_KEY
    ] == run_id
    assert worker["node_config"]["SubnetIds"] == [valid_config()["subnet_id"]]
    assert all(isinstance(command, str) for command in parsed["setup_commands"])
    head_start = "\n".join(parsed["head_start_ray_commands"])
    assert "--include-dashboard=false" in head_start
    assert "--dashboard-host" not in head_start


def test_default_mode_is_dry_run_and_never_calls_aws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_config(tmp_path / "config.json")
    artifacts = tmp_path / "artifacts"

    def forbidden(*_args, **_kwargs):
        pytest.fail("default dry-run reached cloud execution")

    monkeypatch.setattr(harness, "aws_preflight", forbidden)
    monkeypatch.setattr(harness, "execute_campaign", forbidden)
    result = harness.main(
        [
            "--config",
            str(config),
            "--artifact-root",
            str(artifacts),
            "--run-id",
            "dry-run-test",
            "--arms",
            "stock",
            "--topologies",
            "fixed-1",
            "--workloads",
            "incident",
            "--ranks",
            "1",
            "--repetitions",
            "1",
        ]
    )

    assert result == 0
    plan = json.loads((artifacts / "plan.json").read_text())
    assert plan["mode"] == "dry-run"
    assert len(plan["cases"]) == 1
    staged = plan["cases"][0]["staged_harness"]
    staged_directory = Path(staged["directory"])
    assert staged_directory != harness.ROOT
    assert artifacts in staged_directory.parents
    assert staged["file_count"] == len(harness.STAGED_HARNESS_FILES)
    assert staged["total_size_bytes"] < 512 * 1024
    assert len(staged["content_sha256"]) == 64
    assert len(staged["manifest_sha256"]) == 64
    assert set(staged["files"]) == set(harness.STAGED_HARNESS_FILES)
    assert all(
        not (staged_directory / relative).stat().st_mode & 0o222
        for relative in harness.STAGED_HARNESS_FILES
    )
    rendered = yaml.safe_load(Path(plan["cases"][0]["cluster_config"]).read_text())
    assert rendered["file_mounts"]["/home/ray/evidence"] == str(staged_directory)


def test_local_config_rejects_placeholder(tmp_path: Path) -> None:
    document = valid_config()
    document["ami_id"] = "REPLACE_WITH_AMI"
    path = write_config(tmp_path / "config.json", document)

    with pytest.raises(ValueError, match="placeholder"):
        harness.load_local_config(path)


def test_local_config_rejects_secret_fields(tmp_path: Path) -> None:
    document = valid_config()
    document["aws_secret_access_key"] = "not-a-real-secret"
    path = write_config(tmp_path / "config.json", document)

    with pytest.raises(ValueError, match="credential material"):
        harness.load_local_config(path)


def evidence_tags(cluster_name: str, run_id: str, arm: str) -> list[dict[str, str]]:
    return [
        {"Key": harness.PROJECT_TAG_KEY, "Value": harness.PROJECT_TAG_VALUE},
        {"Key": harness.RUN_TAG_KEY, "Value": run_id},
        {"Key": harness.ARM_TAG_KEY, "Value": arm},
        {"Key": harness.CLUSTER_TAG_KEY, "Value": cluster_name},
        {"Key": harness.RAY_CLUSTER_TAG_KEY, "Value": cluster_name},
    ]


class FakeEC2:
    def __init__(self, instance_responses=(), volume_responses=()):
        self.instance_responses = list(instance_responses)
        self.volume_responses = list(volume_responses)
        self.instance_filters = []
        self.volume_filters = []
        self.terminated = []
        self.deleted_volumes = []

    def describe_instances(self, *, Filters):
        self.instance_filters.append(Filters)
        if not self.instance_responses:
            raise AssertionError("unexpected instance poll")
        return self.instance_responses.pop(0)

    def terminate_instances(self, *, InstanceIds):
        self.terminated.append(list(InstanceIds))
        return {}

    def describe_volumes(self, *, Filters):
        self.volume_filters.append(Filters)
        if not self.volume_responses:
            raise AssertionError("unexpected volume poll")
        return self.volume_responses.pop(0)

    def delete_volume(self, *, VolumeId):
        self.deleted_volumes.append(VolumeId)
        return {}


def instance_response(instance_id: str, tags: list[dict[str, str]]) -> dict:
    return {"Reservations": [{"Instances": [{"InstanceId": instance_id, "Tags": tags}]}]}


def empty_instances() -> dict:
    return {"Reservations": []}


def timeline_instance_response(
    instance_id: str, tags: list[dict[str, str]], state: str
) -> dict:
    return {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "InstanceType": "g6.4xlarge",
                        "State": {"Name": state},
                        "Placement": {"AvailabilityZone": "us-west-2d"},
                        "PrivateIpAddress": "10.0.0.1",
                        "Tags": [*tags, {"Key": "ray-node-kind", "Value": "head"}],
                    }
                ]
            }
        ]
    }


def test_instance_timeline_records_pending_to_running_and_stops_once(
    tmp_path: Path,
) -> None:
    run_id = "timeline-run"
    arm = "minimal"
    cluster = f"ray-gpu-adm-{run_id}-001-{arm}-fixed-1-incident-r1-n1"
    tags = evidence_tags(cluster, run_id, arm)
    instance_id = "i-00000000000000000"
    client = FakeEC2(
        instance_responses=[
            timeline_instance_response(instance_id, tags, "pending"),
            timeline_instance_response(instance_id, tags, "running"),
        ]
    )
    output = tmp_path / "timeline.json"
    timeline = harness.InstanceTimeline(
        client=client,
        cluster_name=cluster,
        run_id=run_id,
        arm=arm,
        output=output,
        poll_seconds=60,
    )

    timeline.start()
    report = timeline.stop()
    repeated = timeline.stop()

    assert repeated is report
    assert [
        observation["instances"][0]["state"]
        for observation in report["observations"]
    ] == ["pending", "running"]
    assert report["observations"][1]["elapsed_s"] >= report["observations"][0][
        "elapsed_s"
    ]
    assert report["errors"] == []
    assert json.loads(output.read_text())["observations"][-1]["instances"][0][
        "ray_node_kind"
    ] == "head"
    assert len(client.instance_responses) == 0


def test_instance_timeline_stop_before_start_is_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    run_id = "timeline-run"
    arm = "stock"
    cluster = f"ray-gpu-adm-{run_id}-001-{arm}-fixed-1-incident-r1-n1"
    client = FakeEC2(instance_responses=[empty_instances()])
    timeline = harness.InstanceTimeline(
        client=client,
        cluster_name=cluster,
        run_id=run_id,
        arm=arm,
        output=tmp_path / "timeline.json",
        poll_seconds=5,
    )

    first = timeline.stop()
    second = timeline.stop()

    assert first is second
    assert len(first["observations"]) == 1
    assert first["errors"] == []


def test_instance_timeline_does_not_issue_final_poll_when_thread_is_stuck(
    tmp_path: Path,
) -> None:
    class StuckThread:
        def join(self, timeout):
            assert timeout >= 10

        def is_alive(self):
            return True

    run_id = "timeline-run"
    arm = "stock"
    cluster = f"ray-gpu-adm-{run_id}-001-{arm}-fixed-1-incident-r1-n1"
    client = FakeEC2()
    timeline = harness.InstanceTimeline(
        client=client,
        cluster_name=cluster,
        run_id=run_id,
        arm=arm,
        output=tmp_path / "timeline.json",
        poll_seconds=5,
    )
    timeline._thread = StuckThread()
    timeline._thread_started = True

    report = timeline.stop()

    assert report["observations"] == []
    assert report["errors"] == [
        {
            "type": "RuntimeError",
            "message": "EC2 timeline sampler did not stop",
        }
    ]
    assert client.instance_filters == []


def test_instance_timeline_rejects_nonpositive_poll_interval(tmp_path: Path) -> None:
    run_id = "timeline-run"
    arm = "stock"
    cluster = f"ray-gpu-adm-{run_id}-001-{arm}-fixed-1-incident-r1-n1"

    with pytest.raises(ValueError, match="poll interval"):
        harness.InstanceTimeline(
            client=FakeEC2(),
            cluster_name=cluster,
            run_id=run_id,
            arm=arm,
            output=tmp_path / "timeline.json",
            poll_seconds=0,
        )


def test_ec2_timeline_metrics_measure_launch_and_full_readiness() -> None:
    head = "i-00000000000000000"
    worker = "i-11111111111111111"
    timeline = {
        "observations": [
            {"elapsed_s": 0, "instances": [{"instance_id": head, "state": "pending"}]},
            {
                "elapsed_s": 10,
                "instances": [
                    {"instance_id": head, "state": "running"},
                    {"instance_id": worker, "state": "pending"},
                ],
            },
            {
                "elapsed_s": 30,
                "instances": [
                    {"instance_id": head, "state": "running"},
                    {"instance_id": worker, "state": "running"},
                ],
            },
        ]
    }

    metrics = harness._ec2_timeline_metrics(timeline)

    assert metrics == {
        "ec2_first_instance_seen_s": 0.0,
        "ec2_first_pending_s": 0.0,
        "ec2_first_running_s": 10.0,
        "ec2_peak_running_instances": 2.0,
        "ec2_observed_peak_running_s": 30.0,
        "ec2_all_observed_instances_running_s": 30.0,
        "ec2_running_instance_seconds": 20.0,
        "ec2_pending_instance_seconds": 30.0,
        "ec2_running_worker_seconds": 0.0,
        "ec2_pending_worker_seconds": 0.0,
        "ec2_timeline_duration_s": 30.0,
    }


def test_ec2_timeline_metrics_separate_worker_launch_from_head_start() -> None:
    head = "i-00000000000000000"
    worker = "i-11111111111111111"
    metrics = harness._ec2_timeline_metrics(
        {
            "observations": [
                {
                    "elapsed_s": 0,
                    "instances": [
                        {
                            "instance_id": head,
                            "state": "running",
                            "ray_node_kind": "head",
                        }
                    ],
                },
                {
                    "elapsed_s": 10,
                    "instances": [
                        {
                            "instance_id": head,
                            "state": "running",
                            "ray_node_kind": "head",
                        },
                        {
                            "instance_id": worker,
                            "state": "pending",
                            "ray_node_kind": "worker",
                        },
                    ],
                },
                {
                    "elapsed_s": 30,
                    "instances": [
                        {
                            "instance_id": head,
                            "state": "running",
                            "ray_node_kind": "head",
                        },
                        {
                            "instance_id": worker,
                            "state": "running",
                            "ray_node_kind": "worker",
                        },
                    ],
                },
                {
                    "elapsed_s": 50,
                    "instances": [
                        {
                            "instance_id": head,
                            "state": "running",
                            "ray_node_kind": "head",
                        },
                        {
                            "instance_id": worker,
                            "state": "running",
                            "ray_node_kind": "worker",
                        },
                    ],
                },
            ]
        }
    )

    assert metrics["ec2_first_worker_seen_s"] == 10
    assert metrics["ec2_first_worker_pending_s"] == 10
    assert metrics["ec2_first_worker_running_s"] == 30
    assert metrics["ec2_observed_peak_running_workers_s"] == 30
    assert metrics["ec2_running_worker_seconds"] == 20
    assert metrics["ec2_pending_worker_seconds"] == 20


def test_campaign_statistics_include_provisioning_and_gpu_timing() -> None:
    def result(repetition: int, elapsed: float, ray_up: float, first_running: float):
        instance_id = "i-00000000000000000"
        return {
            "case_id": f"case-{repetition}",
            "arm": "minimal",
            "topology": "autoscale-1-4",
            "workload": "incident",
            "shuffle_ranks": 4,
            "repetition": repetition,
            "status": "completed",
            "ray_up_completed_s": ray_up,
            "initial_cluster_preflight_completed_s": ray_up + 2,
            "ec2_node_timeline": {
                "observations": [
                    {
                        "elapsed_s": first_running,
                        "instances": [
                            {"instance_id": instance_id, "state": "running"}
                        ],
                    }
                ]
            },
            "workload_result": {
                "elapsed_s": elapsed,
                "output_schema": {"fields": [{"name": "id", "type": "int64"}]},
                "output_rows": 10,
                "output_digest": "same",
                "resource_metrics": {
                    "first_gpu_visible_s": 1 + repetition,
                    "time_to_topology_max_gpus_s": 3 + repetition,
                    "cluster_gpu_seconds": 20 + repetition,
                },
            },
        }

    report = harness._campaign_report(
        [result(1, 10, 5, 4), result(2, 14, 7, 8)]
    )

    assert report["passed"] is True
    row = report["statistics"][0]
    assert row["median_elapsed_s"] == 12
    assert row["median_ray_up_s"] == 6
    assert row["median_initial_cluster_preflight_s"] == 8
    assert row["median_ec2_first_running_s"] == 6
    assert row["median_ec2_peak_running_instances"] == 1
    assert row["median_first_gpu_visible_s"] == 2.5
    assert row["median_time_to_topology_max_gpus_s"] == 4.5
    assert row["median_cluster_gpu_seconds"] == 21.5


def test_teardown_rejects_mismatched_returned_target() -> None:
    run_id = "test-run"
    arm = "minimal"
    cluster = f"ray-gpu-adm-{run_id}-001-{arm}-fixed-1-incident-r1-n1"
    tags = evidence_tags(cluster, run_id, arm)
    for tag in tags:
        if tag["Key"] == harness.RUN_TAG_KEY:
            tag["Value"] = "different-run"
    client = FakeEC2(
        instance_responses=[instance_response("i-00000000000000000", tags)]
    )

    with pytest.raises(RuntimeError, match="mismatched tags"):
        harness.terminate_cluster_backstop(
            cluster,
            run_id=run_id,
            arm=arm,
            client=client,
            sleep=lambda _seconds: None,
        )
    assert client.terminated == []


def test_teardown_catches_late_instance_and_requires_three_empty_polls() -> None:
    run_id = "test-run"
    arm = "minimal"
    cluster = f"ray-gpu-adm-{run_id}-001-{arm}-fixed-1-incident-r1-n1"
    tags = evidence_tags(cluster, run_id, arm)
    client = FakeEC2(
        instance_responses=[
            instance_response("i-00000000000000000", tags),
            empty_instances(),
            instance_response("i-11111111111111111", tags),
            empty_instances(),
            empty_instances(),
            empty_instances(),
        ]
    )

    report = harness.terminate_cluster_backstop(
        cluster,
        run_id=run_id,
        arm=arm,
        client=client,
        sleep=lambda _seconds: None,
    )

    assert client.terminated == [
        ["i-00000000000000000"],
        ["i-11111111111111111"],
    ]
    assert report["poll_observations"] == 6
    assert report["stable_empty_observations"] == 3
    filter_names = {item["Name"] for item in client.instance_filters[0]}
    assert f"tag:{harness.RAY_CLUSTER_TAG_KEY}" in filter_names
    assert f"tag:{harness.RUN_TAG_KEY}" in filter_names


def test_volume_verification_uses_exact_tags_and_three_empty_polls() -> None:
    run_id = "test-run"
    arm = "minimal"
    cluster = f"ray-gpu-adm-{run_id}-001-{arm}-fixed-1-incident-r1-n1"
    client = FakeEC2(
        volume_responses=[{"Volumes": []}, {"Volumes": []}, {"Volumes": []}]
    )

    report = harness.verify_no_tagged_volumes(
        cluster,
        run_id=run_id,
        arm=arm,
        client=client,
        sleep=lambda _seconds: None,
    )

    assert report["stable_empty_observations"] == 3
    filter_names = {item["Name"] for item in client.volume_filters[0]}
    assert filter_names == {
        f"tag:{harness.PROJECT_TAG_KEY}",
        f"tag:{harness.RUN_TAG_KEY}",
        f"tag:{harness.ARM_TAG_KEY}",
        f"tag:{harness.CLUSTER_TAG_KEY}",
    }


def test_timeline_instance_ids_rejects_malformed_ids() -> None:
    with pytest.raises(ValueError, match="malformed ID"):
        harness.timeline_instance_ids(
            {"observations": [{"instances": [{"instance_id": "not-an-instance"}]}]}
        )


def test_matrix_skips_ranks_above_topology_capacity() -> None:
    cases = harness.build_cases(
        run_id="matrix-test",
        arms=("stock",),
        topologies=("fixed-1", "fixed-2"),
        workloads=("incident", "fan-in"),
        ranks=(1, 2, 3, 4, "default"),
        repetitions=1,
    )

    incident = [case for case in cases if case.workload == "incident"]
    assert {(case.topology, case.shuffle_ranks) for case in incident} == {
        ("fixed-1", 1),
        ("fixed-1", "default"),
        ("fixed-2", 1),
        ("fixed-2", 2),
        ("fixed-2", "default"),
    }
    assert {case.workload for case in cases} == {"incident", "fan-in"}
    fan_in = [case for case in cases if case.workload == "fan-in"]
    assert {case.shuffle_ranks for case in fan_in} == {"default"}


def test_scale_profile_is_repetition_blocked_and_preserves_exact_pool_shapes() -> None:
    cases = harness.build_scale_cases(run_id="scale-test", repetitions=2)

    assert len(cases) == 2 * len(harness.SCALE_CASE_SPECS)
    expected_roles = {item.role for item in harness.SCALE_CASE_SPECS}
    for repetition in (1, 2):
        block = [item for item in cases if item.repetition == repetition]
        assert {item.role for item in block} == expected_roles
        assert [item.randomization_order for item in block] == list(
            range(1, len(expected_roles) + 1)
        )
        assert {item.randomization_block for item in block} == {repetition}

    by_role = {item.role: item for item in cases if item.repetition == 1}
    stock = by_role["scale-stock-workaround"]
    candidate = by_role["scale-candidate-full"]
    assert (stock.topology, stock.shuffle_ranks) == ("fixed-4", 1)
    assert (stock.map_actors_min, stock.map_actors_max) == (1, 1)
    assert (candidate.topology, candidate.shuffle_ranks) == ("fixed-4", 4)
    assert (candidate.map_actors_min, candidate.map_actors_max) == (4, 4)
    assert by_role["autoscale-elastic-candidate"].map_actors_min == 1
    assert by_role["autoscale-elastic-candidate"].map_actors_max == 4
    assert by_role["autoscale-more-stock"].topology == "autoscale-1-7"
    assert by_role["autoscale-default-rank-candidate"].shuffle_ranks == "default"


def test_scale_profile_owns_factors_and_defaults_to_large_rows(tmp_path: Path) -> None:
    args = harness.parse_args(
        [
            "--profile",
            "scale",
            "--config",
            str(tmp_path / "config.json"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "scale-args",
        ]
    )

    assert args.rows == harness.SCALE_DEFAULT_ROWS
    assert args.blocks == harness.SCALE_DEFAULT_BLOCKS
    assert args.arms == ("stock", "minimal")
    assert args.topologies == ("fixed-4", "autoscale-1-4", "autoscale-1-7")

    with pytest.raises(SystemExit):
        harness.parse_args(
            [
                "--profile",
                "scale",
                "--arms",
                "minimal",
                "--run-id",
                "bad-scale-args",
            ]
        )


def test_full_profile_default_topologies_do_not_gain_scale_only_control() -> None:
    args = harness.parse_args(["--run-id", "full-defaults"])

    assert args.profile == "full"
    assert args.topologies == harness.FULL_TOPOLOGIES
    assert "autoscale-1-7" not in args.topologies


def test_scale_profile_rejects_non_g6_nodes() -> None:
    with pytest.raises(ValueError, match="g6.4xlarge"):
        harness.build_plan(
            SimpleNamespace(profile="scale"),
            config={"gpu_instance_type": "p4d.24xlarge"},
            template="",
        )


def test_four_billion_row_scale_override_has_auditable_capacity_bounds() -> None:
    args = harness.parse_args(
        [
            "--profile",
            "scale",
            "--run-id",
            "four-billion",
            "--rows",
            "4000000000",
            "--blocks",
            "1024",
            "--repetitions",
            "5",
        ]
    )
    cases = harness.build_scale_cases(
        run_id=args.run_id, repetitions=args.repetitions
    )

    budget = harness._scale_campaign_budget(cases, args)

    assert args.rows == 4_000_000_000
    assert args.blocks == 1_024
    assert len(cases) == 45
    assert budget["workload_window_gpu_node_hours_at_minimum_capacity"] == 60
    assert budget["workload_window_gpu_node_hours_at_maximum_capacity"] == 97.5
    assert budget["resource_sample_upper_bound_per_case"] == 363
    assert budget["logical_dataset_lower_bounds"] == {
        "rows": 4_000_000_000,
        "blocks": 1_024,
        "maximum_rows_per_block": 3_906_250,
        "source_id_column_bytes": 32_000_000_000,
        "keyed_id_and_key_columns_bytes": 64_000_000_000,
        "python_cudf_and_shuffle_overhead_excluded": True,
    }

    case = {
        **asdict(cases[0]),
        "remote_wheel_path": "/home/ray/wheels/ray.whl",
        "expected_wheel_sha256": "a" * 64,
        "expected_ray_commit": "b" * 40,
        "expected_harness_sha256": "c" * 64,
        "source_provenance": "tree",
    }
    _preflight, command, _directory = harness._remote_commands(case=case, args=args)
    assert "--rows 4000000000" in command
    assert "--blocks 1024" in command
    assert "timeout --signal=TERM --kill-after=30s 1800s" in command


def test_scale_profile_dry_run_renders_nine_exact_g6_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path / "config.json")
    artifacts = tmp_path / "artifacts"

    def forbidden(*_args, **_kwargs):
        pytest.fail("scale dry-run reached AWS")

    monkeypatch.setattr(harness, "aws_preflight", forbidden)
    result = harness.main(
        [
            "--profile",
            "scale",
            "--config",
            str(config),
            "--artifact-root",
            str(artifacts),
            "--run-id",
            "scale-dry-run",
            "--repetitions",
            "1",
            "--rows",
            "1000",
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "dry-run"
    assert summary["profile"] == "scale"
    assert summary["rows"] == 1_000
    assert summary["blocks"] == harness.SCALE_DEFAULT_BLOCKS
    assert summary["cases"] == 9
    assert summary["scale_campaign_budget"]["cases"] == 9
    plan = json.loads((artifacts / "plan.json").read_text())
    assert plan["mode"] == "dry-run"
    assert plan["matrix"]["profile"] == "scale"
    assert plan["matrix"]["repetition_blocked_randomization"] is True
    assert len(plan["cases"]) == 9
    assert {case["map_actor_pool"]["constructor"] for case in plan["cases"]} == {
        "size",
        "min_size,max_size",
    }
    stock_more = next(
        case for case in plan["cases"] if case["role"] == "autoscale-more-stock"
    )
    rendered = yaml.safe_load(Path(stock_more["cluster_config"]).read_text())
    assert rendered["available_node_types"]["ray.worker.gpu"]["max_workers"] == 6
    assert plan["scale_profile"]["autoscale_more_control"][
        "maximum_gpu_capacity_ratio"
    ] == 1.75


def _scale_result(spec, repetition: int, elapsed_s: float) -> dict[str, object]:
    return {
        "case_id": f"{spec.role}-{repetition}",
        "arm": spec.arm,
        "topology": spec.topology,
        "workload": spec.workload,
        "shuffle_ranks": spec.shuffle_ranks,
        "repetition": repetition,
        "map_actors_min": spec.map_actors_min,
        "map_actors_max": spec.map_actors_max,
        "role": spec.role,
        "correctness_group": spec.correctness_group,
        "randomization_block": repetition,
        "randomization_order": next(
            index
            for index, item in enumerate(harness.SCALE_CASE_SPECS, start=1)
            if item.role == spec.role
        ),
        "status": "completed",
        "teardown": {"verified": True},
        "workload_result": {
            "elapsed_s": elapsed_s,
            "output_schema": {"fields": [{"name": "id", "type": "int64"}]},
            "output_rows": 64,
            "output_digest": "same",
            "resource_metrics": {
                "cluster_gpu_seconds": elapsed_s * 4,
                "owned_gpu_seconds": elapsed_s * 3,
                "peak_cluster_gpus": min(
                    4, harness.TOPOLOGIES[spec.topology].max_gpus
                ),
            },
        },
    }


def test_scale_report_applies_paired_speed_regression_and_cleanup_gates() -> None:
    results = []
    durations = {
        "scale-stock-workaround": 100.0,
        "scale-candidate-full": 80.0,
        "normal-stock-equal": 100.0,
        "normal-candidate-equal": 104.0,
    }
    for repetition in range(1, 6):
        for spec in harness.SCALE_CASE_SPECS:
            results.append(
                _scale_result(
                    spec,
                    repetition,
                    durations.get(spec.role, 90.0) + repetition / 100,
                )
            )

    report = harness._campaign_report(results, profile="scale")

    assert report["passed"] is True
    scale = report["scale_profile"]
    assert scale["merge_gates"]["scale_speedup"]["passed"] is True
    assert scale["merge_gates"]["scale_speedup"]["paired_by_repetition"] is True
    assert scale["merge_gates"]["equal_shape_regression"]["passed"] is True
    assert scale["merge_gates"]["completion"]["passed"] is True
    assert scale["merge_gates"]["correctness"]["passed"] is True
    assert scale["merge_gates"]["cleanup"]["passed"] is True
    assert scale["autoscale_more_resource_tradeoff"][
        "maximum_gpu_capacity_ratio"
    ] == 1.75


def test_scale_report_rejects_sub_threshold_speedup() -> None:
    results = []
    for repetition in range(1, 6):
        for spec in harness.SCALE_CASE_SPECS:
            elapsed = 100.0
            if spec.role == "scale-candidate-full":
                elapsed = 95.0
            results.append(_scale_result(spec, repetition, elapsed))

    report = harness._campaign_report(results, profile="scale")

    assert report["passed"] is False
    assert report["scale_profile"]["merge_gates"]["scale_speedup"][
        "passed"
    ] is False


def test_case_timing_separates_demand_node_readiness_and_progress() -> None:
    result = {
        "ec2_node_timeline": {
            "started_at": "2026-01-01T00:00:00+00:00",
            "observations": [
                {
                    "elapsed_s": 5,
                    "instances": [
                        {
                            "instance_id": "i-00000000000000000",
                            "state": "running",
                            "ray_node_kind": "head",
                        }
                    ],
                },
                {
                    "elapsed_s": 20,
                    "instances": [
                        {
                            "instance_id": "i-00000000000000000",
                            "state": "running",
                            "ray_node_kind": "head",
                        },
                        {
                            "instance_id": "i-11111111111111111",
                            "state": "running",
                            "ray_node_kind": "worker",
                        },
                    ],
                },
            ],
        },
        "workload_result": {
            "started_at": "2026-01-01T00:00:10+00:00",
            "workload_demand_started_at": "2026-01-01T00:00:12+00:00",
            "resource_metrics": {
                "first_gpu_visible_at": "2026-01-01T00:00:13+00:00",
                "observed_peak_gpus_visible_at": "2026-01-01T00:00:25+00:00",
                "topology_max_gpus_visible_at": "2026-01-01T00:00:25+00:00",
                "first_useful_progress_at": "2026-01-01T00:00:18+00:00",
            },
        },
    }

    metrics = harness._case_timing_metrics(result)

    assert metrics["workload_remote_start_offset_s"] == 10
    assert metrics["workload_demand_start_offset_s"] == 12
    assert metrics["workload_demand_to_first_progress_s"] == 6
    assert metrics["ray_first_gpu_visible_to_first_progress_s"] == 5
    assert metrics["ray_topology_max_gpu_visible_to_first_progress_s"] == -7
    assert metrics["ec2_observed_peak_running_to_first_progress_s"] == -2
    assert metrics["ec2_first_worker_running_to_first_progress_s"] == -2


def test_remote_workload_command_carries_exact_elastic_pool_bounds() -> None:
    case = {
        "case_id": "elastic-case",
        "arm": "minimal",
        "topology": "autoscale-1-4",
        "workload": "incident",
        "shuffle_ranks": 4,
        "map_actors_min": 1,
        "map_actors_max": 4,
        "remote_wheel_path": "/home/ray/wheels/ray.whl",
        "expected_wheel_sha256": "a" * 64,
        "expected_ray_commit": "b" * 40,
        "expected_harness_sha256": "c" * 64,
        "source_provenance": "tree",
    }
    args = SimpleNamespace(
        run_id="remote-command",
        rows=100,
        blocks=4,
        groups=2,
        batch_size=8,
        sample_interval_seconds=1,
        workload_timeout_seconds=60,
    )

    _preflight, command, _directory = harness._remote_commands(
        case=case, args=args
    )

    assert "--map-actors-min 1" in command
    assert "--map-actors-max 4" in command
    assert "--map-actors " not in command
    assert "--expected-harness-sha256 " + "c" * 64 in _preflight


def test_run_logged_decodes_timeout_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ray", "down"], 1, output=b"partial bytes")

    monkeypatch.setattr(harness.subprocess, "run", timeout)
    log_path = tmp_path / "timeout.log"
    completed = harness._run_logged(
        ("ray", "down"),
        log_path=log_path,
        environment={},
        timeout_seconds=1,
    )

    assert completed.returncode == 124
    assert "partial bytes" in log_path.read_text()
    assert "LOCAL COMMAND TIMEOUT" in log_path.read_text()


def test_down_exception_cannot_skip_exact_backstops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run_logged(command, **_kwargs):
        if "down" in command:
            raise RuntimeError("simulated down failure")
        return subprocess.CompletedProcess(command, 1, "up failed", None)

    def instances(*_args, **_kwargs):
        calls.append("instances")
        return {"terminated_instance_ids": []}

    def volumes(*_args, **_kwargs):
        calls.append("volumes")
        return {"remaining_volume_ids": []}

    monkeypatch.setattr(harness, "_run_logged", run_logged)
    monkeypatch.setattr(harness, "terminate_cluster_backstop", instances)
    monkeypatch.setattr(harness, "verify_no_tagged_volumes", volumes)
    cluster = "ray-gpu-adm-unit-run-001-stock-fixed-1-incident-r1-n1"
    cluster_path = tmp_path / "cluster.yaml"
    cluster_path.write_text("cluster_name: test\n")
    args = SimpleNamespace(
        ray_cli=Path("/ray"),
        run_id="unit-run",
        teardown_timeout_seconds=1,
        teardown_poll_seconds=0.01,
        workload_timeout_seconds=1,
    )
    case = {
        "case_id": "001-stock-fixed-1-incident-r1-n1",
        "cluster_name": cluster,
        "cluster_config": str(cluster_path),
        "arm": "stock",
        "topology": "fixed-1",
    }

    result = harness.execute_case(
        case=case, args=args, config=valid_config(), ec2_client=object()
    )

    assert calls == ["instances", "volumes"]
    assert result["status"] == "cleanup-error"
    assert result["teardown"]["verified"] is False


def test_pre_down_instance_ids_remain_allowed_for_volume_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_instance = "i-00000000000000000"
    allowed_attachments = []

    class Timeline:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            return {
                "observations": [
                    {"instances": [{"instance_id": observed_instance}]}
                ],
                "errors": [],
            }

    def run_logged(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1 if "up" in command else 0,
            "",
            None,
        )

    def volumes(*_args, **kwargs):
        allowed_attachments.extend(kwargs["allowed_instance_ids"])
        return {"remaining_volume_ids": []}

    monkeypatch.setattr(harness, "InstanceTimeline", Timeline)
    monkeypatch.setattr(harness, "_run_logged", run_logged)
    monkeypatch.setattr(
        harness,
        "terminate_cluster_backstop",
        lambda *_args, **_kwargs: {"terminated_instance_ids": []},
    )
    monkeypatch.setattr(harness, "verify_no_tagged_volumes", volumes)
    cluster = "ray-gpu-adm-unit-run-001-stock-fixed-1-incident-r1-n1"
    cluster_path = tmp_path / "cluster.yaml"
    cluster_path.write_text("cluster_name: test\n")
    args = SimpleNamespace(
        ray_cli=Path("/ray"),
        run_id="unit-run",
        teardown_timeout_seconds=1,
        teardown_poll_seconds=0.01,
        workload_timeout_seconds=1,
    )
    case = {
        "case_id": "001-stock-fixed-1-incident-r1-n1",
        "cluster_name": cluster,
        "cluster_config": str(cluster_path),
        "arm": "stock",
        "topology": "fixed-1",
    }

    result = harness.execute_case(
        case=case, args=args, config=valid_config(), ec2_client=object()
    )

    assert allowed_attachments == [observed_instance]
    assert result["teardown"]["instances_discovered_before_ray_down"] == [
        observed_instance
    ]
    assert result["teardown"]["verified"] is True


def test_campaign_aborts_before_next_case_when_cleanup_is_unproven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ray_cli = tmp_path / "ray"
    ray_cli.write_text("")
    calls = []

    def execute_case(*, case, **_kwargs):
        calls.append(case["case_id"])
        return {
            "case_id": case["case_id"],
            "cluster_name": case["cluster_name"],
            "status": "cleanup-error",
            "teardown": {"verified": False},
        }

    monkeypatch.setattr(harness, "execute_case", execute_case)
    base_case = {
        "arm": "stock",
        "topology": "fixed-1",
        "workload": "incident",
        "shuffle_ranks": 1,
        "repetition": 1,
        "map_actors_min": 1,
        "map_actors_max": 1,
        "role": "full-matrix",
        "correctness_group": None,
        "randomization_block": None,
        "randomization_order": None,
    }
    cases = [
        {
            **base_case,
            "case_id": f"case-{index}",
            "cluster_name": f"cluster-{index}",
        }
        for index in (1, 2)
    ]
    args = SimpleNamespace(
        ray_cli=ray_cli,
        artifact_root=tmp_path / "artifacts",
        profile="full",
    )

    returncode = harness.execute_campaign(
        plan={"cases": cases},
        args=args,
        config=valid_config(),
        session=SimpleNamespace(client=lambda _service: object()),
    )

    assert returncode == 1
    assert calls == ["case-1"]
    results = json.loads((args.artifact_root / "campaign-results.json").read_text())
    assert [item["case_id"] for item in results] == ["case-1"]
    abort = json.loads((args.artifact_root / "campaign-abort.json").read_text())
    assert abort["completed_cases"] == 1
    assert abort["planned_cases"] == 2
    report = json.loads((args.artifact_root / "evidence-report.json").read_text())
    assert report["passed"] is False
    assert report["campaign_abort"] == abort
