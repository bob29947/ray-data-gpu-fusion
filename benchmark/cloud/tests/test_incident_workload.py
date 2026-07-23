from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.cloud import incident_workload as workload


def _arguments(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--arm",
        "minimal",
        "--workload",
        "incident",
        "--topology",
        "fixed-4",
        "--shuffle-ranks",
        "4",
        "--map-actors",
        "4",
        "--max-gpus",
        "4",
        "--result",
        str(tmp_path / "result.json"),
        *extra,
    ]


def _wait_sample(elapsed: float, *, pending: bool = True, sequence: int = 7) -> dict:
    actors = [
        {
            "actor_id": "owner",
            "state": "ALIVE",
            "required_resources": {"GPU": 4},
            "node_id": "node-a",
        }
    ]
    if pending:
        actors.append(
            {
                "actor_id": "requester",
                "state": "PENDING_CREATION",
                "required_resources": {"GPU": 1},
            }
        )
    return {
        "elapsed_s": elapsed,
        "alive_nodes": 4,
        "cluster": {"GPU": 4},
        "available": {"GPU": 0},
        "progress": {"sequence": sequence, "stages": {}},
        "ray_state": {"actors": actors, "tasks": [], "placement_groups": []},
    }


def test_default_shuffle_rank_is_not_replaced_with_an_integer(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments[arguments.index("4", arguments.index("--shuffle-ranks"))] = "default"

    args = workload.parse_args(arguments)

    assert args.shuffle_ranks is None


def test_elastic_actor_pool_preserves_minimum_and_maximum(
    tmp_path: Path, monkeypatch
) -> None:
    arguments = _arguments(tmp_path)
    map_index = arguments.index("--map-actors")
    arguments[map_index : map_index + 2] = [
        "--map-actors-min",
        "1",
        "--map-actors-max",
        "4",
    ]
    args = workload.parse_args(arguments)
    calls = []

    class Strategy:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    import ray.data

    monkeypatch.setattr(ray.data, "ActorPoolStrategy", Strategy)

    workload._actor_pool_strategy(args)

    assert args.map_actors_min == 1
    assert args.map_actors_max == 4
    assert calls == [{"min_size": 1, "max_size": 4}]


def test_fixed_actor_pool_uses_size_constructor(tmp_path: Path, monkeypatch) -> None:
    args = workload.parse_args(_arguments(tmp_path))
    calls = []

    class Strategy:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    import ray.data

    monkeypatch.setattr(ray.data, "ActorPoolStrategy", Strategy)

    workload._actor_pool_strategy(args)

    assert calls == [{"size": 4}]


def test_gpu_map_work_iterations_must_be_nonnegative(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        workload.parse_args(_arguments(tmp_path, "--gpu-map-work-iterations", "-1"))


def test_materialize_boundaries_are_opt_in(tmp_path: Path) -> None:
    assert not workload.parse_args(_arguments(tmp_path)).materialize_boundaries
    assert workload.parse_args(
        _arguments(tmp_path, "--materialize-boundaries")
    ).materialize_boundaries


def test_stopped_sampler_cannot_recreate_gpu_monitors() -> None:
    class ForbiddenLock:
        def acquire(self, **_kwargs):
            raise AssertionError("a stopped sampler must not start another sample")

    sampler = object.__new__(workload.ClusterSampler)
    sampler._stopped = True
    sampler._sample_lock = ForbiddenLock()

    sampler.sample()


def test_closed_wait_requires_sustained_resource_cycle() -> None:
    samples = [_wait_sample(0), _wait_sample(31), _wait_sample(62)]

    classification = workload._classify_closed_wait(samples, max_gpus=4)

    assert classification["is_structural_closed_wait"] is True
    graph = classification["wait_graph"]
    assert graph["owners"] == ["actor:owner"]
    assert graph["requests"] == ["actor:requester"]
    assert {edge["relation"] for edge in graph["edges"]} == {
        "allocated_to",
        "waits_for",
    }


def test_unchanged_progress_without_pending_request_is_not_deadlock() -> None:
    samples = [
        _wait_sample(0, pending=False),
        _wait_sample(31, pending=False),
        _wait_sample(62, pending=False),
    ]

    classification = workload._classify_closed_wait(samples, max_gpus=4)

    assert classification["is_structural_closed_wait"] is False
    assert classification["qualifying_tail_samples"] == 0


def test_recent_progress_breaks_closed_wait_suffix() -> None:
    samples = [_wait_sample(0), _wait_sample(31), _wait_sample(62, sequence=8)]

    classification = workload._classify_closed_wait(samples, max_gpus=4)

    assert classification["is_structural_closed_wait"] is False
    assert classification["qualifying_tail_samples"] == 1


def test_stage_progress_metrics_report_signed_handoff_gaps() -> None:
    metrics = workload._stage_progress_metrics(
        [
            {
                "progress": {
                    "stages": {
                        "upstream": {
                            "first_progress_s": 1.0,
                            "last_progress_s": 4.0,
                            "rows": 10,
                        },
                        "overlap": {
                            "first_progress_s": 3.0,
                            "last_progress_s": 5.0,
                            "rows": 10,
                        },
                        "downstream": {
                            "first_progress_s": 7.0,
                            "last_progress_s": 8.0,
                            "rows": 10,
                        },
                    }
                }
            }
        ]
    )

    assert [item["stage"] for item in metrics["stages"]] == [
        "upstream",
        "overlap",
        "downstream",
    ]
    assert [item["gap_s"] for item in metrics["handoffs"]] == [-1.0, 2.0]


def test_premature_downstream_ownership_joins_named_actors_to_gpu_state() -> None:
    progress = {
        "sequence": 3,
        "origin_epoch_s": 100.0,
        "origin_monotonic_s": 1_000.0,
        "origin_boot_id": "boot-a",
        "telemetry_quiescence": {"complete": True},
        "expected_actor_counts": {
            "gpu-map-groups-1": 1,
            "gpu-map-final": 1,
        },
        "stages": {
            "gpu-map-add-key": {
                "first_input_s": 2.0,
                "first_input_monotonic_s": 1_002.0,
                "first_input_rows_sampled": 1,
                "first_input_boot_id": "boot-a",
                "first_input_clock_basis": "single-boot-monotonic",
                "first_progress_s": 3.0,
                "first_progress_monotonic_s": 1_003.0,
                "first_progress_boot_id": "boot-a",
                "first_progress_clock_basis": "single-boot-monotonic",
            },
            "gpu-map-groups-1": {
                "first_input_s": 10.0,
                "first_input_monotonic_s": 1_010.0,
                "first_input_rows_sampled": 1,
                "first_input_boot_id": "boot-a",
                "first_input_clock_basis": "single-boot-monotonic",
                "first_progress_s": 11.0,
                "first_progress_monotonic_s": 1_011.0,
                "first_progress_boot_id": "boot-a",
                "first_progress_clock_basis": "single-boot-monotonic",
            },
            "gpu-map-final": {
                "first_input_s": 20.0,
                "first_input_monotonic_s": 1_020.0,
                "first_input_rows_sampled": 1,
                "first_input_boot_id": "boot-a",
                "first_input_clock_basis": "single-boot-monotonic",
                "first_progress_s": 21.0,
                "first_progress_monotonic_s": 1_021.0,
                "first_progress_boot_id": "boot-a",
                "first_progress_clock_basis": "single-boot-monotonic",
            },
        },
        "actors": {
            "groups-actor": {
                "stage": "gpu-map-groups-1",
                "gpus": 1.0,
                "ready_s": 1.5,
                "ready_monotonic_s": 1_001.5,
                "ready_boot_id": "boot-a",
                "ready_clock_basis": "single-boot-monotonic",
                "registration_events": 1,
                "first_input_s": 10.5,
            },
            "final-actor": {
                "stage": "gpu-map-final",
                "gpus": 1.0,
                "ready_s": 2.0,
                "ready_monotonic_s": 1_002.0,
                "ready_boot_id": "boot-a",
                "ready_clock_basis": "single-boot-monotonic",
                "registration_events": 1,
                "first_input_s": 20.5,
            },
        },
    }
    actors = [
        {
            "actor_id": "groups-actor",
            "state": "ALIVE",
            "required_resources": {"GPU": 1},
            "num_restarts": 0,
            "job_id": "job-a",
            "class_name": "MapWorker(SumGroup)",
        },
        {
            "actor_id": "final-actor",
            "state": "ALIVE",
            "required_resources": {"GPU": 1},
            "num_restarts": 0,
            "job_id": "job-a",
            "class_name": "MapWorker(Identity)",
        },
    ]
    samples = [
        {
            "elapsed_s": elapsed,
            "origin_epoch_s": 100.0,
            "origin_monotonic_s": 1_000.0,
            "origin_boot_id": "boot-a",
            "progress": progress,
            "ray_state": {"actors": actors, "possibly_truncated": False},
        }
        for elapsed in (2.0, 6.0, 10.0, 15.0, 20.0)
    ]

    metrics = workload._premature_downstream_metrics(samples, target_job_id="job-a")

    assert metrics["premature_downstream_actor_count"] == 2
    assert metrics["premature_downstream_ownership_seconds"] == 26.5
    assert metrics["premature_downstream_gpu_seconds"] == 26.5
    assert metrics["mean_ready_to_stage_first_input_s"] == 13.25
    assert metrics["premature_downstream_gpu_seconds_during_earlier_stage"] == 21.5
    assert metrics["peak_premature_downstream_gpus_during_earlier_stage"] == 2
    assert metrics["peak_premature_downstream_gpus"] == 2
    assert metrics["measurement_complete"] is True
    assert metrics["common_clock_verified"] is True
    assert metrics["gcs_corroboration_complete"] is True
    assert {
        (item["operator"], item["gpu_seconds"]) for item in metrics["per_stage"]
    } == {("SumGroup/map_groups", 8.5), ("Identity/map_batches", 18.0)}
    assert {item["stage_role"] for item in metrics["actors"]} == {
        "post-shuffle-map-groups",
        "post-shuffle-map-batches",
    }

    actors[0]["num_restarts"] = 1
    restarted = workload._premature_downstream_metrics(samples, target_job_id="job-a")
    assert restarted["gcs_corroboration_complete"] is False
    assert restarted["measurement_complete"] is False
    actors[0]["num_restarts"] = 0

    progress["actors"]["groups-actor"]["ready_s"] = 10.0
    progress["actors"]["groups-actor"]["ready_monotonic_s"] = 1_010.0
    progress["actors"]["final-actor"]["ready_s"] = 20.0
    progress["actors"]["final-actor"]["ready_monotonic_s"] = 1_020.0
    zero_valid = workload._premature_downstream_metrics(samples, target_job_id="job-a")
    assert zero_valid["premature_downstream_actor_count"] == 0
    assert zero_valid["gcs_corroboration_complete"] is True
    assert zero_valid["measurement_complete"] is True
    progress["actors"]["groups-actor"]["ready_s"] = 1.5
    progress["actors"]["groups-actor"]["ready_monotonic_s"] = 1_001.5
    progress["actors"]["final-actor"]["ready_s"] = 2.0
    progress["actors"]["final-actor"]["ready_monotonic_s"] = 1_002.0

    samples[0]["ray_state"]["possibly_truncated"] = True
    truncated = workload._premature_downstream_metrics(samples, target_job_id="job-a")
    assert truncated["gcs_state_samples_valid"] is False
    assert truncated["measurement_complete"] is False

    progress["actors"]["groups-actor"]["ready_s"] = 10.0
    progress["actors"]["groups-actor"]["ready_monotonic_s"] = 1_010.0
    progress["actors"]["final-actor"]["ready_s"] = 20.0
    progress["actors"]["final-actor"]["ready_monotonic_s"] = 1_020.0
    zero_intervals = workload._premature_downstream_metrics(
        samples, target_job_id="job-a"
    )
    assert zero_intervals["premature_downstream_actor_count"] == 0
    assert zero_intervals["gcs_corroboration_complete"] is False
    assert zero_intervals["measurement_complete"] is False


def test_premature_ownership_fails_closed_for_missing_or_mixed_clock_telemetry() -> (
    None
):
    base = {
        "elapsed_s": 2.0,
        "origin_epoch_s": 100.0,
        "origin_monotonic_s": 1_000.0,
        "origin_boot_id": "boot-a",
        "progress": {
            "origin_epoch_s": 100.0,
            "origin_monotonic_s": 1_000.0,
            "origin_boot_id": "boot-a",
            "telemetry_quiescence": {"complete": True},
            "expected_actor_counts": {
                "gpu-map-groups-1": 1,
                "gpu-map-final": 1,
            },
            "stages": {
                "gpu-map-groups-1": {
                    "first_input_s": 2.0,
                    "first_input_monotonic_s": 1_002.0,
                    "first_input_rows_sampled": 1,
                    "first_input_boot_id": "boot-a",
                    "first_input_clock_basis": "single-boot-monotonic",
                    "first_progress_s": 2.5,
                    "first_progress_monotonic_s": 1_002.5,
                    "first_progress_boot_id": "boot-a",
                    "first_progress_clock_basis": "single-boot-monotonic",
                },
                "gpu-map-final": {
                    "first_input_s": 3.0,
                    "first_input_monotonic_s": 1_003.0,
                    "first_input_rows_sampled": 1,
                    "first_input_boot_id": "boot-a",
                    "first_input_clock_basis": "single-boot-monotonic",
                    "first_progress_s": 3.5,
                    "first_progress_monotonic_s": 1_003.5,
                    "first_progress_boot_id": "boot-a",
                    "first_progress_clock_basis": "single-boot-monotonic",
                },
            },
            "actors": {
                "groups": {
                    "stage": "gpu-map-groups-1",
                    "ready_s": 1.0,
                    "ready_monotonic_s": 1_001.0,
                    "ready_boot_id": "boot-a",
                    "ready_clock_basis": "single-boot-monotonic",
                    "registration_events": 1,
                    "first_input_s": 2.0,
                }
            },
        },
        "ray_state": {"actors": [], "possibly_truncated": False},
    }
    missing = workload._premature_downstream_metrics([base])
    assert missing["measurement_complete"] is False
    assert missing["telemetry_coverage"]["complete"] is False

    complete = {
        **base,
        "origin_boot_id": "boot-b",
        "progress": {
            **base["progress"],
            "actors": {
                **base["progress"]["actors"],
                "final": {
                    "stage": "gpu-map-final",
                    "ready_s": 1.0,
                    "ready_monotonic_s": 1_001.0,
                    "ready_boot_id": "boot-a",
                    "ready_clock_basis": "single-boot-monotonic",
                    "registration_events": 1,
                    "first_input_s": 3.0,
                },
            },
        },
    }
    mixed_clock = workload._premature_downstream_metrics([complete])
    assert mixed_clock["telemetry_coverage"]["complete"] is True
    assert mixed_clock["common_clock_verified"] is False
    assert mixed_clock["measurement_complete"] is False


def test_progress_tracker_uses_emission_time_and_reorders_stage_first_input() -> None:
    tracker = workload._ProgressTracker(
        100.0,
        {"gpu-map-groups-1": 2},
        1_000.0,
        "boot-a",
    )
    tracker.register_actor("gpu-map-groups-1", "later", 1.0, 101.0, 1_001.0, "boot-a")
    tracker.register_actor("gpu-map-groups-1", "earlier", 1.0, 101.5, 1_001.5, "boot-a")
    # Simulate cross-actor delivery arriving in the opposite order from the
    # actual UDF call events.
    tracker.input("gpu-map-groups-1", "later", 1, 110.0, 1_010.0, "boot-a")
    tracker.input("gpu-map-groups-1", "earlier", 1, 105.0, 1_005.0, "boot-a")
    tracker.record("gpu-map-groups-1", "later", 1, 112.0, 1_012.0, "boot-a")
    tracker.record("gpu-map-groups-1", "earlier", 1, 108.0, 1_008.0, "boot-a")

    snapshot = tracker.snapshot()
    stage = snapshot["stages"]["gpu-map-groups-1"]
    assert stage["first_input_s"] == 5.0
    assert stage["last_input_s"] == 10.0
    assert stage["first_progress_s"] == 8.0
    assert stage["last_progress_s"] == 12.0
    assert workload._progress_telemetry_coverage(snapshot)["complete"] is True

    tracker.register_actor("gpu-map-groups-1", "later", 1.0, 113.0, 1_013.0, "boot-a")
    duplicate = workload._progress_telemetry_coverage(tracker.snapshot())
    assert duplicate["complete"] is False
    assert duplicate["stages"][0]["registration_events"] == 3


def test_fixed_pool_coverage_allows_registered_idle_actors_after_stage_input() -> None:
    coverage = workload._progress_telemetry_coverage(
        {
            "expected_actor_counts": {"gpu-map-final": 2},
            "stages": {
                "gpu-map-final": {
                    "first_input_s": 4.0,
                    "first_input_rows_sampled": 1,
                    "first_progress_s": 4.5,
                }
            },
            "actors": {
                "used": {
                    "stage": "gpu-map-final",
                    "ready_s": 1.0,
                    "registration_events": 1,
                    "first_input_s": 4.0,
                },
                "idle": {
                    "stage": "gpu-map-final",
                    "ready_s": 1.5,
                    "registration_events": 1,
                },
            },
        }
    )

    assert coverage["complete"] is True
    assert coverage["stages"][0]["registered_actors"] == 2
    assert coverage["stages"][0]["actors_with_first_input"] == 1


def test_late_registration_delivery_restores_constructor_event_time() -> None:
    tracker = workload._ProgressTracker(100.0, {"gpu-map-final": 1}, 1_000.0, "boot-a")
    tracker.input("gpu-map-final", "actor", 1, 105.0, 1_005.0, "boot-a")
    tracker.register_actor("gpu-map-final", "actor", 1.0, 101.0, 1_001.0, "boot-a")

    actor = tracker.snapshot()["actors"]["actor"]
    assert actor["ready_s"] == 1.0
    assert actor["first_input_s"] == 5.0


def test_first_usable_input_is_positive_once_and_acknowledged() -> None:
    class RemoteCall:
        def __init__(self):
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return "reference"

    class FakeRay:
        def __init__(self):
            self.references = []

        def get(self, reference):
            self.references.append(reference)

    udf = object.__new__(workload._ProgressUdf)
    udf._stage = "gpu-map-final"
    udf._actor_id = "actor"
    udf._boot_id = "boot-a"
    udf._sent_first_input = False
    udf._progress = type("Progress", (), {"input": RemoteCall()})()
    udf._ray = FakeRay()

    udf._input([])
    udf._input([1, 2])
    udf._input([3])

    assert len(udf._progress.input.calls) == 1
    assert udf._progress.input.calls[0][2] == 2
    assert udf._ray.references == ["reference"]


def test_progress_drain_requires_covered_sequence_quiescence(monkeypatch) -> None:
    tracker = workload._ProgressTracker(100.0, {"gpu-map-final": 1}, 1_000.0, "boot-a")
    tracker.register_actor("gpu-map-final", "actor", 1.0, 101.0, 1_001.0, "boot-a")
    tracker.input("gpu-map-final", "actor", 1, 102.0, 1_002.0, "boot-a")
    tracker.record("gpu-map-final", "actor", 1, 103.0, 1_003.0, "boot-a")

    class Snapshot:
        def remote(self):
            return tracker.snapshot()

    class FakeRay:
        @staticmethod
        def get(reference, timeout):
            assert timeout > 0
            return reference

    monkeypatch.setattr(workload, "_TELEMETRY_QUIESCENCE_SECONDS", 0.0)
    monkeypatch.setattr(workload, "_TELEMETRY_DRAIN_POLL_SECONDS", 0.001)
    final, drain = workload._drain_progress_telemetry(
        FakeRay(),
        type("Progress", (), {"snapshot": Snapshot()})(),
        timeout_seconds=0.2,
    )

    assert drain["complete"] is True
    assert drain["quiescence"]["stable_polls"] >= 3
    assert final["telemetry_quiescence"]["complete"] is True


def test_state_snapshot_uses_core_gcs_records_and_keeps_links(monkeypatch) -> None:
    from benchmark import ray_core_state

    def snapshot(**kwargs):
        assert kwargs == {"limit": workload._STATE_RECORD_LIMIT}
        return {
            "actors": [
                {
                    "actor_id": "actor-a",
                    "state": "ALIVE",
                    "node_id": "node-a",
                    "placement_group_id": "pg-a",
                    "required_resources": {"GPU": 1},
                },
                {
                    "actor_id": "cpu-actor",
                    "state": "ALIVE",
                    "required_resources": {"CPU": 1},
                },
            ],
            "tasks": [
                {
                    "task_id": "task-a",
                    "state": "PENDING_NODE_ASSIGNMENT",
                    "actor_id": "actor-a",
                    "required_resources": {"GPU": 1},
                    "events": [{"state": "PENDING"}],
                },
                {
                    "task_id": "cpu-task",
                    "state": "RUNNING",
                    "required_resources": {"CPU": 1},
                },
            ],
            "placement_groups": [
                {
                    "placement_group_id": "pg-a",
                    "state": "PENDING",
                    "bundles": [{"bundle_index": 0, "resources": {"GPU": 1}}],
                    "stats": {"scheduling_attempt": 1},
                },
                {
                    "placement_group_id": "cpu-pg",
                    "state": "PENDING",
                    "bundles": [{"bundle_index": 0, "resources": {"CPU": 1}}],
                },
            ],
        }

    monkeypatch.setattr(ray_core_state, "read_core_state", snapshot)

    snapshot = workload._ray_state_snapshot()

    assert snapshot["actors"][0]["node_id"] == "node-a"
    assert snapshot["actors"][0]["placement_group_id"] == "pg-a"
    assert snapshot["tasks"][0]["events"] == [{"state": "PENDING"}]
    assert snapshot["placement_groups"][0]["bundles"] == [
        {"bundle_index": 0, "resources": {"GPU": 1}}
    ]
    assert [item["actor_id"] for item in snapshot["actors"]] == ["actor-a"]
    assert [item["task_id"] for item in snapshot["tasks"]] == ["task-a"]
    assert [item["placement_group_id"] for item in snapshot["placement_groups"]] == [
        "pg-a"
    ]
    assert snapshot["counts"]["actors"] == {"ALIVE": 2}
    assert snapshot["counts"]["tasks"] == {
        "PENDING_NODE_ASSIGNMENT": 1,
        "RUNNING": 1,
    }
    assert snapshot["record_filter"] == "nonterminal GPU owners and requests"


def test_atomic_json_uses_collision_free_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    threads = [
        threading.Thread(target=workload._atomic_json, args=(path, {"writer": index}))
        for index in range(20)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert json.loads(path.read_text())["writer"] in range(20)
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_row_content_digest_is_not_a_row_count_only_hash() -> None:
    baseline = workload._expected_row_fingerprint([(1, 10)])
    shifted = workload._expected_row_fingerprint([(2, 10)])
    duplicated = workload._expected_row_fingerprint([(1, 5), (1, 5)])

    assert baseline["summary"]["count"] == shifted["summary"]["count"]
    assert baseline["sha256"] != shifted["sha256"]
    assert baseline["sha256"] != duplicated["sha256"]


def test_only_marked_cleanup_failure_is_expected_evidence() -> None:
    marker_error = RuntimeError(workload._EXPECTED_FAILURE_MARKER)

    assert workload._is_expected_injected_failure(
        "failure-cleanup", marker_error, "remote traceback"
    )
    assert not workload._is_expected_injected_failure(
        "incident", marker_error, "remote traceback"
    )
    assert not workload._is_expected_injected_failure(
        "failure-cleanup", RuntimeError("unplanned"), "remote traceback"
    )


def test_l4_identity_check_is_exact() -> None:
    assert workload._is_nvidia_l4("NVIDIA L4")
    assert workload._is_nvidia_l4(b"NVIDIA L4")
    assert not workload._is_nvidia_l4("NVIDIA L40")
    assert not workload._is_nvidia_l4("NVIDIA L4-virtual")


def test_pci_address_normalizes_cuda_and_nvml_domain_width() -> None:
    assert workload._pci_address("0000:34:00.0") == (0, 0x34, 0)
    assert workload._pci_address("00000000:34:00.0") == (0, 0x34, 0)


def test_required_l4_monitor_rejects_multiple_nvml_devices(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: 1),
        )
    )
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 2,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    monitor = workload._NodeGpuMonitor("node-a", require_one_l4=True)
    sample = monitor.sample()

    assert sample["error"]["type"] == "RuntimeError"
    assert "exactly one CUDA and NVML GPU" in sample["error"]["message"]
