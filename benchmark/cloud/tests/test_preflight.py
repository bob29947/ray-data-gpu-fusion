from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.cloud import preflight


class FakeRayNodes:
    def __init__(self, snapshots: list[list[dict[str, object]]]) -> None:
        self._snapshots = iter(snapshots)
        self._current: list[dict[str, object]] = []

    def nodes(self) -> list[dict[str, object]]:
        self._current = next(self._snapshots, self._current)
        return self._current


def gpu_node(node_id: str) -> dict[str, object]:
    return {"Alive": True, "NodeID": node_id, "Resources": {"GPU": 1.0}}


def test_wait_for_gpu_nodes_polls_until_fixed_minimum() -> None:
    clock = iter((0.0, 1.0, 2.0, 3.0))
    ray = FakeRayNodes([[gpu_node("a")], [gpu_node("a"), gpu_node("b")]])

    nodes, observations = preflight._wait_for_gpu_nodes(
        ray,
        minimum=2,
        timeout_seconds=10,
        poll_seconds=1,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
    )

    assert [node["NodeID"] for node in nodes] == ["a", "b"]
    assert [item["gpu_node_ids"] for item in observations] == [["a"], ["a", "b"]]


def test_wait_for_gpu_nodes_is_bounded() -> None:
    clock = iter((0.0, 1.0, 5.0))
    ray = FakeRayNodes([[], []])

    with pytest.raises(RuntimeError, match="timed out"):
        preflight._wait_for_gpu_nodes(
            ray,
            minimum=1,
            timeout_seconds=4,
            poll_seconds=1,
            monotonic=lambda: next(clock),
            sleep=lambda _seconds: None,
        )


def test_gpu_node_rejects_non_g6_shape() -> None:
    ray = FakeRayNodes(
        [[{"Alive": True, "NodeID": "multi", "Resources": {"GPU": 2.0}}]]
    )

    with pytest.raises(RuntimeError, match="advertises 2.0"):
        preflight._alive_gpu_nodes(ray)


def test_ray_install_report_proves_wheel_commit_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "ray.whl"
    wheel.write_bytes(b"wheel bytes")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    fake_ray = SimpleNamespace(__version__="3.0.0.dev0", __commit__="abc123")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        preflight, "_resource_admission_fields", lambda: ["max_units", "may_submit"]
    )
    monkeypatch.setattr(
        preflight,
        "_ray_direct_url",
        lambda: {
            "url": wheel.resolve().as_uri(),
            "archive_info": {"hash": f"sha256={digest}"},
        },
    )

    report = preflight.ray_install_report(
        arm="minimal",
        expected_wheel_path=wheel,
        expected_wheel_sha256=digest,
        expected_ray_commit="abc123",
        expected_source_provenance="pins/pr-candidate.json:abc123",
    )

    assert report["mounted_wheel_sha256"] == digest
    assert report["commit"] == "abc123"
    assert report["grant_fields"] == ["max_units", "may_submit"]


def test_ray_install_report_rejects_wrong_mounted_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "ray.whl"
    wheel.write_bytes(b"not the expected wheel")
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(__version__="3.0.0.dev0", __commit__="abc123"),
    )

    with pytest.raises(RuntimeError, match="mounted Ray wheel digest mismatch"):
        preflight.ray_install_report(
            arm="stock",
            expected_wheel_path=wheel,
            expected_wheel_sha256="0" * 64,
            expected_ray_commit="abc123",
            expected_source_provenance="stock",
        )


def test_staged_harness_report_verifies_manifest_and_every_file(
    tmp_path: Path,
) -> None:
    relative = "benchmark/cloud/workload.py"
    staged_file = tmp_path / relative
    staged_file.parent.mkdir(parents=True)
    staged_file.write_bytes(b"immutable workload")
    files = {
        relative: {
            "sha256": hashlib.sha256(staged_file.read_bytes()).hexdigest(),
            "size_bytes": staged_file.stat().st_size,
        }
    }
    core = {"schema_version": 1, "files": files}
    content_sha256 = hashlib.sha256(
        preflight._canonical_json_bytes(core)
    ).hexdigest()
    (tmp_path / preflight.STAGED_HARNESS_MANIFEST).write_text(
        json.dumps({**core, "content_sha256": content_sha256})
    )

    report = preflight.staged_harness_report(
        expected_harness_sha256=content_sha256,
        root=tmp_path,
    )

    assert report["content_sha256"] == content_sha256
    assert report["file_count"] == 1
    staged_file.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="failed verification"):
        preflight.staged_harness_report(
            expected_harness_sha256=content_sha256,
            root=tmp_path,
        )


def test_nonterminal_resource_filter_keeps_cleanup_failures() -> None:
    records = [
        {"actor_id": "done", "state": "DEAD"},
        {"actor_id": "live", "state": "ALIVE"},
        {"actor_id": "pending", "state": "PENDING_CREATION"},
    ]

    assert [
        record["actor_id"]
        for record in preflight._nonterminal_records(
            records, terminal_states=preflight.TERMINAL_ACTOR_STATES
        )
    ] == ["live", "pending"]


def test_leak_audit_records_known_detached_service_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray
    from benchmark import ray_core_state

    class Reader:
        def actors(self, *, job_id):
            return [
                {
                    "actor_id": "service",
                    "class_name": "_StatsActor",
                    "name": "datasets_stats_actor",
                    "state": "ALIVE",
                    "job_id": job_id,
                    "is_detached": True,
                    "required_resources": {},
                }
            ]

        def placement_groups(self, *, job_id):
            return []

    monkeypatch.setattr(ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray_core_state, "CoreGcsStateReader", Reader)

    report = preflight.audit_job_resources(
        job_id="job-a",
        timeout_seconds=1,
        poll_seconds=0.1,
        monotonic=lambda: 0,
        sleep=lambda _seconds: None,
    )

    assert report["clean"] is True
    assert report["nonterminal_actors"] == []
    assert report["permitted_detached_ray_data_services"][0]["actor_id"] == "service"


def test_leak_mode_always_prints_structured_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        preflight,
        "audit_job_resources",
        lambda **_kwargs: {
            "job_id": "01000000",
            "clean": False,
            "nonterminal_actors": [{"actor_id": "abc", "state": "ALIVE"}],
            "nonterminal_placement_groups": [],
        },
    )

    assert preflight.main(["--check-job-id", "01000000"]) == 1
    output = capsys.readouterr().out.removeprefix("PREFLIGHT=")
    assert json.loads(output)["leak_audit"]["clean"] is False


def test_runtime_mode_requires_all_provenance_arguments() -> None:
    with pytest.raises(SystemExit):
        preflight.parse_args(["--local-node-only", "--arm", "minimal"])
