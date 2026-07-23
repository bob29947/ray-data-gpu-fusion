from __future__ import annotations

import json

import pytest

from benchmark.local import simulate_autoscaling


def test_dry_run_describes_candidate_ray_data_admission_path(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("RAY_DATA_CLUSTER_AUTOSCALER", raising=False)
    result = tmp_path / "simulation.json"

    assert simulate_autoscaling.main(["--result", str(result)]) == 0

    document = json.loads(result.read_text())
    assert document["status"] == "dry_run"
    assert document["will_start_ray"] is False
    assert document["operator_shape"] == {
        "kind": "Ray Data actor map",
        "actor_pool_strategy": {"min_size": 2, "max_size": 2},
        "per_actor_resources": {"CPU": 0, "GPU": 1},
        "explicit_execution_gpu_ceiling": 2,
        "resource_admission_enabled": True,
        "ray_data_cluster_autoscaler": "V2",
    }
    assert "pending demand" in " ".join(document["experiment_steps"])
    assert "cannot replace" in document["disclaimer"]


def test_simulation_rejects_a_different_logical_gpu_ceiling() -> None:
    with pytest.raises(SystemExit):
        simulate_autoscaling.parse_args(
            ["--result", "unused.json", "--max-logical-gpus", "3"]
        )


def test_pending_evidence_requires_two_gpu_actors_and_zero_progress() -> None:
    actors = [
        {
            "actor_id": "a",
            "state": "ALIVE",
            "required_resources": {"GPU": 1.0},
        },
        {
            "actor_id": "b",
            "state": "PENDING_CREATION",
            "required_resources": {"GPU": 1.0},
        },
        {
            "actor_id": "telemetry",
            "state": "ALIVE",
            "required_resources": {},
        },
    ]
    demand = simulate_autoscaling._active_gpu_actor_demand(actors)
    sample = {
        "cluster_resources": {"GPU": 1.0},
        "available_resources": {},
        "gpu_actor_demand": demand,
        "progress": {},
    }

    assert demand["active_gpu_actor_requests"] == 2
    assert demand["pending_gpu_actor_requests"] == 1
    assert simulate_autoscaling._pending_without_progress(sample, "cold")

    sample["progress"] = {"cold": {"batches": 1}}
    assert not simulate_autoscaling._pending_without_progress(sample, "cold")


def test_timing_metrics_separate_capacity_wait_and_warm_replay() -> None:
    metrics = simulate_autoscaling._timing_metrics(
        {
            "cold_request": 10.0,
            "pending_demand": 11.0,
            "capacity_add_request": 13.0,
            "node_visible": 14.5,
            "both_gpu_actors_alive": 14.8,
            "cold_first_progress": 15.0,
            "cold_completion": 17.0,
            "warm_request": 20.0,
            "warm_first_progress": 20.2,
            "warm_completion": 21.0,
        }
    )

    assert metrics == {
        "resource_request_to_node_visible_s": 4.5,
        "pending_demand_to_node_visible_s": 3.5,
        "capacity_add_request_to_node_visible_s": 1.5,
        "node_visible_to_first_progress_s": 0.5,
        "node_visible_to_completion_s": 2.5,
        "cold_request_to_completion_s": 7.0,
        "warm_request_to_first_progress_s": pytest.approx(0.2),
        "warm_request_to_completion_s": 1.0,
        "node_visible_to_two_actor_ready_s": pytest.approx(0.3),
        "two_actor_ready_to_first_progress_s": pytest.approx(0.2),
    }
