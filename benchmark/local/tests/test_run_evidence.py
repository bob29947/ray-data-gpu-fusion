from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
import zipfile

import pytest

from benchmark.local import cluster_daemon
from benchmark.local import probe
from benchmark.local import run_evidence
from benchmark.local import simulate_autoscaling


def test_core_matrix_has_every_rank_and_nonshuffle_once() -> None:
    cases = run_evidence.build_cases(
        run_id="unit",
        arms=list(run_evidence.ARM_NAMES),
        capacities=[1, 2, 4],
        workloads=["incident", "actor-only"],
        ranks=["all"],
        repetitions=1,
        seed=7,
    )
    assert len(cases) == 52
    incident_rank_sets = {}
    for case in cases:
        if case.workload == "actor-only":
            assert case.shuffle_ranks == "default"
        else:
            incident_rank_sets.setdefault((case.arm, case.capacity), set()).add(
                case.shuffle_ranks
            )
    for arm in run_evidence.ARM_NAMES:
        for capacity in (1, 2, 4):
            assert incident_rank_sets[(arm, capacity)] == {
                *range(1, capacity + 1),
                "default",
            }


def test_case_randomization_is_seeded() -> None:
    kwargs = dict(
        run_id="unit",
        arms=["stock", "minimal"],
        capacities=[1, 2],
        workloads=["incident"],
        ranks=["all"],
        repetitions=2,
    )
    first = run_evidence.build_cases(**kwargs, seed=11)
    second = run_evidence.build_cases(**kwargs, seed=11)
    third = run_evidence.build_cases(**kwargs, seed=12)
    assert first == second
    assert first != third


def test_scale_profile_builds_paired_full_resource_and_workaround_cases() -> None:
    cases = run_evidence.build_scale_evidence_cases(
        run_id="scale-unit", repetitions=5, seed=17
    )

    assert len(cases) == 20
    assert all(
        len({case.repetition for case in cases[index : index + 4]}) == 1
        for index in range(0, len(cases), 4)
    )
    for repetition in range(1, 6):
        repeated = [case for case in cases if case.repetition == repetition]
        assert {
            (
                case.comparison_id,
                case.arm,
                case.capacity,
                case.workload,
                case.shuffle_ranks,
                case.map_actors_per_stage,
            )
            for case in repeated
        } == {
            ("incident-scale-speedup", "stock", 4, "incident", 1, 1),
            ("incident-scale-speedup", "minimal", 4, "incident", 4, 4),
            (
                "actor-only-regression",
                "stock",
                4,
                "actor-only",
                "default",
                1,
            ),
            (
                "actor-only-regression",
                "minimal",
                4,
                "actor-only",
                "default",
                1,
            ),
        }
    assert cases == run_evidence.build_scale_evidence_cases(
        run_id="scale-unit", repetitions=5, seed=17
    )
    assert cases != run_evidence.build_scale_evidence_cases(
        run_id="scale-unit", repetitions=5, seed=18
    )


def test_dgx_scale_profile_builds_blocked_strong_weak_and_control_cases() -> None:
    cases = run_evidence.build_dgx_scale_cases(
        run_id="dgx-unit", repetitions=3, seed=23
    )

    assert len(cases) == 78
    assert all(
        len(
            {
                case.repetition
                for case in cases[index : index + run_evidence.DGX_CASES_PER_REPETITION]
            }
        )
        == 1
        for index in range(0, len(cases), run_evidence.DGX_CASES_PER_REPETITION)
    )
    for repetition in range(1, 4):
        repeated = [case for case in cases if case.repetition == repetition]
        assert {
            case.capacity
            for case in repeated
            if case.scaling_mode == "strong" and case.arm == "minimal"
        } == set(run_evidence.DGX_CAPACITIES)
        assert {
            case.capacity
            for case in repeated
            if case.scaling_mode == "strong" and case.arm == "stock"
        } == {4, 8, 16}
        stock_shapes = {
            (
                case.capacity,
                case.map_actors_per_stage,
                case.shuffle_ranks,
                case.workaround_id,
            )
            for case in repeated
            if case.scaling_mode == "strong" and case.arm == "stock"
        }
        assert stock_shapes == {
            (capacity, map_actors, shuffle_ranks, f"m{map_actors}-r{shuffle_ranks}")
            for capacity, values in run_evidence.DGX_STOCK_WORKAROUNDS.items()
            for map_actors, shuffle_ranks in values
        }
        assert len(stock_shapes) == 14
        assert all(
            3 * map_actors + shuffle_ranks <= capacity
            for capacity, map_actors, shuffle_ranks, _ in stock_shapes
        )
        candidate_strong = [
            case
            for case in repeated
            if case.scaling_mode == "strong" and case.arm == "minimal"
        ]
        assert all(
            case.shuffle_ranks == case.capacity
            and case.map_actors_per_stage == case.capacity
            and case.rows == run_evidence.SCALE_DEFAULT_ROWS
            and case.blocks == run_evidence.SCALE_DEFAULT_BLOCKS
            for case in candidate_strong
        )
        weak = [case for case in repeated if case.scaling_mode == "weak"]
        assert {case.capacity for case in weak} == set(run_evidence.DGX_CAPACITIES)
        assert all(
            case.rows == case.capacity * run_evidence.DGX_WEAK_ROWS_PER_GPU
            and case.blocks == case.capacity * run_evidence.DGX_WEAK_BLOCKS_PER_GPU
            and case.shuffle_ranks == case.capacity
            and case.map_actors_per_stage == case.capacity
            for case in weak
        )
        controls = [case for case in repeated if case.scaling_mode == "actor-control"]
        assert {(case.arm, case.capacity) for case in controls} == {
            ("stock", 16),
            ("minimal", 16),
        }
        assert all(
            case.workload == "actor-only"
            and case.shuffle_ranks == "default"
            and case.map_actors_per_stage == run_evidence.DGX_ACTOR_CONTROL_MAP_ACTORS
            for case in controls
        )
    assert cases == run_evidence.build_dgx_scale_cases(
        run_id="dgx-unit", repetitions=3, seed=23
    )
    assert cases != run_evidence.build_dgx_scale_cases(
        run_id="dgx-unit", repetitions=3, seed=24
    )


def test_dgx_weak_scaling_can_be_bounded_without_hiding_actual_work() -> None:
    cases = run_evidence.build_dgx_scale_cases(
        run_id="dgx-bounded",
        repetitions=1,
        seed=1,
        weak_max_rows=4_000_000_000,
    )
    weak = {
        case.capacity: (case.rows, case.blocks)
        for case in cases
        if case.scaling_mode == "weak"
    }

    assert weak == {
        1: (1_000_000_000, 256),
        2: (2_000_000_000, 512),
        4: (4_000_000_000, 1_024),
        8: (4_000_000_000, 1_024),
        16: (4_000_000_000, 1_024),
    }


def test_rank_filter_omits_values_above_capacity() -> None:
    assert run_evidence._ranks_for("incident", 2, [1, 2, 3, 4, "default"]) == [
        1,
        2,
        "default",
    ]
    assert run_evidence._ranks_for("aggregate-cpu-gap", 2, [1, 2, 3, 4, "default"]) == [
        1,
        2,
        "default",
    ]
    assert run_evidence._ranks_for("actor-only", 2, [1, 2]) == ["default"]


def test_gpu_indices_accept_zero() -> None:
    args = run_evidence.parse_args(
        [
            "--run-id",
            "unit",
            "--gpu-indices",
            "0,1,2,3",
            "--case-timeout-seconds",
            "45",
        ]
    )
    assert args.gpu_indices == [0, 1, 2, 3]


def test_execution_rejects_non_workspace_results_root(tmp_path) -> None:
    with pytest.raises(SystemExit):
        run_evidence.parse_args(
            [
                "--run-id",
                "unit",
                "--output-root",
                str(tmp_path),
                "--case-timeout-seconds",
                "45",
                "--execute-local",
            ]
        )


def test_curated_workaround_can_fix_map_actor_count() -> None:
    args = run_evidence.parse_args(
        [
            "--run-id",
            "unit",
            "--map-actors-per-stage",
            "1",
            "--case-timeout-seconds",
            "45",
        ]
    )

    assert args.map_actors_per_stage == 1


def test_curated_workaround_can_use_elastic_map_actor_range() -> None:
    args = run_evidence.parse_args(
        [
            "--run-id",
            "unit",
            "--map-actors-per-stage",
            "1",
            "--map-actors-max-per-stage",
            "16",
            "--case-timeout-seconds",
            "45",
        ]
    )

    assert args.map_actors_per_stage == 1
    assert args.map_actors_max_per_stage == 16


def test_map_actor_max_cannot_be_smaller_than_minimum() -> None:
    with pytest.raises(SystemExit):
        run_evidence.parse_args(
            [
                "--run-id",
                "unit",
                "--map-actors-per-stage",
                "2",
                "--map-actors-max-per-stage",
                "1",
                "--case-timeout-seconds",
                "45",
            ]
        )


def test_gpu_map_work_iterations_must_be_nonnegative() -> None:
    with pytest.raises(SystemExit):
        run_evidence.parse_args(
            [
                "--run-id",
                "unit",
                "--gpu-map-work-iterations",
                "-1",
                "--case-timeout-seconds",
                "45",
            ]
        )


def test_materialize_boundaries_are_opt_in() -> None:
    args = run_evidence.parse_args(
        [
            "--run-id",
            "unit",
            "--materialize-boundaries",
            "--case-timeout-seconds",
            "45",
        ]
    )

    assert args.materialize_boundaries


def test_scale_profile_defaults_to_five_large_repetitions() -> None:
    args = run_evidence.parse_args(
        [
            "--run-id",
            "unit",
            "--profile",
            "scale",
            "--case-timeout-seconds",
            "45",
        ]
    )

    assert args.repetitions == 5
    assert args.rows == run_evidence.SCALE_DEFAULT_ROWS
    assert args.blocks == run_evidence.SCALE_DEFAULT_BLOCKS


def test_scale_profile_rejects_global_map_actor_override() -> None:
    with pytest.raises(SystemExit):
        run_evidence.parse_args(
            [
                "--run-id",
                "unit",
                "--profile",
                "scale",
                "--map-actors-per-stage",
                "1",
                "--case-timeout-seconds",
                "45",
            ]
        )


def test_dgx_scale_defaults_to_one_screening_repetition() -> None:
    args = run_evidence.parse_args(
        [
            "--run-id",
            "unit",
            "--profile",
            "dgx-scale",
            "--case-timeout-seconds",
            "45",
        ]
    )

    assert args.repetitions == 1
    assert args.rows == run_evidence.SCALE_DEFAULT_ROWS
    assert args.blocks == run_evidence.SCALE_DEFAULT_BLOCKS
    assert args.dgx_weak_rows_per_gpu == run_evidence.DGX_WEAK_ROWS_PER_GPU


def test_dgx_scale_owns_its_factor_matrix() -> None:
    with pytest.raises(SystemExit):
        run_evidence.parse_args(
            [
                "--run-id",
                "unit",
                "--profile",
                "dgx-scale",
                "--capacities",
                "1,2,4",
                "--case-timeout-seconds",
                "45",
            ]
        )


def _scale_execution(
    *,
    repetition: int,
    comparison_id: str,
    arm: str,
    elapsed_s: float,
) -> dict:
    return {
        "outcome": "success",
        "case": {
            "case_id": f"{comparison_id}-{arm}-{repetition}",
            "comparison_id": comparison_id,
            "arm": arm,
            "repetition": repetition,
        },
        "workload": {
            "elapsed_s": elapsed_s,
            "output_rows": 100,
            "output_schema_hash": "schema",
            "output_digest": f"digest-{comparison_id}-{repetition}",
        },
        "job_cleanup": {"cleanup_proven": True},
        "cluster_cleanup": {
            "cleanup_proven": True,
            "ray_spill_directory_removed": True,
        },
    }


def test_scale_report_pairs_repetitions_and_applies_both_performance_gates() -> None:
    executions = []
    for repetition in range(1, 6):
        executions.extend(
            [
                _scale_execution(
                    repetition=repetition,
                    comparison_id="incident-scale-speedup",
                    arm="stock",
                    elapsed_s=10.0,
                ),
                _scale_execution(
                    repetition=repetition,
                    comparison_id="incident-scale-speedup",
                    arm="minimal",
                    elapsed_s=8.0,
                ),
                _scale_execution(
                    repetition=repetition,
                    comparison_id="actor-only-regression",
                    arm="stock",
                    elapsed_s=10.0,
                ),
                _scale_execution(
                    repetition=repetition,
                    comparison_id="actor-only-regression",
                    arm="minimal",
                    elapsed_s=10.4,
                ),
            ]
        )

    report = run_evidence.build_scale_evidence_report(
        executions=executions,
        repetitions=5,
        seed=9,
        executed=True,
    )

    assert report["status"] == "pass"
    assert report["blockers"] == []
    assert report["incident_speedup"]["best_stock_workaround"] == (
        "map-actors-1_shuffle-rank-1"
    )
    assert report["incident_speedup"]["comparison"]["mode"] == ("paired_by_repetition")
    assert report["actor_only_regression"]["comparison"]["mode"] == (
        "paired_by_repetition"
    )
    assert report["gates"]["incident_at_least_10_percent_faster"] is True
    assert (
        report["gates"]["equal_shape_actor_only_regression_at_most_5_percent"] is True
    )
    assert (
        report["comparison_contract"]["actor_only_equal_shape"]["map_actors_per_stage"]
        == 1
    )


def _dgx_execution(case: run_evidence.LocalCase) -> dict:
    if case.scaling_mode == "strong":
        candidate_elapsed = 160.0 / case.capacity
        fastest_stock = f"m{(case.capacity - 1) // 3}-r1"
        stock_penalty = 1.0 if case.workaround_id == fastest_stock else 1.1
        elapsed = (
            candidate_elapsed
            if case.arm == "minimal"
            else candidate_elapsed * 1.25 * stock_penalty
        )
        digest = f"strong-{case.repetition}"
    elif case.scaling_mode == "weak":
        elapsed = 100.0
        digest = f"weak-g{case.capacity}-n{case.repetition}"
    else:
        elapsed = 102.0 if case.arm == "minimal" else 100.0
        digest = f"actor-{case.repetition}"
    premature = case.scaling_mode == "strong" and case.arm == "stock"
    assert case.rows is not None
    return {
        "outcome": "success",
        "case": {
            **case.__dict__,
            "map_actors_per_stage": case.map_actors_per_stage,
        },
        "object_store_memory_bytes": (
            run_evidence.DGX_OBJECT_STORE_BYTES_PER_GPU * case.capacity
        ),
        "workload": {
            "elapsed_s": elapsed,
            "input_rows_per_second": case.rows / elapsed,
            "output_rows": (
                case.rows
                if case.scaling_mode == "actor-control"
                else min(case.rows, 64)
            ),
            "output_schema_hash": "schema",
            "output_digest": digest,
            "global_bytes_spilled": case.capacity * 1_000,
            "global_bytes_restored": case.capacity * 500,
            "resource_metrics": {
                "cluster_gpu_seconds": elapsed * case.capacity,
                "owned_gpu_seconds": elapsed * case.capacity * 0.9,
                "premature_downstream_ownership": {
                    "measurement_complete": True,
                    "premature_downstream_actor_count": 2 if premature else 0,
                    "premature_downstream_ownership_seconds": (
                        40.0 if premature else 0.0
                    ),
                    "premature_downstream_gpu_seconds": (40.0 if premature else 0.0),
                    "mean_ready_to_stage_first_input_s": (20.0 if premature else 0.0),
                    "peak_premature_downstream_gpus": 2 if premature else 0,
                    "premature_gpu_seconds_fraction_of_cluster": (
                        40.0 / (elapsed * case.capacity) if premature else 0.0
                    ),
                    "premature_downstream_gpu_seconds_during_earlier_stage": (
                        30.0 if premature else 0.0
                    ),
                    "peak_premature_downstream_gpus_during_earlier_stage": (
                        2 if premature else 0
                    ),
                    "gcs_corroboration_complete": True,
                    "gcs_state_samples_valid": True,
                    "gcs_corroborated_actor_count": 2 if premature else 0,
                    "gcs_expected_actor_count": 2 if premature else 0,
                    "common_clock_verified": True,
                    "clock_basis": "single-boot-monotonic",
                    "telemetry_quiescence": {"complete": True},
                    "per_stage": (
                        [
                            {
                                "stage": "gpu-map-groups-1",
                                "operator": "SumGroup/map_groups",
                                "actor_count": 1,
                                "gpu_seconds": 20.0,
                                "cluster_gpu_seconds_fraction": (
                                    20.0 / (elapsed * case.capacity)
                                ),
                            },
                            {
                                "stage": "gpu-map-final",
                                "operator": "Identity/map_batches",
                                "actor_count": 1,
                                "gpu_seconds": 20.0,
                                "cluster_gpu_seconds_fraction": (
                                    20.0 / (elapsed * case.capacity)
                                ),
                            },
                        ]
                        if premature
                        else []
                    ),
                },
            },
        },
        "job_cleanup": {"cleanup_proven": True},
        "cluster_cleanup": {
            "cleanup_proven": True,
            "ray_spill_directory_removed": True,
        },
    }


def test_dgx_report_proves_speedup_efficiency_telemetry_and_control() -> None:
    cases = run_evidence.build_dgx_scale_cases(
        run_id="dgx-report", repetitions=5, seed=29
    )
    report = run_evidence.build_dgx_scale_report(
        executions=[_dgx_execution(case) for case in cases],
        repetitions=5,
        seed=29,
        executed=True,
        artifacts_ready=True,
        sixteen_physical_gpus_visible=True,
    )

    assert report["status"] == "pass"
    assert report["blockers"] == []
    assert report["expected_cases"] == 130
    assert report["correctness"]["passed"] is True
    assert len(report["configuration_statistics"]) == 26
    for capacity in (4, 8, 16):
        comparison = report["strong_scaling"]["candidate_vs_stock_by_capacity"][
            str(capacity)
        ]
        assert comparison["passed"] is True
        assert comparison["comparison"]["mode"] == "paired_by_repetition"
        assert comparison["best_stock_workaround"] == (f"m{(capacity - 1) // 3}-r1")
        assert (
            comparison["top_two_stock_workarounds"][0]["workaround_id"]
            == comparison["best_stock_workaround"]
        )
    candidate_curve = report["strong_scaling"]["candidate_curve"]
    assert candidate_curve["baseline_capacity"] == 1
    assert candidate_curve["points"][-1]["speedup_vs_baseline"] == 16
    assert candidate_curve["points"][-1]["parallel_efficiency"] == 1
    weak_curve = report["weak_scaling"]["candidate_curve"]
    assert weak_curve["points"][-1]["parallel_efficiency"] == 1
    assert report["actor_only_regression"]["passed"] is True
    assert (
        report["comparison_contract"]["actor_only_equal_shape"]["map_actors_per_stage"]
        == run_evidence.DGX_ACTOR_CONTROL_MAP_ACTORS
    )
    causal = report["premature_downstream_ownership"]["candidate_vs_stock_by_capacity"][
        "16"
    ]
    assert causal["candidate_reduced_premature_gpu_seconds"] is True
    assert (
        causal["candidate_reduced_premature_gpu_overlap_during_earlier_stage"] is True
    )
    assert (
        causal["metrics"]["premature_downstream_gpu_seconds"]["median_avoided"] == 40.0
    )


def test_dgx_report_fails_closed_on_corrupt_evidence_contracts() -> None:
    cases = run_evidence.build_dgx_scale_cases(
        run_id="dgx-fail-closed", repetitions=5, seed=31
    )
    executions = [_dgx_execution(case) for case in cases]

    executions[0]["object_store_memory_bytes"] -= 1
    executions[1]["cluster_cleanup"]["ray_spill_directory_removed"] = False
    strong = next(
        execution
        for execution in executions
        if execution["case"]["scaling_mode"] == "strong"
    )
    strong["workload"]["resource_metrics"]["premature_downstream_ownership"][
        "measurement_complete"
    ] = False
    strong["workload"]["output_digest"] = "corrupt"
    for execution in executions:
        case = execution["case"]
        if (
            case["scaling_mode"] == "strong"
            and case["arm"] == "minimal"
            and case["capacity"] == 8
        ):
            execution["workload"]["elapsed_s"] = 1_000.0
        if case["scaling_mode"] == "actor-control" and case["arm"] == "minimal":
            execution["workload"]["elapsed_s"] = 107.0

    report = run_evidence.build_dgx_scale_report(
        executions=executions,
        repetitions=5,
        seed=31,
        executed=True,
        artifacts_ready=True,
        sixteen_physical_gpus_visible=True,
    )

    assert report["status"] == "fail"
    for gate in (
        "every_case_used_8gib_object_store_per_gpu",
        "all_job_and_cluster_cleanup_proven",
        "all_downstream_actor_stage_and_first_input_measurements_present",
        "all_correctness_oracles_agree",
        "candidate_at_least_10_percent_faster_at_4_8_16",
        "equal_shape_actor_only_regression_at_most_5_percent",
    ):
        assert report["gates"][gate] is False
        assert gate in report["blockers"]


def test_staged_harness_is_content_addressed_read_only_and_verified(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    (source_root / "nested").mkdir(parents=True)
    (source_root / "runner.py").write_text("print('runner')\n")
    (source_root / "nested" / "probe.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(run_evidence, "ROOT", source_root)
    monkeypatch.setattr(
        run_evidence,
        "LOCAL_STAGED_HARNESS_FILES",
        ("runner.py", "nested/probe.py"),
    )

    staged = run_evidence.stage_local_harness(tmp_path / "run")
    directory = run_evidence.Path(staged["directory"])
    assert staged["file_count"] == 2
    assert directory.name.startswith("harness-")
    assert all(
        path.stat().st_mode & 0o222 == 0
        for path in (
            directory / "runner.py",
            directory / "nested" / "probe.py",
            directory / run_evidence.LOCAL_STAGED_HARNESS_MANIFEST,
        )
    )
    assert (
        run_evidence._verify_local_harness(directory, str(staged["content_sha256"]))[
            "content_sha256"
        ]
        == staged["content_sha256"]
    )

    staged_runner = directory / "runner.py"
    staged_runner.chmod(0o644)
    staged_runner.write_text("print('mutated')\n")
    with pytest.raises(ValueError, match="failed verification"):
        run_evidence._verify_local_harness(directory, str(staged["content_sha256"]))


def test_started_run_is_immutable_even_for_later_dry_run(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "immutable"
    run_dir.mkdir()
    (run_dir / "execution-started.json").write_text("{}\n")
    monkeypatch.setattr(
        run_evidence,
        "stage_local_harness",
        lambda _: pytest.fail("immutability must be checked before staging"),
    )

    with pytest.raises(RuntimeError, match="already started and is immutable"):
        run_evidence.main(
            [
                "--run-id",
                "immutable",
                "--output-root",
                str(tmp_path),
                "--case-timeout-seconds",
                "45",
            ]
        )


def test_bounded_dgx_weak_curve_is_not_labeled_full_evidence() -> None:
    report = run_evidence.build_dgx_scale_report(
        executions=[],
        repetitions=5,
        seed=1,
        executed=False,
        artifacts_ready=True,
        sixteen_physical_gpus_visible=True,
        weak_max_rows=4_000_000_000,
    )

    assert report["dataset_contract"]["evidence_shape"] is False
    assert (
        report["gates"]["weak_dataset_is_unbounded_1b_rows_and_256_blocks_per_gpu"]
        is False
    )


def test_dgx_dry_run_writes_26_case_plan_without_starting_ray(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        run_evidence,
        "gpu_inventory",
        lambda: {
            "status": "ready",
            "devices": [{"index": index} for index in range(16)],
        },
    )
    monkeypatch.setattr(
        run_evidence,
        "inspect_artifacts",
        lambda arms: {"ready": True, "blockers": [], "arms": {arm: {} for arm in arms}},
    )
    monkeypatch.setattr(
        run_evidence,
        "execute_case",
        lambda **_: pytest.fail("dry run must not execute a case"),
    )

    assert (
        run_evidence.main(
            [
                "--run-id",
                "dgx-plan",
                "--output-root",
                str(tmp_path),
                "--profile",
                "dgx-scale",
                "--case-timeout-seconds",
                "45",
            ]
        )
        == 0
    )
    plan = json.loads((tmp_path / "dgx-plan" / "plan.json").read_text())
    assert plan["mode"] == "dry-run"
    assert plan["capacities"] == [1, 2, 4, 8, 16]
    assert plan["case_count"] == 26
    assert plan["dgx_scale_contract"]["cases_per_repetition"] == 26
    assert plan["dgx_scale_contract"]["actor_control_map_actors_per_stage"] == 5
    assert plan["parameters"]["rows"] == "per-case"
    assert plan["safety"]["ray_tmp_root"] == "/dev/shm/ray-admission"
    assert plan["safety"]["ray_object_spilling_leaf"] == ("<case-directory>/ray-spill")
    assert plan["safety"]["spill_removal_required_for_cleanup"] is True
    report = json.loads(
        (tmp_path / "dgx-plan" / "dgx-scale-evidence-report.json").read_text()
    )
    assert report["status"] == "not_run"


def test_structural_timeout_outcome_requires_classifier() -> None:
    command = {"timed_out": True}
    structural = {
        "resource_metrics": {"closed_wait": {"is_structural_closed_wait": True}}
    }
    steady = {"resource_metrics": {"closed_wait": {"is_structural_closed_wait": False}}}
    assert (
        run_evidence._outcome(structural, command) == "structural_closed_wait_timeout"
    )
    assert run_evidence._outcome(steady, command) == "timeout_unclassified"


def test_timeout_metrics_are_finalized_from_persisted_samples() -> None:
    samples = [
        {
            "elapsed_s": elapsed,
            "alive_nodes": 1,
            "cluster": {"GPU": 1},
            "available": {},
            "progress": {"sequence": 0, "stages": {}},
            "gpu_telemetry": [],
            "ray_state": {
                "actors": [
                    {
                        "actor_id": "owner",
                        "state": "ALIVE",
                        "required_resources": {"GPU": 1},
                    },
                    {
                        "actor_id": "waiter",
                        "state": "PENDING_CREATION",
                        "required_resources": {"GPU": 1},
                    },
                ],
                "tasks": [],
                "placement_groups": [],
            },
        }
        for elapsed in (0, 16, 32)
    ]

    workload = run_evidence._finalize_timeout_metrics(
        {"status": "materializing", "resource_samples": samples}, max_gpus=1
    )

    assert workload["timeout_metrics_finalized_by"] == "local-evidence-runner"
    assert workload["resource_metrics"]["structural_closed_wait"] is True
    assert run_evidence._outcome(workload, {"timed_out": True}) == (
        "structural_closed_wait_timeout"
    )


def test_final_artifact_provenance_separates_source_from_installed_base() -> None:
    readiness = run_evidence.inspect_artifacts(list(run_evidence.ARM_NAMES))
    assert readiness["ready"], readiness["blockers"]
    reports = readiness["arms"]

    stock = reports["stock"]
    assert stock["source_commit"] == stock["base_commit"]
    assert stock["installed_ray_commit"] == stock["base_commit"]

    for name in ("pg-only", "minimal", "prototype"):
        derived = reports[name]
        assert derived["source_commit"] != derived["base_commit"]
        assert derived["installed_ray_commit"] == derived["base_commit"]

    assert reports["minimal"]["wheel_sha256"] != reports["prototype"]["wheel_sha256"]


def test_runtime_probe_accepts_separate_source_and_installed_commits(tmp_path) -> None:
    source = "1" * 40
    base = "2" * 40
    args = probe.parse_args(
        [
            "--mode",
            "runtime-gpu",
            "--result",
            str(tmp_path / "result.json"),
            "--arm",
            "minimal",
            "--capacity",
            "1",
            "--wheel",
            str(tmp_path / "ray.whl"),
            "--wheel-sha256",
            "3" * 64,
            "--source-commit",
            source,
            "--base-commit",
            base,
            "--installed-ray-commit",
            base,
            "--overlay",
            str(tmp_path / "overlay"),
        ]
    )
    assert args.source_commit == source
    assert args.base_commit == base
    assert args.installed_ray_commit == base


def test_safe_tmp_removal_rejects_broad_target(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="refusing"):
        run_evidence._safe_remove_case_tmp(tmp_path)


def _spill_test_paths(tmp_path):
    results_root = tmp_path / "benchmark" / "results" / "local"
    case_dir = results_root / "unit-run" / "cases" / "0001-unit-case"
    case_dir.mkdir(parents=True)
    return results_root, case_dir, case_dir / "ray-spill"


def test_spill_directory_must_be_exact_absent_case_leaf(tmp_path) -> None:
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)

    assert (
        cluster_daemon.validate_object_spilling_directory(
            spill_dir,
            case_directory=case_dir,
            results_root=results_root,
            must_exist=False,
        )
        == spill_dir
    )
    with pytest.raises(ValueError, match="exactly"):
        cluster_daemon.validate_object_spilling_directory(
            case_dir / "not-the-spill-leaf",
            case_directory=case_dir,
            results_root=results_root,
            must_exist=False,
        )
    with pytest.raises(ValueError, match="must be absolute"):
        cluster_daemon.validate_object_spilling_directory(
            Path("ray-spill"),
            case_directory=case_dir,
            results_root=results_root,
            must_exist=False,
        )

    spill_dir.mkdir()
    with pytest.raises(ValueError, match="must not preexist"):
        cluster_daemon.validate_object_spilling_directory(
            spill_dir,
            case_directory=case_dir,
            results_root=results_root,
            must_exist=False,
        )


def test_spill_directory_rejects_non_results_root(tmp_path) -> None:
    results_root = tmp_path / "somewhere" / "else"
    case_dir = results_root / "unit-run" / "cases" / "0001-unit-case"
    case_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="benchmark/results/local"):
        cluster_daemon.validate_object_spilling_directory(
            case_dir / "ray-spill",
            case_directory=case_dir,
            results_root=results_root,
            must_exist=False,
        )


def test_cluster_daemon_accepts_exact_absent_spill_argument(tmp_path) -> None:
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    args = cluster_daemon.parse_args(
        [
            "--capacity",
            "1",
            "--num-cpus",
            "1",
            "--object-store-memory",
            str(75 * 1024**2),
            "--tmp-dir",
            "/dev/shm/ray-admission/lunit-parse",
            "--object-spilling-directory",
            str(spill_dir),
            "--case-directory",
            str(case_dir),
            "--results-root",
            str(results_root),
            "--ready",
            str(case_dir / "ready.json"),
            "--stop-request",
            str(case_dir / "stop.request"),
            "--stopped",
            str(case_dir / "stopped.json"),
        ]
    )

    assert args.object_spilling_directory == spill_dir
    assert not spill_dir.exists()


def test_cluster_daemon_uses_keyword_accepted_by_every_built_arm() -> None:
    daemon_tree = ast.parse(Path(cluster_daemon.__file__).read_text())
    ray_init_keywords = {
        keyword.arg
        for node in ast.walk(daemon_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ray"
        and node.func.attr == "init"
        for keyword in node.keywords
    }
    assert "object_spilling_directory" in ray_init_keywords
    assert "_object_spilling_directory" not in ray_init_keywords

    for layer in {arm.wheel_layer for arm in run_evidence.ARMS.values()}:
        wheels = sorted((run_evidence.ROOT / "wheels" / layer).glob("ray-*.whl"))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0]) as archive:
            worker_tree = ast.parse(
                archive.read("ray/_private/worker.py").decode("utf-8")
            )
        popped_kwargs = {
            node.args[0].value
            for node in ast.walk(worker_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        assert "object_spilling_directory" in popped_kwargs, layer


def test_safe_spill_removal_rejects_symlink_leaf(tmp_path) -> None:
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep")
    spill_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        run_evidence._safe_remove_case_spill(
            spill_dir,
            case_dir=case_dir,
            results_root=results_root,
        )
    assert sentinel.read_text() == "keep"


def test_safe_spill_removal_rejects_parent_swap(tmp_path, monkeypatch) -> None:
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    spill_dir.mkdir()
    owned_sentinel = spill_dir / "owned"
    owned_sentinel.write_text("keep-owned")
    replacement_case = tmp_path / "replacement-case"
    replacement_spill = replacement_case / "ray-spill"
    replacement_spill.mkdir(parents=True)
    replacement_sentinel = replacement_spill / "replacement"
    replacement_sentinel.write_text("keep-replacement")
    renamed_case = case_dir.with_name("0001-renamed-owned-case")
    real_validate = run_evidence.validate_object_spilling_directory

    def validate_then_swap(*args, **kwargs):
        validated = real_validate(*args, **kwargs)
        case_dir.rename(renamed_case)
        replacement_case.rename(case_dir)
        return validated

    monkeypatch.setattr(
        run_evidence,
        "validate_object_spilling_directory",
        validate_then_swap,
    )
    with pytest.raises(RuntimeError, match="changed"):
        run_evidence._safe_remove_case_spill(
            spill_dir,
            case_dir=case_dir,
            results_root=results_root,
        )
    assert (renamed_case / "ray-spill" / "owned").read_text() == "keep-owned"
    assert (case_dir / "ray-spill" / "replacement").read_text() == ("keep-replacement")


def test_spill_snapshot_records_backing_storage_and_physical_files(tmp_path) -> None:
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    before = cluster_daemon.snapshot_object_spilling_directory(
        spill_dir,
        case_directory=case_dir,
        results_root=results_root,
    )
    assert before["exists"] is False
    assert before["spill_file_count"] == 0
    assert before["spill_file_bytes"] == 0

    nested = spill_dir / "nested"
    nested.mkdir(parents=True)
    (spill_dir / "first").write_bytes(b"1234")
    (nested / "second").write_bytes(b"567")
    after = cluster_daemon.snapshot_object_spilling_directory(
        spill_dir,
        case_directory=case_dir,
        results_root=results_root,
    )

    assert after["exists"] is True
    assert after["path"] == str(spill_dir)
    assert after["spill_file_count"] == 2
    assert after["spill_file_bytes"] == 7
    assert isinstance(after["filesystem_device"], str)
    assert after["filesystem_device"]
    assert isinstance(after["filesystem_device_id"], int)
    assert isinstance(after["filesystem_type"], str)
    assert after["filesystem_type"]
    assert after["available_bytes"] > 0
    assert after["scan_race_count"] == 0
    assert after["scan_complete"] is True


def test_live_spill_snapshot_labels_a_file_deletion_race(tmp_path, monkeypatch) -> None:
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    spill_dir.mkdir()
    monkeypatch.setattr(
        cluster_daemon.os,
        "walk",
        lambda *_args, **_kwargs: [(str(spill_dir), [], ["vanished"])],
    )

    snapshot = cluster_daemon.snapshot_object_spilling_directory(
        spill_dir,
        case_directory=case_dir,
        results_root=results_root,
    )

    assert snapshot["spill_file_count"] == 0
    assert snapshot["spill_file_bytes"] == 0
    assert snapshot["scan_race_count"] == 1
    assert snapshot["scan_complete"] is False


def test_live_spill_snapshot_labels_a_disappearing_subdirectory(
    tmp_path, monkeypatch
) -> None:
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    spill_dir.mkdir()

    def disappearing_walk(*_args, **kwargs):
        kwargs["onerror"](FileNotFoundError("spill subtree vanished"))
        return []

    monkeypatch.setattr(cluster_daemon.os, "walk", disappearing_walk)
    snapshot = cluster_daemon.snapshot_object_spilling_directory(
        spill_dir,
        case_directory=case_dir,
        results_root=results_root,
    )

    assert snapshot["scan_race_count"] == 1
    assert snapshot["scan_complete"] is False


def test_failed_startup_with_no_owned_processes_proves_cleanup(
    tmp_path, monkeypatch
) -> None:
    class Daemon:
        returncode = 1
        pid = 10_000_001

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(run_evidence, "_process_group_is_gone", lambda _pgid: True)
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    spill_dir.mkdir()
    (spill_dir / "physical-spill").write_bytes(b"spill")
    stopped = case_dir / "stopped.json"
    stopped.write_text(
        json.dumps(
            {
                "status": "stopped",
                "owned_processes": [],
                "owned_processes_alive_after_shutdown": [],
            }
        )
    )
    report = run_evidence._stop_cluster(
        daemon=Daemon(),
        daemon_log=io.StringIO(),
        stop_request=case_dir / "stop.request",
        stopped_path=stopped,
        ready={"status": "error", "owned_processes": []},
        tmp_dir=run_evidence.SHM_ROOT / "lunit-never-started",
        spill_dir=spill_dir,
        case_dir=case_dir,
        results_root=results_root,
    )

    assert report["cleanup_proven"] is True
    assert report["gcs_address_closed"] is True
    assert report["ray_spill_storage_before_shutdown"]["spill_file_count"] == 1
    assert report["ray_spill_storage_before_shutdown"]["spill_file_bytes"] == 5
    assert report["ray_spill_directory_removed"] is True
    assert not spill_dir.exists()


def test_cluster_cleanup_does_not_remove_spill_while_owned_process_is_alive(
    tmp_path, monkeypatch
) -> None:
    class Daemon:
        returncode = 0
        pid = 10_000_002

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(run_evidence, "_process_group_is_gone", lambda _pgid: True)
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    spill_dir.mkdir()
    stopped = case_dir / "stopped.json"
    stopped.write_text(
        json.dumps(
            {
                "status": "error",
                "owned_processes": [],
                "owned_processes_alive_after_shutdown": [],
            }
        )
    )

    report = run_evidence._stop_cluster(
        daemon=Daemon(),
        daemon_log=io.StringIO(),
        stop_request=case_dir / "stop.request",
        stopped_path=stopped,
        ready={
            "status": "ready",
            "address": "127.0.0.1:1",
            "owned_processes": [{"kind": "raylet", "pid": os.getpid()}],
        },
        tmp_dir=run_evidence.SHM_ROOT / "lunit-still-alive",
        spill_dir=spill_dir,
        case_dir=case_dir,
        results_root=results_root,
    )

    assert report["cleanup_proven"] is False
    assert report["ray_spill_directory_removed"] is False
    assert spill_dir.is_dir()


def test_cluster_cleanup_keeps_spill_while_daemon_process_group_exists(
    tmp_path, monkeypatch
) -> None:
    class Daemon:
        returncode = 0
        pid = 10_000_003

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(run_evidence, "_process_group_is_gone", lambda _pgid: False)
    results_root, case_dir, spill_dir = _spill_test_paths(tmp_path)
    spill_dir.mkdir()
    stopped = case_dir / "stopped.json"
    stopped.write_text(
        json.dumps(
            {
                "status": "stopped",
                "owned_processes": [],
                "owned_processes_alive_after_shutdown": [],
            }
        )
    )

    report = run_evidence._stop_cluster(
        daemon=Daemon(),
        daemon_log=io.StringIO(),
        stop_request=case_dir / "stop.request",
        stopped_path=stopped,
        ready={"status": "ready", "address": "127.0.0.1:1"},
        tmp_dir=run_evidence.SHM_ROOT / "lunit-group-alive",
        spill_dir=spill_dir,
        case_dir=case_dir,
        results_root=results_root,
    )

    assert report["cleanup_proven"] is False
    assert report["daemon_process_group_gone"] is False
    assert report["ray_spill_directory_removed"] is False
    assert spill_dir.is_dir()


def test_dry_prerequisite_report_is_hard_blocked() -> None:
    plan = {
        "artifact_readiness": {"ready": True},
        "gpu_inventory": {"devices": [{}, {}, {}, {}]},
        "capacities": [1, 2, 4],
    }
    report = run_evidence.build_prerequisite_report(
        plan=plan, executions=[], executed=False
    )
    assert report["status"] == "not_ready_for_cloud"
    assert "executed" in report["blockers"]


def test_autoscaling_simulation_defaults_to_dry_run(tmp_path) -> None:
    result = tmp_path / "simulation.json"
    assert simulate_autoscaling.main(["--result", str(result)]) == 0
    document = json.loads(result.read_text())
    assert document["status"] == "dry_run"
    assert document["will_start_ray"] is False
    assert "cannot replace" in document["disclaimer"]
