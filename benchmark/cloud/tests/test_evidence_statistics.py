from __future__ import annotations

import json
import math

import pytest

from benchmark import evidence_statistics as evidence


def runs(values: list[float], repetitions: list[int] | None = None) -> list[dict]:
    if repetitions is None:
        repetitions = list(range(len(values)))
    return [
        {"repetition": repetition, "elapsed_s": value}
        for repetition, value in zip(repetitions, values, strict=True)
    ]


def compare(
    candidate: list[float], baseline: list[float]
) -> dict[str, object]:
    return evidence.compare_randomized_runs(
        runs(candidate),
        runs(baseline),
        seed=17,
        resamples=500,
    )


def test_sample_summary_has_robust_statistics_and_seeded_ci() -> None:
    first = evidence.summarize_samples(
        [1, 2, 2, 4, 10], seed=91, resamples=500
    )
    second = evidence.summarize_samples(
        [1, 2, 2, 4, 10], seed=91, resamples=500
    )

    assert first == second
    assert first["samples"] == 5
    assert first["median"] == 2
    assert first["median_absolute_deviation"] == 1
    assert first["min"] == 1
    assert first["max"] == 10
    assert first["median_ci"]["status"] == "available"
    assert first["median_ci"]["completed_resamples"] == 500
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("values", ([], [1, 2, 3, 4]))
def test_sample_summary_marks_small_n_ci_as_insufficient(
    values: list[int],
) -> None:
    summary = evidence.summarize_samples(values, seed=3, resamples=100)

    assert summary["median_ci"]["status"] == "insufficient_samples"
    assert summary["median_ci"]["lower"] is None
    assert summary["median_ci"]["upper"] is None
    assert summary["median_ci"]["actual_samples"] == len(values)


def test_aligned_repetitions_use_paired_effects_even_when_order_differs() -> None:
    candidate = runs([8, 18, 27, 36, 45], [0, 1, 2, 3, 4])
    candidate.reverse()
    baseline = runs([10, 20, 30, 40, 50])

    result = evidence.compare_randomized_runs(
        candidate, baseline, seed=4, resamples=500
    )

    assert result["mode"] == "paired_by_repetition"
    assert result["pairing"] == {
        "key": "repetition",
        "paired_runs": 5,
        "reason_unpaired": None,
    }
    assert [row["repetition"] for row in result["per_repetition"]] == list(
        range(5)
    )
    effects = result["effects"]
    assert effects["baseline_minus_candidate"]["estimate"] == 3
    assert math.isclose(
        effects["baseline_over_candidate"]["estimate"], 10 / 9
    )
    assert math.isclose(
        effects["relative_time_reduction"]["estimate"], 0.1
    )
    assert effects["relative_time_reduction"]["samples"] == 5
    assert effects["relative_time_reduction"]["median_absolute_deviation"] == 0
    assert math.isclose(effects["relative_time_reduction"]["min"], 0.1)
    assert math.isclose(effects["relative_time_reduction"]["max"], 0.2)
    assert effects["relative_time_reduction"]["ci"]["status"] == "available"
    json.dumps(result, allow_nan=False)


def test_mismatched_repetitions_use_clearly_labeled_unpaired_fallback() -> None:
    result = evidence.compare_randomized_runs(
        runs([8, 9, 10, 11, 12], [0, 1, 2, 3, 4]),
        runs([10, 11, 12, 13, 14], [1, 2, 3, 4, 5]),
        seed=5,
        resamples=500,
    )

    assert result["mode"] == "unpaired"
    assert result["pairing"]["paired_runs"] == 0
    assert "do not match exactly" in result["pairing"]["reason_unpaired"]
    assert result["per_repetition"] == []
    assert result["effects"]["baseline_minus_candidate"]["estimate"] == 2
    assert result["effects"]["baseline_minus_candidate"]["samples"] is None
    assert result["effects"]["baseline_minus_candidate"]["estimate_kind"] == (
        "effect_of_group_medians"
    )
    assert result["effects"]["relative_time_reduction"]["ci"]["method"] == (
        "unpaired_percentile_bootstrap_relative_time_reduction"
    )


def test_missing_repetition_key_uses_unpaired_fallback() -> None:
    candidate = [{"elapsed_s": value} for value in [8, 8, 8, 8, 8]]
    result = evidence.compare_randomized_runs(
        candidate, runs([10, 10, 10, 10, 10]), resamples=100
    )

    assert result["mode"] == "unpaired"
    assert "no 'repetition'" in result["pairing"]["reason_unpaired"]


def test_unpaired_small_groups_do_not_claim_an_effect_ci() -> None:
    result = evidence.compare_randomized_runs(
        runs([8, 9], [0, 1]), runs([10, 11], [1, 2]), resamples=100
    )

    relative_ci = result["effects"]["relative_time_reduction"]["ci"]
    assert relative_ci["status"] == "insufficient_samples"
    assert relative_ci["actual_samples"] == {"candidate": 2, "baseline": 2}


def test_invalid_or_duplicate_duration_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate candidate_runs repetition"):
        evidence.compare_randomized_runs(
            runs([8, 9], [1, 1]), runs([10, 11]), resamples=10
        )
    with pytest.raises(ValueError, match="greater than zero"):
        evidence.compare_randomized_runs(
            runs([0], [0]), runs([10], [0]), resamples=10
        )


def test_regression_gate_allows_at_most_five_percent() -> None:
    passing = evidence.regression_gate(
        compare([10.4] * 5, [10.0] * 5), max_regression=0.05
    )
    failing = evidence.regression_gate(
        compare([10.6] * 5, [10.0] * 5), max_regression=0.05
    )

    assert passing["status"] == "pass"
    assert passing["passed"] is True
    assert math.isclose(passing["observed_regression"], 0.04)
    assert failing["status"] == "fail"
    assert failing["passed"] is False


def test_gates_refuse_to_pass_with_too_few_runs() -> None:
    comparison = evidence.compare_randomized_runs(
        runs([5, 5]), runs([10, 10]), resamples=100
    )

    gate = evidence.speedup_gate(comparison, minimum_speedup=0.10)

    assert gate["point_estimate_passed"] is True
    assert gate["status"] == "insufficient_samples"
    assert gate["passed"] is None


def test_confidence_bound_can_be_required_for_gate() -> None:
    comparison = compare([8.0] * 5, [10.0] * 5)

    gate = evidence.speedup_gate(
        comparison, minimum_speedup=0.10, require_confidence=True
    )

    assert gate["decision_basis"] == "lower_confidence_bound"
    assert gate["confidence_supported"] is True
    assert gate["passed"] is True


def test_speedup_gate_uses_fastest_completing_stock_workaround() -> None:
    result = evidence.speedup_vs_best_completing_stock(
        runs([8.5] * 5),
        {
            "rank-1": runs([12.0] * 5),
            "rank-2": runs([10.0] * 5),
            "rank-4-timeout": [],
        },
        seed=8,
        resamples=500,
    )

    assert result["best_stock_workaround"] == "rank-2"
    assert result["stock_medians"] == {"rank-1": 12.0, "rank-2": 10.0}
    assert result["excluded_stock_workarounds"] == ["rank-4-timeout"]
    assert math.isclose(result["observed_relative_time_reduction"], 0.15)
    assert result["status"] == "pass"
    assert result["passed"] is True
    json.dumps(result, allow_nan=False)


def test_speedup_gate_reports_when_no_stock_workaround_completes() -> None:
    result = evidence.speedup_vs_best_completing_stock(
        runs([8.0] * 5), {"rank-1": [], "rank-4": []}, resamples=100
    )

    assert result["status"] == "no_completing_stock_workaround"
    assert result["passed"] is None
    assert result["best_stock_workaround"] is None
