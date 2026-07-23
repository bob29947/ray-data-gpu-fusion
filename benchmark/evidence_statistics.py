"""Dependency-free statistics for randomized GPU admission benchmarks.

The comparison functions in this module treat shorter durations as better.  A
positive relative time reduction means that the candidate is faster than the
baseline.  Callers should pass completed runs only; failed or timed-out runs
belong in the liveness report rather than in duration statistics.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from numbers import Real

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_MIN_CI_SAMPLES = 5
DEFAULT_CONFIDENCE_LEVEL = 0.95


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_bootstrap_options(
    *, confidence_level: float, resamples: int, min_ci_samples: int
) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if (
        isinstance(min_ci_samples, bool)
        or not isinstance(min_ci_samples, int)
        or min_ci_samples < 2
    ):
        raise ValueError("min_ci_samples must be an integer of at least 2")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def _insufficient_ci(
    *,
    method: str,
    confidence_level: float,
    resamples: int,
    min_ci_samples: int,
    actual_samples: int | Mapping[str, int],
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "status": "insufficient_samples",
        "method": method,
        "confidence_level": confidence_level,
        "requested_resamples": resamples,
        "completed_resamples": 0,
        "minimum_samples": min_ci_samples,
        "actual_samples": actual_samples,
        "lower": None,
        "upper": None,
        "reason": reason
        or f"at least {min_ci_samples} samples are required for a confidence interval",
    }


def _available_ci(
    estimates: list[float],
    *,
    method: str,
    confidence_level: float,
    resamples: int,
    min_ci_samples: int,
    actual_samples: int | Mapping[str, int],
    seed: int,
) -> dict[str, object]:
    estimates.sort()
    tail = (1.0 - confidence_level) / 2.0
    return {
        "status": "available",
        "method": method,
        "confidence_level": confidence_level,
        "requested_resamples": resamples,
        "completed_resamples": len(estimates),
        "minimum_samples": min_ci_samples,
        "actual_samples": actual_samples,
        "seed": seed,
        "lower": _percentile(estimates, tail),
        "upper": _percentile(estimates, 1.0 - tail),
        "reason": None,
    }


def _bootstrap_median_ci(
    values: Sequence[float],
    *,
    seed: int,
    confidence_level: float,
    resamples: int,
    min_ci_samples: int,
    method: str = "percentile_bootstrap_median",
) -> dict[str, object]:
    if len(values) < min_ci_samples:
        return _insufficient_ci(
            method=method,
            confidence_level=confidence_level,
            resamples=resamples,
            min_ci_samples=min_ci_samples,
            actual_samples=len(values),
        )
    generator = random.Random(seed)
    size = len(values)
    estimates = [
        float(statistics.median(values[generator.randrange(size)] for _ in range(size)))
        for _ in range(resamples)
    ]
    return _available_ci(
        estimates,
        method=method,
        confidence_level=confidence_level,
        resamples=resamples,
        min_ci_samples=min_ci_samples,
        actual_samples=size,
        seed=seed,
    )


def summarize_samples(
    values: Sequence[Real],
    *,
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    min_ci_samples: int = DEFAULT_MIN_CI_SAMPLES,
) -> dict[str, object]:
    """Return JSON-serializable robust summary statistics and a median CI."""

    _validate_bootstrap_options(
        confidence_level=confidence_level,
        resamples=resamples,
        min_ci_samples=min_ci_samples,
    )
    samples = [
        _finite_number(value, label=f"values[{index}]")
        for index, value in enumerate(values)
    ]
    if not samples:
        return {
            "samples": 0,
            "median": None,
            "median_absolute_deviation": None,
            "min": None,
            "max": None,
            "median_ci": _insufficient_ci(
                method="percentile_bootstrap_median",
                confidence_level=confidence_level,
                resamples=resamples,
                min_ci_samples=min_ci_samples,
                actual_samples=0,
                reason="no completed samples",
            ),
        }
    median = float(statistics.median(samples))
    deviations = [abs(value - median) for value in samples]
    return {
        "samples": len(samples),
        "median": median,
        "median_absolute_deviation": float(statistics.median(deviations)),
        "min": min(samples),
        "max": max(samples),
        "median_ci": _bootstrap_median_ci(
            samples,
            seed=seed,
            confidence_level=confidence_level,
            resamples=resamples,
            min_ci_samples=min_ci_samples,
        ),
    }


def _json_safe_repetition(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _repetition_sort_key(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _extract_runs(
    runs: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    repetition_key: str,
    label: str,
) -> tuple[list[float], dict[object, float] | None, str | None]:
    values: list[float] = []
    by_repetition: dict[object, float] = {}
    pairing_reason = None
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise TypeError(f"{label}[{index}] must be a mapping")
        if value_key not in run:
            raise ValueError(f"{label}[{index}] has no {value_key!r}")
        value = _finite_number(run[value_key], label=f"{label}[{index}].{value_key}")
        if value <= 0:
            raise ValueError(f"{label}[{index}].{value_key} must be greater than zero")
        values.append(value)
        if repetition_key not in run:
            pairing_reason = f"one or more {label} runs have no {repetition_key!r}"
            continue
        repetition = run[repetition_key]
        try:
            hash(repetition)
        except TypeError as error:
            raise TypeError(
                f"{label}[{index}].{repetition_key} must be hashable"
            ) from error
        if repetition in by_repetition:
            raise ValueError(f"duplicate {label} repetition {repetition!r}")
        by_repetition[repetition] = value
    if pairing_reason is not None:
        return values, None, pairing_reason
    return values, by_repetition, None


def _effect(
    estimate: float | None,
    ci: Mapping[str, object],
    *,
    estimate_kind: str,
    sample_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "estimate": estimate,
        "estimate_kind": estimate_kind,
        "samples": sample_summary.get("samples") if sample_summary else None,
        "median_absolute_deviation": (
            sample_summary.get("median_absolute_deviation")
            if sample_summary
            else None
        ),
        "min": sample_summary.get("min") if sample_summary else None,
        "max": sample_summary.get("max") if sample_summary else None,
        "ci": dict(ci),
    }


def _unpaired_effect_cis(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int,
    confidence_level: float,
    resamples: int,
    min_ci_samples: int,
) -> dict[str, dict[str, object]]:
    actual_samples = {"candidate": len(candidate), "baseline": len(baseline)}
    methods = {
        "baseline_minus_candidate": (
            "unpaired_percentile_bootstrap_difference_of_medians"
        ),
        "baseline_over_candidate": (
            "unpaired_percentile_bootstrap_ratio_of_medians"
        ),
        "relative_time_reduction": (
            "unpaired_percentile_bootstrap_relative_time_reduction"
        ),
    }
    if min(len(candidate), len(baseline)) < min_ci_samples:
        return {
            name: _insufficient_ci(
                method=method,
                confidence_level=confidence_level,
                resamples=resamples,
                min_ci_samples=min_ci_samples,
                actual_samples=actual_samples,
            )
            for name, method in methods.items()
        }
    generator = random.Random(seed)
    candidate_size = len(candidate)
    baseline_size = len(baseline)
    estimates = {name: [] for name in methods}
    for _ in range(resamples):
        candidate_median = float(
            statistics.median(
                candidate[generator.randrange(candidate_size)]
                for _ in range(candidate_size)
            )
        )
        baseline_median = float(
            statistics.median(
                baseline[generator.randrange(baseline_size)]
                for _ in range(baseline_size)
            )
        )
        estimates["baseline_minus_candidate"].append(
            baseline_median - candidate_median
        )
        estimates["baseline_over_candidate"].append(
            baseline_median / candidate_median
        )
        estimates["relative_time_reduction"].append(
            1.0 - candidate_median / baseline_median
        )
    return {
        name: _available_ci(
            effect_estimates,
            method=methods[name],
            confidence_level=confidence_level,
            resamples=resamples,
            min_ci_samples=min_ci_samples,
            actual_samples=actual_samples,
            seed=seed,
        )
        for name, effect_estimates in estimates.items()
    }


def compare_randomized_runs(
    candidate_runs: Sequence[Mapping[str, object]],
    baseline_runs: Sequence[Mapping[str, object]],
    *,
    value_key: str = "elapsed_s",
    repetition_key: str = "repetition",
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    min_ci_samples: int = DEFAULT_MIN_CI_SAMPLES,
) -> dict[str, object]:
    """Compare durations, pairing randomized runs only when repetition sets match.

    Relative time reduction is ``1 - candidate / baseline``.  It is positive
    when the candidate is faster and negative when the candidate regresses.
    """

    _validate_bootstrap_options(
        confidence_level=confidence_level,
        resamples=resamples,
        min_ci_samples=min_ci_samples,
    )
    candidate, candidate_by_repetition, candidate_reason = _extract_runs(
        candidate_runs,
        value_key=value_key,
        repetition_key=repetition_key,
        label="candidate_runs",
    )
    baseline, baseline_by_repetition, baseline_reason = _extract_runs(
        baseline_runs,
        value_key=value_key,
        repetition_key=repetition_key,
        label="baseline_runs",
    )
    candidate_summary = summarize_samples(
        candidate,
        seed=seed + 1,
        confidence_level=confidence_level,
        resamples=resamples,
        min_ci_samples=min_ci_samples,
    )
    baseline_summary = summarize_samples(
        baseline,
        seed=seed + 2,
        confidence_level=confidence_level,
        resamples=resamples,
        min_ci_samples=min_ci_samples,
    )

    candidate_keys = (
        set(candidate_by_repetition) if candidate_by_repetition is not None else None
    )
    baseline_keys = (
        set(baseline_by_repetition) if baseline_by_repetition is not None else None
    )
    if candidate_reason is not None:
        pairing_reason = candidate_reason
    elif baseline_reason is not None:
        pairing_reason = baseline_reason
    elif not candidate_keys or not baseline_keys:
        pairing_reason = "one or both groups have no completed runs"
    elif candidate_keys != baseline_keys:
        pairing_reason = "candidate and baseline repetition sets do not match exactly"
    else:
        pairing_reason = None

    per_repetition: list[dict[str, object]] = []
    if pairing_reason is None:
        assert candidate_by_repetition is not None
        assert baseline_by_repetition is not None
        for repetition in sorted(candidate_by_repetition, key=_repetition_sort_key):
            candidate_value = candidate_by_repetition[repetition]
            baseline_value = baseline_by_repetition[repetition]
            per_repetition.append(
                {
                    "repetition": _json_safe_repetition(repetition),
                    "candidate": candidate_value,
                    "baseline": baseline_value,
                    "baseline_minus_candidate": baseline_value - candidate_value,
                    "baseline_over_candidate": baseline_value / candidate_value,
                    "relative_time_reduction": 1.0
                    - candidate_value / baseline_value,
                }
            )
        mode = "paired_by_repetition"
        difference_summary = summarize_samples(
            [float(row["baseline_minus_candidate"]) for row in per_repetition],
            seed=seed + 3,
            confidence_level=confidence_level,
            resamples=resamples,
            min_ci_samples=min_ci_samples,
        )
        ratio_summary = summarize_samples(
            [float(row["baseline_over_candidate"]) for row in per_repetition],
            seed=seed + 4,
            confidence_level=confidence_level,
            resamples=resamples,
            min_ci_samples=min_ci_samples,
        )
        reduction_summary = summarize_samples(
            [float(row["relative_time_reduction"]) for row in per_repetition],
            seed=seed + 5,
            confidence_level=confidence_level,
            resamples=resamples,
            min_ci_samples=min_ci_samples,
        )
        effects = {
            "baseline_minus_candidate": _effect(
                difference_summary["median"],
                difference_summary["median_ci"],
                estimate_kind="median_of_paired_differences",
                sample_summary=difference_summary,
            ),
            "baseline_over_candidate": _effect(
                ratio_summary["median"],
                ratio_summary["median_ci"],
                estimate_kind="median_of_paired_ratios",
                sample_summary=ratio_summary,
            ),
            "relative_time_reduction": _effect(
                reduction_summary["median"],
                reduction_summary["median_ci"],
                estimate_kind="median_of_paired_reductions",
                sample_summary=reduction_summary,
            ),
        }
    else:
        mode = "unpaired"
        effect_cis = _unpaired_effect_cis(
            candidate,
            baseline,
            seed=seed + 3,
            confidence_level=confidence_level,
            resamples=resamples,
            min_ci_samples=min_ci_samples,
        )
        candidate_median = candidate_summary["median"]
        baseline_median = baseline_summary["median"]
        if isinstance(candidate_median, float) and isinstance(baseline_median, float):
            estimates: dict[str, float | None] = {
                "baseline_minus_candidate": baseline_median - candidate_median,
                "baseline_over_candidate": baseline_median / candidate_median,
                "relative_time_reduction": 1.0
                - candidate_median / baseline_median,
            }
        else:
            estimates = {name: None for name in effect_cis}
        effects = {
            name: _effect(
                estimates[name],
                effect_cis[name],
                estimate_kind="effect_of_group_medians",
            )
            for name in effect_cis
        }

    result = {
        "schema_version": 1,
        "mode": mode,
        "pairing": {
            "key": repetition_key,
            "paired_runs": len(per_repetition),
            "reason_unpaired": pairing_reason,
        },
        "value_key": value_key,
        "direction": "lower_is_better",
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "effects": effects,
        "per_repetition": per_repetition,
    }
    # Keep the public contract honest even for unusual, but hashable, run keys.
    json.dumps(result, allow_nan=False)
    return result


def _gate_result(
    comparison: Mapping[str, object],
    *,
    gate: str,
    required_reduction: float,
    minimum_runs: int,
    require_confidence: bool,
) -> dict[str, object]:
    if isinstance(minimum_runs, bool) or not isinstance(minimum_runs, int):
        raise TypeError("minimum_runs must be an integer")
    if minimum_runs < 1:
        raise ValueError("minimum_runs must be positive")
    reduction = comparison.get("effects", {}).get("relative_time_reduction", {})  # type: ignore[union-attr]
    estimate = reduction.get("estimate") if isinstance(reduction, Mapping) else None
    ci = reduction.get("ci") if isinstance(reduction, Mapping) else None
    candidate = comparison.get("candidate")
    baseline = comparison.get("baseline")
    candidate_samples = (
        candidate.get("samples") if isinstance(candidate, Mapping) else 0
    )
    baseline_samples = baseline.get("samples") if isinstance(baseline, Mapping) else 0
    effective_samples = min(int(candidate_samples or 0), int(baseline_samples or 0))
    point_estimate_passed = (
        float(estimate) >= required_reduction
        if isinstance(estimate, (int, float))
        else None
    )
    lower = ci.get("lower") if isinstance(ci, Mapping) else None
    confidence_supported = (
        float(lower) >= required_reduction
        if isinstance(lower, (int, float))
        else None
    )
    if effective_samples < minimum_runs or point_estimate_passed is None:
        status = "insufficient_samples"
        passed = None
    elif require_confidence and confidence_supported is None:
        status = "insufficient_confidence"
        passed = None
    else:
        passed = confidence_supported if require_confidence else point_estimate_passed
        status = "pass" if passed else "fail"
    return {
        "gate": gate,
        "status": status,
        "passed": passed,
        "decision_basis": (
            "lower_confidence_bound" if require_confidence else "point_estimate"
        ),
        "minimum_runs": minimum_runs,
        "effective_samples": effective_samples,
        "required_relative_time_reduction": required_reduction,
        "observed_relative_time_reduction": estimate,
        "point_estimate_passed": point_estimate_passed,
        "confidence_supported": confidence_supported,
        "comparison": dict(comparison),
    }


def regression_gate(
    comparison: Mapping[str, object],
    *,
    max_regression: float = 0.05,
    minimum_runs: int = DEFAULT_MIN_CI_SAMPLES,
    require_confidence: bool = False,
) -> dict[str, object]:
    """Evaluate the normal-workload gate (candidate regression at most 5%)."""

    maximum = _finite_number(max_regression, label="max_regression")
    if maximum < 0:
        raise ValueError("max_regression must not be negative")
    result = _gate_result(
        comparison,
        gate="maximum_normal_workload_regression",
        required_reduction=-maximum,
        minimum_runs=minimum_runs,
        require_confidence=require_confidence,
    )
    result["maximum_regression"] = maximum
    observed = result["observed_relative_time_reduction"]
    result["observed_regression"] = (
        -float(observed) if isinstance(observed, (int, float)) else None
    )
    return result


def speedup_gate(
    comparison: Mapping[str, object],
    *,
    minimum_speedup: float = 0.10,
    minimum_runs: int = DEFAULT_MIN_CI_SAMPLES,
    require_confidence: bool = False,
) -> dict[str, object]:
    """Evaluate the workload gate (candidate time reduction at least 10%)."""

    minimum = _finite_number(minimum_speedup, label="minimum_speedup")
    if not 0 <= minimum < 1:
        raise ValueError("minimum_speedup must be at least 0 and less than 1")
    result = _gate_result(
        comparison,
        gate="minimum_workload_speedup",
        required_reduction=minimum,
        minimum_runs=minimum_runs,
        require_confidence=require_confidence,
    )
    result["minimum_speedup"] = minimum
    return result


def speedup_vs_best_completing_stock(
    candidate_runs: Sequence[Mapping[str, object]],
    stock_workarounds: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    value_key: str = "elapsed_s",
    repetition_key: str = "repetition",
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    min_ci_samples: int = DEFAULT_MIN_CI_SAMPLES,
    minimum_speedup: float = 0.10,
    minimum_runs: int = DEFAULT_MIN_CI_SAMPLES,
    require_confidence: bool = False,
) -> dict[str, object]:
    """Select the fastest completing stock arm and evaluate the speedup gate."""

    stock_medians: dict[str, float] = {}
    excluded: list[str] = []
    for name in sorted(stock_workarounds):
        runs = stock_workarounds[name]
        values, _, _ = _extract_runs(
            runs,
            value_key=value_key,
            repetition_key=repetition_key,
            label=f"stock_workarounds[{name!r}]",
        )
        if not values:
            excluded.append(name)
            continue
        stock_medians[name] = float(statistics.median(values))
    if not stock_medians:
        return {
            "gate": "speedup_vs_best_completing_stock",
            "status": "no_completing_stock_workaround",
            "passed": None,
            "best_stock_workaround": None,
            "stock_medians": {},
            "excluded_stock_workarounds": excluded,
            "comparison": None,
        }
    best_name = min(stock_medians, key=lambda name: (stock_medians[name], name))
    comparison = compare_randomized_runs(
        candidate_runs,
        stock_workarounds[best_name],
        value_key=value_key,
        repetition_key=repetition_key,
        seed=seed,
        confidence_level=confidence_level,
        resamples=resamples,
        min_ci_samples=min_ci_samples,
    )
    result = speedup_gate(
        comparison,
        minimum_speedup=minimum_speedup,
        minimum_runs=minimum_runs,
        require_confidence=require_confidence,
    )
    result.update(
        gate="speedup_vs_best_completing_stock",
        best_stock_workaround=best_name,
        stock_medians=stock_medians,
        excluded_stock_workarounds=excluded,
    )
    json.dumps(result, allow_nan=False)
    return result


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_MIN_CI_SAMPLES",
    "compare_randomized_runs",
    "regression_gate",
    "speedup_gate",
    "speedup_vs_best_completing_stock",
    "summarize_samples",
]
