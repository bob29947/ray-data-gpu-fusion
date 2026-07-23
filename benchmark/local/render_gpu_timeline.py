#!/usr/bin/env python3
"""Render reviewer-safe GPU ownership evidence from a completed workload JSON.

The renderer deliberately accepts only successful incident-shaped runs with
complete, common-clock progress telemetry.  It never imports Ray and never
queries a live cluster.  Its three outputs are deterministic functions of the
source workload bytes:

* ``<prefix>.svg``: compact ownership timeline with premature allocations in red.
* ``<prefix>.csv``: sample-level ownership values by operator role.
* ``<prefix>.json``: machine-readable causal and aggregate summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA_VERSION = 1
CLOCK_BASIS = "single-boot-monotonic"
EVENT_MEASURE = "lifetime-resource constructor-ready to stage-first-input reservation"
GCS_CORROBORATION = (
    "at least one in-window ALIVE GCS GPU-owner sample per counted actor; "
    "corroboration, not continuous sampling proof"
)
EXPECTED_STAGES = (
    "gpu-map-add-key",
    "gpu-map-groups-1",
    "gpu-map-final",
)
DOWNSTREAM_STAGES = ("gpu-map-groups-1", "gpu-map-final")
ROLE_DEFINITIONS = (
    ("upstream_map_batches", "upstream map_batches", "#2563eb"),
    ("shuffle", "GPU shuffle", "#7c3aed"),
    ("post_shuffle_map_groups", "post-shuffle map_groups", "#f59e0b"),
    ("post_shuffle_map_batches", "final map_batches", "#10b981"),
    ("other_gpu_actor", "other GPU actor", "#64748b"),
    ("unattributed_reservation", "unattributed reservation", "#cbd5e1"),
)
ROLE_KEYS = tuple(role for role, _, _ in ROLE_DEFINITIONS)
TIMELINE_FIELDS = (
    "elapsed_s",
    "cluster_gpus",
    "available_gpus",
    "ray_owned_gpus",
    *ROLE_KEYS,
    "lifetime_reserved_post_shuffle_map_groups_gpus",
    "lifetime_reserved_post_shuffle_map_batches_gpus",
    "premature_downstream_gpus_during_earlier_stage",
)
CSV_FIELDS = (
    "source_workload_sha256",
    "event_measure",
    "gcs_corroboration",
    *TIMELINE_FIELDS,
)


class EvidenceError(ValueError):
    """Raised when the source cannot support the causal claim."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise EvidenceError(f"{label} must be an array")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be a finite number")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    number = _number(value, label)
    if not number.is_integer() or number < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return int(number)


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} must be a nonempty string")
    return value


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _gpu_count(resources: object) -> float:
    if not isinstance(resources, Mapping):
        return 0.0
    return sum(
        _number(value, f"GPU resource {key}")
        for key, value in resources.items()
        if str(key).upper() == "GPU"
    )


def _stage_role(stage: str) -> str:
    if stage == "gpu-map-add-key":
        return "upstream_map_batches"
    if stage.startswith("gpu-map-groups-"):
        return "post_shuffle_map_groups"
    if stage == "gpu-map-final":
        return "post_shuffle_map_batches"
    raise EvidenceError(f"unsupported GPU stage in causal telemetry: {stage!r}")


def _stage_order(stage: str) -> int:
    if stage == "gpu-map-add-key":
        return 0
    if stage.startswith("gpu-map-groups-"):
        return int(stage.rsplit("-", 1)[1])
    if stage == "gpu-map-final":
        return 3
    raise EvidenceError(f"unsupported GPU stage order: {stage!r}")


def _class_role(class_name: str) -> str | None:
    if "GPUShuffleActor" in class_name:
        return "shuffle"
    if "AddKey" in class_name:
        return "upstream_map_batches"
    if "SumGroup" in class_name:
        return "post_shuffle_map_groups"
    if "Identity" in class_name:
        return "post_shuffle_map_batches"
    return None


def _actor_role(actor: Mapping[str, object], actor_stages: Mapping[str, str]) -> str:
    actor_id = str(actor.get("actor_id", ""))
    if actor_id in actor_stages:
        return _stage_role(actor_stages[actor_id])
    return _class_role(str(actor.get("class_name", ""))) or "other_gpu_actor"


def _integrate(rows: Sequence[Mapping[str, float]], key: str) -> float:
    total = 0.0
    for left, right in zip(rows, rows[1:]):
        duration = right["elapsed_s"] - left["elapsed_s"]
        total += duration * (left[key] + right[key]) / 2.0
    return total


def _validate_progress(
    document: Mapping[str, object], samples: Sequence[Mapping[str, object]]
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    list[dict[str, object]],
    Mapping[str, object],
]:
    drain = _mapping(document.get("telemetry_drain"), "telemetry_drain")
    if drain.get("complete") is not True:
        raise EvidenceError("telemetry drain is incomplete")

    latest = _mapping(samples[-1].get("progress"), "final progress sample")
    if _mapping(document.get("final_progress"), "top-level final progress") != latest:
        raise EvidenceError("top-level and sampled final progress snapshots disagree")
    if latest.get("error") is not None:
        raise EvidenceError("final progress sample contains an error")
    coverage = _mapping(latest.get("telemetry_coverage"), "telemetry coverage")
    if coverage.get("complete") is not True:
        raise EvidenceError("fixed-pool progress telemetry is incomplete")
    quiescence = _mapping(
        latest.get("telemetry_quiescence"), "progress telemetry quiescence"
    )
    if quiescence.get("complete") is not True:
        raise EvidenceError("progress telemetry did not become quiescent")
    stable_sequence = _integer(
        quiescence.get("stable_sequence"), "quiescence stable sequence"
    )
    stable_polls = _integer(
        quiescence.get("stable_polls"), "quiescence stable polls", minimum=1
    )
    required_polls = _integer(
        quiescence.get("required_polls"), "quiescence required polls", minimum=1
    )
    stable_duration = _number(
        quiescence.get("stable_duration_s"), "quiescence stable duration"
    )
    required_duration = _number(
        quiescence.get("required_duration_s"), "quiescence required duration"
    )
    if (
        stable_sequence != _integer(latest.get("sequence"), "progress sequence")
        or stable_polls < required_polls
        or stable_duration < required_duration
        or required_duration < 0
    ):
        raise EvidenceError("progress telemetry quiescence proof is inconsistent")
    if _mapping(drain.get("quiescence"), "drain quiescence") != quiescence:
        raise EvidenceError("drain and final progress quiescence disagree")

    origin_epoch = _number(
        document.get("telemetry_origin_epoch_s"), "top-level origin_epoch_s"
    )
    origin_monotonic = _number(
        document.get("telemetry_origin_monotonic_s"),
        "top-level origin_monotonic_s",
    )
    origin_boot = _nonempty_string(
        document.get("telemetry_origin_boot_id"), "top-level origin_boot_id"
    )

    def validate_origin(value: Mapping[str, object], label: str) -> None:
        epoch = _number(value.get("origin_epoch_s"), f"{label} origin_epoch_s")
        monotonic = _number(
            value.get("origin_monotonic_s"), f"{label} origin_monotonic_s"
        )
        boot = _nonempty_string(value.get("origin_boot_id"), f"{label} origin_boot_id")
        if (
            not math.isclose(epoch, origin_epoch, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(monotonic, origin_monotonic, rel_tol=0.0, abs_tol=1e-9)
            or boot != origin_boot
        ):
            raise EvidenceError(f"{label} does not match the top-level clock origin")

    validate_origin(latest, "final progress")
    for index, sample in enumerate(samples):
        validate_origin(sample, f"sample {index}")
        validate_origin(
            _mapping(sample.get("progress"), f"sample {index} progress"),
            f"sample {index} progress",
        )

    def event_time(state: Mapping[str, object], stage: str, event: str) -> float:
        elapsed = _number(state.get(f"first_{event}_s"), f"{stage} first {event}")
        monotonic = _number(
            state.get(f"first_{event}_monotonic_s"),
            f"{stage} first {event} monotonic time",
        )
        boot = _nonempty_string(
            state.get(f"first_{event}_boot_id"),
            f"{stage} first {event} boot ID",
        )
        basis = state.get(f"first_{event}_clock_basis")
        if (
            elapsed < 0
            or monotonic < origin_monotonic
            or not math.isclose(
                elapsed,
                monotonic - origin_monotonic,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or boot != origin_boot
            or basis != CLOCK_BASIS
        ):
            raise EvidenceError(
                f"{stage} first {event} is not verified single-boot monotonic telemetry"
            )
        return elapsed

    stages = _mapping(latest.get("stages"), "final progress stages")
    markers = []
    for stage in EXPECTED_STAGES:
        state = _mapping(stages.get(stage), f"progress stage {stage}")
        first_input = event_time(state, stage, "input")
        first_progress = event_time(state, stage, "progress")
        if (
            _number(
                state.get("first_input_rows_sampled"),
                f"{stage} first-input sampled rows",
            )
            <= 0
        ):
            raise EvidenceError(f"{stage} first-input sampled rows must be positive")
        if first_progress < first_input:
            raise EvidenceError(f"invalid first-input/progress order for {stage}")
        markers.append(
            {
                "stage": stage,
                "role": _stage_role(stage),
                "first_input_s": _rounded(first_input),
                "first_progress_s": _rounded(first_progress),
            }
        )

    expected_counts = _mapping(
        latest.get("expected_actor_counts"), "expected fixed-pool actor counts"
    )
    if set(expected_counts) != set(DOWNSTREAM_STAGES):
        raise EvidenceError("renderer requires the incident-shaped downstream stages")
    registry = _mapping(latest.get("actors"), "final progress actor registry")
    actor_stages: dict[str, str] = {}
    registered: dict[str, list[tuple[str, Mapping[str, object]]]] = {
        stage: [] for stage in DOWNSTREAM_STAGES
    }
    for actor_id, raw in registry.items():
        actor = _mapping(raw, f"progress actor {actor_id}")
        stage = str(actor.get("stage", ""))
        if stage not in {*EXPECTED_STAGES, "gpu-map-groups-2"}:
            continue
        actor_stages[str(actor_id)] = stage
        if stage in registered:
            registered[stage].append((str(actor_id), actor))

    marker_by_stage = {str(item["stage"]): item for item in markers}
    premature = []
    for stage in DOWNSTREAM_STAGES:
        expected = _integer(
            expected_counts.get(stage), f"expected {stage} actors", minimum=1
        )
        if expected < 1 or len(registered[stage]) != expected:
            raise EvidenceError(
                f"{stage} registered {len(registered[stage])} actors, expected {expected}"
            )
        stage_input = float(marker_by_stage[stage]["first_input_s"])
        registration_events = 0
        for actor_id, actor in registered[stage]:
            ready = _number(actor.get("ready_s"), f"{stage} actor ready time")
            ready_monotonic = _number(
                actor.get("ready_monotonic_s"),
                f"{stage} actor ready monotonic time",
            )
            ready_boot = _nonempty_string(
                actor.get("ready_boot_id"), f"{stage} actor ready boot ID"
            )
            registrations = _integer(
                actor.get("registration_events"),
                f"{stage} actor registration events",
            )
            registration_events += registrations
            raw_first_input = actor.get("first_input_s")
            first_input = (
                _number(raw_first_input, f"{stage} actor first input")
                if raw_first_input is not None
                else None
            )
            gpus = _number(actor.get("gpus"), f"{stage} actor GPUs")
            if (
                ready < 0
                or ready_monotonic < origin_monotonic
                or not math.isclose(
                    ready,
                    ready_monotonic - origin_monotonic,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or ready_boot != origin_boot
                or actor.get("ready_clock_basis") != CLOCK_BASIS
                or registrations != 1
                or (first_input is not None and first_input < ready)
                or gpus <= 0
            ):
                raise EvidenceError(f"invalid lifecycle telemetry for {stage} actor")
            if ready < stage_input:
                premature.append(
                    {
                        "actor_id": actor_id,
                        "stage": stage,
                        "role": _stage_role(stage),
                        "gpus": gpus,
                        "ready_s": ready,
                        "stage_first_input_s": stage_input,
                        "actor_first_input_s": first_input,
                    }
                )
        if registration_events != expected:
            raise EvidenceError(
                f"{stage} registration events {registration_events}, expected {expected}"
            )

    metric = _mapping(
        _mapping(document.get("resource_metrics"), "resource_metrics").get(
            "premature_downstream_ownership"
        ),
        "premature downstream metric",
    )
    if metric.get("measurement_complete") is not True:
        raise EvidenceError("premature downstream measurement is incomplete")
    if (
        metric.get("common_clock_verified") is not True
        or metric.get("clock_basis") != CLOCK_BASIS
    ):
        raise EvidenceError("producer did not verify a single-boot monotonic clock")
    if metric.get("gcs_state_samples_valid") is not True:
        raise EvidenceError("producer reports invalid GCS state samples")
    if metric.get("gcs_corroboration_complete") is not True:
        raise EvidenceError("producer GCS corroboration is incomplete")
    if metric.get("gcs_corroboration_issue") is not None:
        raise EvidenceError("producer reports GCS corroboration issues")
    if (
        _mapping(metric.get("telemetry_quiescence"), "metric telemetry quiescence")
        != quiescence
    ):
        raise EvidenceError("metric and final progress quiescence disagree")

    metric_count = _integer(
        metric.get("premature_downstream_actor_count"), "premature actor count"
    )
    corroborated_count = _integer(
        metric.get("gcs_corroborated_actor_count"),
        "GCS corroborated actor count",
    )
    expected_gcs_count = _integer(
        metric.get("gcs_expected_actor_count"), "GCS expected actor count"
    )
    exact_gpu_seconds = sum(
        (float(actor["stage_first_input_s"]) - float(actor["ready_s"]))
        * float(actor["gpus"])
        for actor in premature
    )
    metric_gpu_seconds = _number(
        metric.get("premature_downstream_gpu_seconds"), "premature GPU-seconds"
    )
    if metric_count != len(premature) or not math.isclose(
        metric_gpu_seconds, exact_gpu_seconds, rel_tol=1e-7, abs_tol=1e-6
    ):
        raise EvidenceError("premature ownership metric disagrees with actor events")
    if corroborated_count != len(premature) or expected_gcs_count != len(premature):
        raise EvidenceError("producer GCS corroboration counts disagree with actors")
    for key in (
        "premature_downstream_gpu_seconds_during_earlier_stage",
        "peak_premature_downstream_gpus_during_earlier_stage",
    ):
        if _number(metric.get(key), key) < 0:
            raise EvidenceError(f"{key} must be nonnegative")
    return markers, actor_stages, premature, metric


def build_evidence(
    document: Mapping[str, object], *, source_name: str, source_sha256: str
) -> tuple[dict[str, object], list[dict[str, float]]]:
    """Validate a source document and return its summary and timeline rows."""

    if document.get("status") != "success":
        raise EvidenceError("workload must have completed successfully")
    if document.get("sampler_errors"):
        raise EvidenceError("workload contains sampler errors")
    raw_samples = _sequence(document.get("resource_samples"), "resource_samples")
    if len(raw_samples) < 2:
        raise EvidenceError("at least two resource samples are required")

    samples: list[Mapping[str, object]] = []
    previous = -1.0
    for index, raw in enumerate(raw_samples):
        sample = _mapping(raw, f"resource sample {index}")
        elapsed = _number(sample.get("elapsed_s"), f"sample {index} elapsed_s")
        if elapsed < 0 or elapsed <= previous:
            raise EvidenceError("resource sample times must be strictly increasing")
        previous = elapsed
        state = _mapping(sample.get("ray_state"), f"sample {index} ray_state")
        if state.get("error") is not None:
            raise EvidenceError(f"sample {index} contains a Ray-state error")
        counts = _mapping(state.get("counts"), f"sample {index} state counts")
        actor_counts = _mapping(counts.get("actors"), f"sample {index} actor counts")
        query_limit = _integer(state.get("query_limit"), "state query limit", minimum=1)
        actor_count = sum(
            _integer(value, f"sample {index} actor count")
            for value in actor_counts.values()
        )
        if state.get("possibly_truncated") is not False or actor_count >= query_limit:
            raise EvidenceError("GPU actor state may be truncated")
        _sequence(state.get("actors"), f"sample {index} actors")
        if (
            _mapping(sample.get("progress"), f"sample {index} progress").get("error")
            is not None
        ):
            raise EvidenceError(f"sample {index} contains a progress error")
        samples.append(sample)

    markers, actor_stages, premature, producer_metric = _validate_progress(
        document, samples
    )
    target_job = str(document.get("ray_job_id", ""))
    if not target_job:
        raise EvidenceError("ray_job_id is required to exclude external GPU owners")

    marker_by_stage = {str(item["stage"]): item for item in markers}
    timeline: list[dict[str, float]] = []
    premature_by_id = {str(actor["actor_id"]): actor for actor in premature}
    alive_observations = {actor_id: 0 for actor_id in premature_by_id}
    actor_signatures: dict[str, tuple[str, str, float]] = {}
    for index, sample in enumerate(samples):
        elapsed = _number(sample["elapsed_s"], f"sample {index} elapsed_s")
        cluster = _gpu_count(_mapping(sample.get("cluster"), "cluster resources"))
        available = _gpu_count(_mapping(sample.get("available"), "available resources"))
        row = {key: 0.0 for key in TIMELINE_FIELDS}
        row.update(
            elapsed_s=elapsed,
            cluster_gpus=cluster,
            available_gpus=available,
            ray_owned_gpus=max(0.0, cluster - available),
        )
        state = _mapping(sample["ray_state"], "Ray state")
        sample_actor_ids: set[str] = set()
        alive_premature: dict[str, float] = {}
        for raw_actor in _sequence(state["actors"], "Ray actors"):
            actor = _mapping(raw_actor, "Ray actor")
            actor_id = str(actor.get("actor_id", ""))
            if not actor_id or actor_id in sample_actor_ids:
                raise EvidenceError(
                    f"sample {index} has a missing or duplicate actor ID"
                )
            sample_actor_ids.add(actor_id)
            gpus = _gpu_count(actor.get("required_resources"))
            actor_job = str(actor.get("job_id", ""))
            tracked = premature_by_id.get(actor_id)
            if tracked is not None:
                restarts = _number(
                    actor.get("num_restarts"), f"actor {actor_id} restart count"
                )
                if not restarts.is_integer() or restarts != 0:
                    raise EvidenceError(
                        f"premature actor {actor_id} has a restart history"
                    )
                expected_gpus = float(tracked["gpus"])
                if not math.isclose(gpus, expected_gpus, abs_tol=1e-9):
                    raise EvidenceError(
                        f"premature actor {actor_id} has conflicting GPU resources"
                    )
                expected_role = str(tracked["role"])
                class_name = str(actor.get("class_name", ""))
                if _class_role(class_name) != expected_role:
                    raise EvidenceError(
                        f"premature actor {actor_id} has a conflicting operator class"
                    )
                if actor_job != target_job:
                    raise EvidenceError(
                        f"premature actor {actor_id} has a conflicting Ray job"
                    )
                signature = (actor_job, class_name, _rounded(gpus))
                prior = actor_signatures.setdefault(actor_id, signature)
                if prior != signature:
                    raise EvidenceError(
                        f"premature actor {actor_id} has conflicting GCS records"
                    )
                in_interval = (
                    float(tracked["ready_s"])
                    <= elapsed
                    < float(tracked["stage_first_input_s"])
                )
                if str(actor.get("state", "")).upper() == "ALIVE" and in_interval:
                    alive_observations[actor_id] += 1
                    alive_premature[actor_id] = gpus
            if gpus <= 0 or str(actor.get("state", "")).upper() != "ALIVE":
                continue
            if actor_job != target_job:
                raise EvidenceError(
                    f"external job {actor_job or '<missing>'} owns a sampled GPU"
                )
            row[_actor_role(actor, actor_stages)] += gpus
        known = sum(row[key] for key in ROLE_KEYS if key != "unattributed_reservation")
        row["unattributed_reservation"] = max(0.0, row["ray_owned_gpus"] - known)
        for actor in premature:
            if float(actor["ready_s"]) <= elapsed < float(actor["stage_first_input_s"]):
                column = (
                    "lifetime_reserved_post_shuffle_map_groups_gpus"
                    if actor["stage"] == "gpu-map-groups-1"
                    else "lifetime_reserved_post_shuffle_map_batches_gpus"
                )
                row[column] += float(actor["gpus"])
        active_order = max(
            (
                _stage_order(str(marker["stage"]))
                for marker in markers
                if float(marker["first_input_s"]) <= elapsed
            ),
            default=0,
        )
        row["premature_downstream_gpus_during_earlier_stage"] = sum(
            gpus
            for actor_id, gpus in alive_premature.items()
            if _stage_order(str(premature_by_id[actor_id]["stage"])) > active_order
        )
        timeline.append({key: _rounded(value) for key, value in row.items()})

    missing_corroboration = sorted(
        actor_id for actor_id, count in alive_observations.items() if count < 1
    )
    if missing_corroboration:
        raise EvidenceError(
            "premature actor has no in-window GCS ALIVE GPU-owner observation: "
            + ", ".join(missing_corroboration)
        )

    end = timeline[-1]["elapsed_s"]
    for marker in markers:
        if float(marker["first_progress_s"]) > end + 1e-6:
            raise EvidenceError("stage progress lies beyond the final resource sample")

    exact_gpu_seconds = sum(
        (float(actor["stage_first_input_s"]) - float(actor["ready_s"]))
        * float(actor["gpus"])
        for actor in premature
    )
    cluster_gpu_seconds = _integrate(timeline, "cluster_gpus")
    earlier_stage_gpu_seconds = _integrate(
        timeline, "premature_downstream_gpus_during_earlier_stage"
    )
    earlier_stage_peak_gpus = max(
        row["premature_downstream_gpus_during_earlier_stage"] for row in timeline
    )
    producer_earlier_seconds = _number(
        producer_metric.get("premature_downstream_gpu_seconds_during_earlier_stage"),
        "producer earlier-stage premature GPU-seconds",
    )
    producer_earlier_peak = _number(
        producer_metric.get("peak_premature_downstream_gpus_during_earlier_stage"),
        "producer peak earlier-stage premature GPUs",
    )
    if not math.isclose(
        earlier_stage_gpu_seconds,
        producer_earlier_seconds,
        rel_tol=1e-7,
        abs_tol=1e-6,
    ) or not math.isclose(
        earlier_stage_peak_gpus,
        producer_earlier_peak,
        rel_tol=1e-7,
        abs_tol=1e-6,
    ):
        raise EvidenceError(
            "producer earlier-stage premature metrics disagree with GCS samples"
        )
    aliases = {
        actor_id: f"actor-{index:03d}"
        for index, actor_id in enumerate(
            sorted(
                (str(actor["actor_id"]) for actor in premature),
                key=lambda actor_id: (
                    actor_stages[actor_id],
                    next(
                        float(item["ready_s"])
                        for item in premature
                        if item["actor_id"] == actor_id
                    ),
                    actor_id,
                ),
            ),
            start=1,
        )
    }
    actor_rows = [
        {
            "actor": aliases[str(actor["actor_id"])],
            "stage": actor["stage"],
            "role": actor["role"],
            "gpus": _rounded(float(actor["gpus"])),
            "ready_s": _rounded(float(actor["ready_s"])),
            "stage_first_input_s": _rounded(float(actor["stage_first_input_s"])),
            "actor_first_input_s": (
                _rounded(float(actor["actor_first_input_s"]))
                if actor["actor_first_input_s"] is not None
                else None
            ),
            "lifetime_reservation_gpu_seconds": _rounded(
                (float(actor["stage_first_input_s"]) - float(actor["ready_s"]))
                * float(actor["gpus"])
            ),
            "gcs_alive_observation_count": alive_observations[str(actor["actor_id"])],
            "gcs_corroborated": True,
        }
        for actor in sorted(
            premature,
            key=lambda item: (
                str(item["stage"]),
                float(item["ready_s"]),
                str(item["actor_id"]),
            ),
        )
    ]
    by_stage = []
    for stage in DOWNSTREAM_STAGES:
        actors = [actor for actor in actor_rows if actor["stage"] == stage]
        by_stage.append(
            {
                "stage": stage,
                "role": _stage_role(stage),
                "actor_count": len(actors),
                "lifetime_reservation_gpu_seconds": _rounded(
                    sum(
                        float(actor["lifetime_reservation_gpu_seconds"])
                        for actor in actors
                    )
                ),
                "first_input_s": marker_by_stage[stage]["first_input_s"],
                "first_progress_s": marker_by_stage[stage]["first_progress_s"],
            }
        )

    role_summaries = [
        {
            "role": role,
            "label": label,
            "peak_gpus": _rounded(max(row[role] for row in timeline)),
            "sampled_gpu_seconds": _rounded(_integrate(timeline, role)),
        }
        for role, label, _ in ROLE_DEFINITIONS
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": {"filename": source_name, "sha256": source_sha256},
        "workload": {
            key: document.get(key)
            for key in (
                "arm",
                "workload",
                "topology",
                "topology_max_gpus",
                "rows",
                "blocks",
                "groups",
                "shuffle_ranks",
            )
        },
        "causal_telemetry": {
            "complete": True,
            "common_clock": True,
            "clock_basis": CLOCK_BASIS,
            "telemetry_quiescent": True,
            "gcs_state_samples_valid": True,
            "sample_count": len(timeline),
            "first_sample_s": timeline[0]["elapsed_s"],
            "last_sample_s": timeline[-1]["elapsed_s"],
            "actor_state_truncated": False,
        },
        "stage_markers": markers,
        "ownership": {
            "roles": role_summaries,
            "peak_ray_owned_gpus": _rounded(
                max(row["ray_owned_gpus"] for row in timeline)
            ),
            "cluster_gpu_seconds": _rounded(cluster_gpu_seconds),
        },
        "premature_downstream": {
            "measurement_complete": True,
            "event_measure": EVENT_MEASURE,
            "gcs_corroboration": {
                "complete": True,
                "requirement": (
                    "each counted actor has at least one in-window ALIVE GCS record "
                    "with matching GPU resources and zero restarts"
                ),
                "interpretation": GCS_CORROBORATION,
            },
            "actor_count": len(actor_rows),
            "lifetime_reservation_gpu_seconds": _rounded(exact_gpu_seconds),
            "premature_downstream_gpu_seconds_during_earlier_stage": _rounded(
                earlier_stage_gpu_seconds
            ),
            "peak_premature_downstream_gpus_during_earlier_stage": _rounded(
                earlier_stage_peak_gpus
            ),
            "lifetime_reservation_fraction_of_cluster_gpu_seconds": (
                _rounded(exact_gpu_seconds / cluster_gpu_seconds)
                if cluster_gpu_seconds > 0
                else None
            ),
            "actors": actor_rows,
            "by_stage": by_stage,
        },
    }
    return summary, timeline


def render_csv(timeline: Sequence[Mapping[str, float]], source_sha256: str) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in timeline:
        writer.writerow(
            {
                "source_workload_sha256": source_sha256,
                "event_measure": EVENT_MEASURE,
                "gcs_corroboration": GCS_CORROBORATION,
                **{key: f"{float(row[key]):.6f}" for key in TIMELINE_FIELDS},
            }
        )
    return stream.getvalue()


def render_svg(
    summary: Mapping[str, object], timeline: Sequence[Mapping[str, float]]
) -> str:
    width = 1080
    left, right = 232, 28
    plot_width = width - left - right
    row_height, row_gap = 30, 10
    plot_top = 142
    plot_height = len(ROLE_DEFINITIONS) * (row_height + row_gap) - row_gap
    footer_top = plot_top + plot_height + 38
    height = footer_top + 112
    end = max(1e-9, float(timeline[-1]["elapsed_s"]))
    max_gpus = max(1.0, max(float(row["cluster_gpus"]) for row in timeline))

    def x(value: float) -> float:
        return left + plot_width * max(0.0, min(end, value)) / end

    source = _mapping(summary["source"], "summary source")
    premature = _mapping(
        summary["premature_downstream"], "premature downstream summary"
    )
    workload = _mapping(summary["workload"], "workload summary")
    escaped_sha = html.escape(str(source["sha256"]))
    title = html.escape(
        f"GPU ownership: {workload.get('arm')} / {workload.get('workload')} / "
        f"{workload.get('topology_max_gpus')} GPUs"
    )
    lifetime = (
        "Constructor-to-stage-input lifetime reservation: "
        f"{float(premature['lifetime_reservation_gpu_seconds']):.2f} GPU-s "
        f"across {int(premature['actor_count'])} actor(s)"
    )
    earlier_stage = (
        "Sampled while an earlier stage was active: "
        f"{float(premature['premature_downstream_gpu_seconds_during_earlier_stage']):.2f} "
        "GPU-s; peak "
        f"{float(premature['peak_premature_downstream_gpus_during_earlier_stage']):g} GPUs"
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="title description">',
        '<title id="title">GPU ownership timeline</title>',
        f'<desc id="description">{html.escape(lifetime)}. '
        f"{html.escape(earlier_stage)}. Source SHA-256 "
        f"{escaped_sha}.</desc>",
        '<defs><pattern id="waste-hatch" width="6" height="6" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="#fee2e2"/>'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#dc2626" '
        'stroke-width="2"/></pattern></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="sans-serif" font-size="20" '
        f'font-weight="700" fill="#0f172a">{title}</text>',
        f'<text x="24" y="55" font-family="sans-serif" font-size="14" '
        f'font-weight="600" fill="#b91c1c">{html.escape(lifetime)}</text>',
        f'<text x="24" y="76" font-family="sans-serif" font-size="13" '
        f'font-weight="600" fill="#b91c1c">{html.escape(earlier_stage)}</text>',
        f'<text x="24" y="97" font-family="monospace" font-size="10" '
        f'fill="#475569">source workload SHA-256: {escaped_sha}</text>',
        '<text x="24" y="119" font-family="sans-serif" font-size="11" '
        'fill="#475569">red hatch = constructor-ready to stage-first-input '
        "lifetime reservation; GCS ALIVE corroborated, not continuously sampled</text>",
    ]
    for index, (role, label, color) in enumerate(ROLE_DEFINITIONS):
        y = plot_top + index * (row_height + row_gap)
        peak = max(float(row[role]) for row in timeline)
        lines.extend(
            (
                f'<text x="{left - 10}" y="{y + 19}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12" fill="#0f172a">'
                f"{html.escape(label)} (peak {peak:g})</text>",
                f'<rect x="{left}" y="{y}" width="{plot_width}" '
                f'height="{row_height}" rx="2" fill="#f8fafc" '
                'stroke="#e2e8f0"/>',
            )
        )
        for current, following in zip(timeline, timeline[1:]):
            value = float(current[role])
            if value <= 0:
                continue
            x0, x1 = x(float(current["elapsed_s"])), x(float(following["elapsed_s"]))
            bar_height = max(2.0, (row_height - 4) * min(value, max_gpus) / max_gpus)
            lines.append(
                f'<rect class="ownership {role}" x="{x0:.2f}" '
                f'y="{y + row_height - 2 - bar_height:.2f}" '
                f'width="{max(0.5, x1 - x0):.2f}" height="{bar_height:.2f}" '
                f'fill="{color}" opacity="0.82"/>'
            )

    role_y = {
        role: plot_top + index * (row_height + row_gap)
        for index, (role, _, _) in enumerate(ROLE_DEFINITIONS)
    }
    for actor in _sequence(premature["actors"], "premature actors"):
        item = _mapping(actor, "premature actor")
        role = str(item["role"])
        y = role_y[role]
        x0 = x(float(item["ready_s"]))
        x1 = x(float(item["stage_first_input_s"]))
        lines.append(
            f'<rect class="premature-allocation {role}" x="{x0:.2f}" '
            f'y="{y + 2}" width="{max(1.0, x1 - x0):.2f}" '
            f'height="{row_height - 4}" fill="url(#waste-hatch)" '
            'stroke="#dc2626" stroke-width="1.5"/>'
        )

    marker_colors = {
        "gpu-map-add-key": "#2563eb",
        "gpu-map-groups-1": "#d97706",
        "gpu-map-final": "#059669",
    }
    for marker in _sequence(summary["stage_markers"], "stage markers"):
        item = _mapping(marker, "stage marker")
        stage = str(item["stage"])
        color = marker_colors[stage]
        input_x = x(float(item["first_input_s"]))
        progress_x = x(float(item["first_progress_s"]))
        lines.append(
            f'<line class="first-input {stage}" x1="{input_x:.2f}" '
            f'x2="{input_x:.2f}" y1="{plot_top - 5}" '
            f'y2="{plot_top + plot_height + 5}" stroke="{color}" '
            'stroke-width="1.4"/>'
        )
        lines.append(
            f'<line class="first-progress {stage}" x1="{progress_x:.2f}" '
            f'x2="{progress_x:.2f}" y1="{plot_top - 5}" '
            f'y2="{plot_top + plot_height + 5}" stroke="{color}" '
            'stroke-width="1.2" stroke-dasharray="4 3"/>'
        )

    for tick in range(6):
        value = end * tick / 5
        tick_x = x(value)
        lines.extend(
            (
                f'<line x1="{tick_x:.2f}" x2="{tick_x:.2f}" '
                f'y1="{plot_top + plot_height}" y2="{plot_top + plot_height + 5}" '
                'stroke="#64748b"/>',
                f'<text x="{tick_x:.2f}" y="{plot_top + plot_height + 19}" '
                'text-anchor="middle" font-family="sans-serif" font-size="10" '
                f'fill="#475569">{value:.1f}s</text>',
            )
        )
    lines.append(
        f'<text x="24" y="{footer_top}" font-family="sans-serif" '
        'font-size="12" font-weight="700" fill="#0f172a">Stage event markers</text>'
    )
    for index, marker in enumerate(
        _sequence(summary["stage_markers"], "stage markers")
    ):
        item = _mapping(marker, "stage marker")
        stage = str(item["stage"])
        y = footer_top + 21 + index * 18
        lines.append(
            f'<text x="24" y="{y}" font-family="monospace" font-size="11" '
            f'fill="{marker_colors[stage]}">{html.escape(stage)}: '
            f"first input {float(item['first_input_s']):.3f}s; first progress "
            f"{float(item['first_progress_s']):.3f}s</text>"
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_workload(source_path: Path, output_prefix: Path) -> tuple[Path, Path, Path]:
    source_bytes = source_path.read_bytes()
    try:
        document = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid workload JSON: {error}") from error
    summary, timeline = build_evidence(
        _mapping(document, "workload document"),
        source_name=source_path.name,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    outputs = (
        Path(f"{output_prefix}.svg"),
        Path(f"{output_prefix}.csv"),
        Path(f"{output_prefix}.json"),
    )
    if source_path.resolve() in {path.resolve() for path in outputs}:
        raise EvidenceError("output path would overwrite the source workload")
    svg = render_svg(summary, timeline)
    csv_text = render_csv(timeline, str(summary["source"]["sha256"]))
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    for path, content in zip(outputs, (svg, csv_text, json_text)):
        _atomic_text(path, content)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", type=Path, help="completed workload.json")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="artifact prefix (default: gpu-ownership-timeline beside workload)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prefix = args.output_prefix or args.workload.parent / "gpu-ownership-timeline"
    try:
        outputs = render_workload(args.workload, prefix)
    except (EvidenceError, OSError) as error:
        raise SystemExit(f"cannot render GPU ownership evidence: {error}") from error
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
