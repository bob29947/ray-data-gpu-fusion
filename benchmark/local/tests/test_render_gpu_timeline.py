from __future__ import annotations

import csv
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from benchmark.local import render_gpu_timeline


ORIGIN = 1_900_000_000.0
MONOTONIC_ORIGIN = 10_000.0
BOOT_ID = "boot-fixture"
JOB_ID = "01000000"
QUIESCENCE = {
    "complete": True,
    "stable_sequence": 30,
    "stable_polls": 3,
    "required_polls": 3,
    "stable_duration_s": 0.5,
    "required_duration_s": 0.5,
}


def _actor(actor_id: str, class_name: str) -> dict[str, object]:
    return {
        "actor_id": actor_id,
        "class_name": class_name,
        "job_id": JOB_ID,
        "num_restarts": 0,
        "state": "ALIVE",
        "required_resources": {"GPU": 1.0},
    }


def _progress(*, final: bool) -> dict[str, object]:
    document: dict[str, object] = {
        "origin_epoch_s": ORIGIN,
        "origin_monotonic_s": MONOTONIC_ORIGIN,
        "origin_boot_id": BOOT_ID,
        "sequence": 0,
        "stages": {},
        "actors": {},
        "expected_actor_counts": {
            "gpu-map-groups-1": 1,
            "gpu-map-final": 2,
        },
    }
    if final:
        document.update(
            sequence=30,
            stages={
                "gpu-map-add-key": {
                    "first_input_s": 1.0,
                    "first_input_monotonic_s": MONOTONIC_ORIGIN + 1.0,
                    "first_input_boot_id": BOOT_ID,
                    "first_input_clock_basis": "single-boot-monotonic",
                    "first_input_rows_sampled": 1,
                    "first_progress_s": 1.5,
                    "first_progress_monotonic_s": MONOTONIC_ORIGIN + 1.5,
                    "first_progress_boot_id": BOOT_ID,
                    "first_progress_clock_basis": "single-boot-monotonic",
                },
                "gpu-map-groups-1": {
                    "first_input_s": 8.0,
                    "first_input_monotonic_s": MONOTONIC_ORIGIN + 8.0,
                    "first_input_boot_id": BOOT_ID,
                    "first_input_clock_basis": "single-boot-monotonic",
                    "first_input_rows_sampled": 1,
                    "first_progress_s": 9.0,
                    "first_progress_monotonic_s": MONOTONIC_ORIGIN + 9.0,
                    "first_progress_boot_id": BOOT_ID,
                    "first_progress_clock_basis": "single-boot-monotonic",
                },
                "gpu-map-final": {
                    "first_input_s": 12.0,
                    "first_input_monotonic_s": MONOTONIC_ORIGIN + 12.0,
                    "first_input_boot_id": BOOT_ID,
                    "first_input_clock_basis": "single-boot-monotonic",
                    "first_input_rows_sampled": 1,
                    "first_progress_s": 13.0,
                    "first_progress_monotonic_s": MONOTONIC_ORIGIN + 13.0,
                    "first_progress_boot_id": BOOT_ID,
                    "first_progress_clock_basis": "single-boot-monotonic",
                },
            },
            actors={
                "upstream": {
                    "stage": "gpu-map-add-key",
                    "gpus": 1.0,
                    "ready_s": 0.8,
                    "ready_monotonic_s": MONOTONIC_ORIGIN + 0.8,
                    "ready_boot_id": BOOT_ID,
                    "ready_clock_basis": "single-boot-monotonic",
                    "registration_events": 1,
                    "first_input_s": 1.0,
                },
                "groups": {
                    "stage": "gpu-map-groups-1",
                    "gpus": 1.0,
                    "ready_s": 2.0,
                    "ready_monotonic_s": MONOTONIC_ORIGIN + 2.0,
                    "ready_boot_id": BOOT_ID,
                    "ready_clock_basis": "single-boot-monotonic",
                    "registration_events": 1,
                    "first_input_s": 8.0,
                },
                "final": {
                    "stage": "gpu-map-final",
                    "gpus": 1.0,
                    "ready_s": 2.5,
                    "ready_monotonic_s": MONOTONIC_ORIGIN + 2.5,
                    "ready_boot_id": BOOT_ID,
                    "ready_clock_basis": "single-boot-monotonic",
                    "registration_events": 1,
                    "first_input_s": 12.0,
                },
                "idle-final": {
                    "stage": "gpu-map-final",
                    "gpus": 1.0,
                    "ready_s": 3.0,
                    "ready_monotonic_s": MONOTONIC_ORIGIN + 3.0,
                    "ready_boot_id": BOOT_ID,
                    "ready_clock_basis": "single-boot-monotonic",
                    "registration_events": 1,
                },
            },
            telemetry_coverage={"complete": True},
            telemetry_quiescence=dict(QUIESCENCE),
        )
    return document


def _sample(
    elapsed: float, actors: list[dict[str, object]], *, final: bool = False
) -> dict[str, object]:
    owned = len(actors)
    return {
        "elapsed_s": elapsed,
        "origin_epoch_s": ORIGIN,
        "origin_monotonic_s": MONOTONIC_ORIGIN,
        "origin_boot_id": BOOT_ID,
        "cluster": {"GPU": 5.0},
        "available": {"GPU": float(5 - owned)},
        "progress": _progress(final=final),
        "ray_state": {
            "actors": actors,
            "tasks": [],
            "placement_groups": [],
            "counts": {
                "actors": {"ALIVE": owned, "DEAD": 3},
                "tasks": {},
                "placement_groups": {},
            },
            "query_limit": 2_000,
            "possibly_truncated": False,
        },
    }


def _workload() -> dict[str, object]:
    all_actors = [
        _actor("upstream", "MapWorker(MapBatches(AddKey))"),
        _actor("shuffle", "GPUShuffleActor"),
        _actor("groups", "MapWorker(MapBatches(SumGroup))"),
        _actor("final", "MapWorker(MapBatches(Identity))"),
        _actor("idle-final", "MapWorker(MapBatches(Identity))"),
    ]
    return {
        "schema_version": 1,
        "status": "success",
        "arm": "stock",
        "workload": "incident",
        "topology": "local-g5",
        "topology_max_gpus": 5,
        "rows": 1_000_000,
        "blocks": 32,
        "groups": 64,
        "shuffle_ranks": 1,
        "ray_job_id": JOB_ID,
        "telemetry_origin_epoch_s": ORIGIN,
        "telemetry_origin_monotonic_s": MONOTONIC_ORIGIN,
        "telemetry_origin_boot_id": BOOT_ID,
        "telemetry_drain": {
            "complete": True,
            "quiescence": dict(QUIESCENCE),
        },
        "resource_samples": [
            _sample(0.0, []),
            _sample(3.0, all_actors),
            _sample(6.0, all_actors),
            _sample(9.0, all_actors),
            _sample(12.0, all_actors),
            _sample(14.0, [], final=True),
        ],
        "final_progress": _progress(final=True),
        "resource_metrics": {
            "premature_downstream_ownership": {
                "measurement_complete": True,
                "common_clock_verified": True,
                "clock_basis": "single-boot-monotonic",
                "telemetry_quiescence": dict(QUIESCENCE),
                "gcs_state_samples_valid": True,
                "gcs_corroboration_complete": True,
                "gcs_corroboration_issue": None,
                "gcs_corroborated_actor_count": 3,
                "gcs_expected_actor_count": 3,
                "premature_downstream_actor_count": 3,
                "premature_downstream_gpu_seconds": 24.5,
                "premature_downstream_gpu_seconds_during_earlier_stage": 24.0,
                "peak_premature_downstream_gpus_during_earlier_stage": 3.0,
            }
        },
    }


def _write_workload(path: Path, document: dict[str, object]) -> bytes:
    content = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    return content


def _edit_final_progress(document: dict[str, object], edit) -> None:
    edit(document["final_progress"])
    edit(document["resource_samples"][-1]["progress"])


def _stale_sample_progress_clock(document: dict[str, object]) -> None:
    document["resource_samples"][1]["progress"]["origin_boot_id"] = "stale-boot"


def _stale_stage_event(document: dict[str, object]) -> None:
    _edit_final_progress(
        document,
        lambda progress: progress["stages"]["gpu-map-groups-1"].update(
            first_input_s=8.25
        ),
    )


def _stale_actor_ready(document: dict[str, object]) -> None:
    _edit_final_progress(
        document,
        lambda progress: progress["actors"]["groups"].update(ready_s=2.25),
    )


def _duplicate_registration_event(document: dict[str, object]) -> None:
    _edit_final_progress(
        document,
        lambda progress: progress["actors"]["groups"].update(registration_events=2),
    )


def _zero_first_input_rows(document: dict[str, object]) -> None:
    _edit_final_progress(
        document,
        lambda progress: progress["stages"]["gpu-map-groups-1"].update(
            first_input_rows_sampled=0
        ),
    )


def _false_quiescence_with_complete_drain(document: dict[str, object]) -> None:
    _edit_final_progress(
        document,
        lambda progress: progress["telemetry_quiescence"].update(complete=False),
    )


def _inconsistent_quiescence_proof(document: dict[str, object]) -> None:
    _edit_final_progress(
        document,
        lambda progress: progress["telemetry_quiescence"].update(stable_polls=1),
    )
    document["telemetry_drain"]["quiescence"]["stable_polls"] = 1
    document["resource_metrics"]["premature_downstream_ownership"][
        "telemetry_quiescence"
    ]["stable_polls"] = 1


def _hidden_gcs_issue(document: dict[str, object]) -> None:
    document["resource_metrics"]["premature_downstream_ownership"][
        "gcs_corroboration_issue"
    ] = ["hand-edited issue"]


def test_render_workload_is_deterministic_read_only_and_causal(tmp_path: Path) -> None:
    source = tmp_path / "workload.json"
    source_bytes = _write_workload(source, _workload())
    prefix = tmp_path / "reviewer-gpu-timeline"

    outputs = render_gpu_timeline.render_workload(source, prefix)
    first_outputs = [path.read_bytes() for path in outputs]
    render_gpu_timeline.render_workload(source, prefix)

    assert source.read_bytes() == source_bytes
    assert [path.read_bytes() for path in outputs] == first_outputs
    svg_path, csv_path, json_path = outputs
    assert (
        'class="premature-allocation post_shuffle_map_groups"' in svg_path.read_text()
    )
    assert (
        'class="premature-allocation post_shuffle_map_batches"' in svg_path.read_text()
    )
    assert "first input 8.000s; first progress 9.000s" in svg_path.read_text()
    assert "GCS ALIVE corroborated, not continuously sampled" in svg_path.read_text()
    ET.fromstring(svg_path.read_text())

    rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
    row_at_three = next(row for row in rows if row["elapsed_s"] == "3.000000")
    row_at_nine = next(row for row in rows if row["elapsed_s"] == "9.000000")
    assert row_at_three["post_shuffle_map_groups"] == "1.000000"
    assert row_at_three["lifetime_reserved_post_shuffle_map_groups_gpus"] == (
        "1.000000"
    )
    assert row_at_three["lifetime_reserved_post_shuffle_map_batches_gpus"] == (
        "2.000000"
    )
    assert row_at_nine["lifetime_reserved_post_shuffle_map_groups_gpus"] == ("0.000000")
    assert row_at_nine["lifetime_reserved_post_shuffle_map_batches_gpus"] == (
        "2.000000"
    )
    assert row_at_three["premature_downstream_gpus_during_earlier_stage"] == (
        "3.000000"
    )
    assert (
        row_at_three["source_workload_sha256"]
        == hashlib.sha256(source_bytes).hexdigest()
    )
    assert "constructor-ready" in row_at_three["event_measure"]
    assert "not continuous sampling proof" in row_at_three["gcs_corroboration"]

    summary = json.loads(json_path.read_text())
    assert summary["source"] == {
        "filename": "workload.json",
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    assert summary["causal_telemetry"]["complete"] is True
    assert summary["premature_downstream"]["actor_count"] == 3
    assert summary["premature_downstream"]["lifetime_reservation_gpu_seconds"] == 24.5
    assert (
        summary["premature_downstream"][
            "premature_downstream_gpu_seconds_during_earlier_stage"
        ]
        == 24.0
    )
    assert (
        summary["premature_downstream"][
            "peak_premature_downstream_gpus_during_earlier_stage"
        ]
        == 3.0
    )
    assert summary["premature_downstream"]["gcs_corroboration"]["complete"] is True
    assert (
        "not continuous"
        in summary["premature_downstream"]["gcs_corroboration"]["interpretation"]
    )
    assert [actor["actor"] for actor in summary["premature_downstream"]["actors"]] == [
        "actor-001",
        "actor-002",
        "actor-003",
    ]
    idle_actor = next(
        actor
        for actor in summary["premature_downstream"]["actors"]
        if actor["actor_first_input_s"] is None
    )
    assert idle_actor["stage"] == "gpu-map-final"
    assert all(
        actor["gcs_alive_observation_count"] >= 1
        for actor in summary["premature_downstream"]["actors"]
    )
    assert '"actor_id"' not in json_path.read_text()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda document: document.update(telemetry_drain={"complete": False}),
            "telemetry drain is incomplete",
        ),
        (
            lambda document: document["resource_metrics"][
                "premature_downstream_ownership"
            ].update(measurement_complete=False),
            "premature downstream measurement is incomplete",
        ),
        (
            lambda document: (
                document["resource_samples"][-1]["progress"]["stages"].pop(
                    "gpu-map-final"
                ),
                document["final_progress"]["stages"].pop("gpu-map-final"),
            ),
            "progress stage gpu-map-final must be an object",
        ),
    ],
)
def test_build_evidence_fails_closed_on_incomplete_causal_telemetry(
    mutation, message: str
) -> None:
    document = _workload()
    mutation(document)

    with pytest.raises(render_gpu_timeline.EvidenceError, match=message):
        render_gpu_timeline.build_evidence(
            document,
            source_name="workload.json",
            source_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (_stale_sample_progress_clock, "does not match the top-level clock origin"),
        (_stale_stage_event, "not verified single-boot monotonic telemetry"),
        (_stale_actor_ready, "invalid lifecycle telemetry"),
        (_duplicate_registration_event, "invalid lifecycle telemetry"),
        (_zero_first_input_rows, "sampled rows must be positive"),
        (_false_quiescence_with_complete_drain, "did not become quiescent"),
        (_inconsistent_quiescence_proof, "quiescence proof is inconsistent"),
        (_hidden_gcs_issue, "producer reports GCS corroboration issues"),
    ],
)
def test_build_evidence_rejects_stale_or_hand_edited_schema(
    mutation, message: str
) -> None:
    document = _workload()
    mutation(document)

    with pytest.raises(render_gpu_timeline.EvidenceError, match=message):
        render_gpu_timeline.build_evidence(
            document,
            source_name="workload.json",
            source_sha256="0" * 64,
        )


def test_build_evidence_rejects_external_gpu_ownership() -> None:
    document = _workload()
    document["resource_samples"][1]["ray_state"]["actors"][0]["job_id"] = "other"

    with pytest.raises(render_gpu_timeline.EvidenceError, match="external job"):
        render_gpu_timeline.build_evidence(
            document,
            source_name="workload.json",
            source_sha256="0" * 64,
        )


def test_build_evidence_rejects_uncorroborated_premature_actor() -> None:
    document = _workload()
    for sample in document["resource_samples"]:
        if sample["elapsed_s"] < 8.0:
            sample["ray_state"]["actors"] = [
                actor
                for actor in sample["ray_state"]["actors"]
                if actor["actor_id"] != "groups"
            ]

    with pytest.raises(
        render_gpu_timeline.EvidenceError, match="no in-window GCS ALIVE"
    ):
        render_gpu_timeline.build_evidence(
            document,
            source_name="workload.json",
            source_sha256="0" * 64,
        )


def test_build_evidence_rejects_restarted_premature_actor() -> None:
    document = _workload()
    for sample in document["resource_samples"]:
        for actor in sample["ray_state"]["actors"]:
            if actor["actor_id"] == "groups":
                actor["num_restarts"] = 1

    with pytest.raises(render_gpu_timeline.EvidenceError, match="restart history"):
        render_gpu_timeline.build_evidence(
            document,
            source_name="workload.json",
            source_sha256="0" * 64,
        )


def test_build_evidence_rejects_conflicting_premature_actor_record() -> None:
    document = _workload()
    groups = next(
        actor
        for actor in document["resource_samples"][1]["ray_state"]["actors"]
        if actor["actor_id"] == "groups"
    )
    groups["class_name"] = "MapWorker(MapBatches(Identity))"

    with pytest.raises(render_gpu_timeline.EvidenceError, match="conflicting operator"):
        render_gpu_timeline.build_evidence(
            document,
            source_name="workload.json",
            source_sha256="0" * 64,
        )


def test_render_workload_refuses_to_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "evidence.json"
    original = _write_workload(source, _workload())

    with pytest.raises(render_gpu_timeline.EvidenceError, match="overwrite"):
        render_gpu_timeline.render_workload(source, tmp_path / "evidence")

    assert source.read_bytes() == original
