#!/usr/bin/env python3
"""Plan and, only with --execute-local, run single-host GPU evidence.

Each case gets an isolated Ray cluster, wheel overlay, bounded workload driver,
post-job actor/placement-group audit, and process-level teardown proof.  The
default action only writes the randomized plan and cloud-prerequisite report.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import stat
import statistics
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark import evidence_statistics  # noqa: E402
from benchmark.local.cluster_daemon import (  # noqa: E402
    snapshot_object_spilling_directory,
    validate_object_spilling_directory,
)


RESULTS_ROOT = ROOT / "benchmark" / "results" / "local"
SHM_ROOT = Path("/dev/shm/ray-admission")
INCIDENT_WORKLOAD = ROOT / "benchmark" / "cloud" / "incident_workload.py"
SCALE_DEFAULT_ROWS = 4_000_000_000
SCALE_DEFAULT_BLOCKS = 1_024
DGX_CAPACITIES = (1, 2, 4, 8, 16)
DGX_STOCK_WORKAROUNDS = {
    capacity: tuple(
        sorted(
            {
                (map_actors, shuffle_ranks)
                for map_actors in range(1, (capacity - 1) // 3 + 1)
                for shuffle_ranks in (1, capacity - 3 * map_actors)
            }
        )
    )
    for capacity in (4, 8, 16)
}
DGX_STOCK_COMPLETING_CAPACITIES = tuple(DGX_STOCK_WORKAROUNDS)
DGX_STOCK_WORKAROUND_COUNT = sum(
    len(values) for values in DGX_STOCK_WORKAROUNDS.values()
)
DGX_CASES_PER_REPETITION = len(DGX_CAPACITIES) * 2 + DGX_STOCK_WORKAROUND_COUNT + 2
DGX_DEFAULT_REPETITIONS = 1
DGX_WEAK_ROWS_PER_GPU = 1_000_000_000
DGX_WEAK_BLOCKS_PER_GPU = 256
DGX_ACTOR_CONTROL_CAPACITY = 16
DGX_ACTOR_CONTROL_MAP_ACTORS = DGX_ACTOR_CONTROL_CAPACITY // 3
DGX_OBJECT_STORE_BYTES_PER_GPU = 8 * 1024**3
DGX_MIN_CPUS = 64
ARM_NAMES = ("stock", "pg-only", "minimal", "prototype")
WORKLOADS = (
    "incident",
    "aggregate-cpu-gap",
    "actor-only",
    "map-heavy",
    "shuffle-heavy",
    "forced-spill",
    "fan-in",
    "failure-cleanup",
)
SHUFFLE_WORKLOADS = frozenset(
    {
        "incident",
        "aggregate-cpu-gap",
        "shuffle-heavy",
        "forced-spill",
        "failure-cleanup",
    }
)
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
LOCAL_STAGED_HARNESS_FILES = (
    "benchmark/__init__.py",
    "benchmark/evidence_statistics.py",
    "benchmark/ray_core_state.py",
    "benchmark/cloud/__init__.py",
    "benchmark/cloud/incident_workload.py",
    "benchmark/local/__init__.py",
    "benchmark/local/cluster_daemon.py",
    "benchmark/local/probe.py",
    "benchmark/local/run_evidence.py",
)
LOCAL_STAGED_HARNESS_MANIFEST = "MANIFEST.json"


@dataclass(frozen=True)
class ArmDefinition:
    name: str
    wheel_layer: str
    manifest: str
    admission_enabled: bool


@dataclass(frozen=True)
class LocalCase:
    case_id: str
    arm: str
    capacity: int
    workload: str
    shuffle_ranks: int | str
    repetition: int
    map_actors_per_stage: int | str = "capacity"
    comparison_id: str | None = None
    scaling_mode: str = "configured"
    rows: int | None = None
    blocks: int | None = None
    workaround_id: str | None = None


ARMS = {
    item.name: item
    for item in (
        ArmDefinition("stock", "stock", "pins/stock-ray.json", False),
        ArmDefinition("pg-only", "pr-candidate", "pins/pr-candidate.json", False),
        ArmDefinition("minimal", "pr-candidate", "pins/pr-candidate.json", True),
        ArmDefinition("prototype", "prototype", "pins/prototype.json", True),
    )
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(document: object) -> bytes:
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode()


def _verify_local_harness(directory: Path, expected_sha256: str) -> dict[str, object]:
    manifest_path = directory / LOCAL_STAGED_HARNESS_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid staged local harness: {manifest_path}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "files",
        "content_sha256",
    }:
        raise ValueError("staged local harness manifest has an invalid schema")
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict):
        raise ValueError("staged local harness manifest has an invalid schema")
    core = {"schema_version": 1, "files": files}
    content_sha256 = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    if (
        content_sha256 != expected_sha256
        or manifest["content_sha256"] != expected_sha256
    ):
        raise ValueError("staged local harness digest does not match its manifest")
    total_size = manifest_path.stat().st_size
    for relative_name, entry in files.items():
        relative = Path(relative_name)
        if (
            not isinstance(relative_name, str)
            or not relative_name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_name
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "size_bytes"}
        ):
            raise ValueError(f"unsafe staged local harness entry: {relative_name!r}")
        path = directory / relative
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not path.is_file()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise ValueError(
                f"staged local harness file failed verification: {relative_name}"
            )
        total_size += size
    return {
        "directory": str(directory.resolve()),
        "content_sha256": content_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": files,
    }


def stage_local_harness(run_dir: Path) -> dict[str, object]:
    """Freeze every Python input used across a multi-case local campaign."""

    missing = [
        ROOT / relative_name
        for relative_name in LOCAL_STAGED_HARNESS_FILES
        if not (ROOT / relative_name).is_file()
    ]
    if missing:
        raise ValueError(f"staged local harness source is missing: {missing[0]}")
    files = {
        relative_name: {
            "sha256": _sha256_file(ROOT / relative_name),
            "size_bytes": (ROOT / relative_name).stat().st_size,
        }
        for relative_name in LOCAL_STAGED_HARNESS_FILES
    }
    core = {"schema_version": 1, "files": files}
    content_sha256 = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    directory = run_dir / f"harness-{content_sha256[:16]}"
    if directory.exists():
        return _verify_local_harness(directory, content_sha256)
    directory.mkdir(parents=True)
    for relative_name in LOCAL_STAGED_HARNESS_FILES:
        source = ROOT / relative_name
        destination = directory / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o444)
    manifest_path = directory / LOCAL_STAGED_HARNESS_MANIFEST
    _atomic_json(manifest_path, {**core, "content_sha256": content_sha256})
    manifest_path.chmod(0o444)
    return _verify_local_harness(directory, content_sha256)


def _csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def _positive_csv(value: str) -> list[int]:
    try:
        values = [int(item) for item in _csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be positive integers") from error
    if any(item < 1 for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be distinct positive integers")
    return values


def _nonnegative_csv(value: str) -> list[int]:
    try:
        values = [int(item) for item in _csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be nonnegative integers"
        ) from error
    if any(item < 0 for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be distinct nonnegative integers")
    return values


def _rank_csv(value: str) -> list[int | str]:
    if value == "all":
        return ["all"]
    result: list[int | str] = []
    for item in _csv(value):
        if item == "default":
            parsed: int | str = "default"
        else:
            try:
                parsed = int(item)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "ranks must be positive integers, 'default', or 'all'"
                ) from error
            if parsed < 1:
                raise argparse.ArgumentTypeError("ranks must be positive")
        if parsed in result:
            raise argparse.ArgumentTypeError("ranks must be distinct")
        result.append(parsed)
    return result


def _validate_selection(
    values: Iterable[str], allowed: Iterable[str], label: str
) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")


def _ranks_for(
    workload: str, capacity: int, requested: Sequence[int | str]
) -> list[int | str]:
    if workload not in SHUFFLE_WORKLOADS:
        return ["default"]
    if requested == ["all"]:
        return [*range(1, capacity + 1), "default"]
    ranks = [item for item in requested if item == "default" or int(item) <= capacity]
    if not ranks:
        raise ValueError(f"no selected rank fits capacity {capacity}")
    return ranks


def build_cases(
    *,
    run_id: str,
    arms: Sequence[str],
    capacities: Sequence[int],
    workloads: Sequence[str],
    ranks: Sequence[int | str],
    repetitions: int,
    seed: int,
) -> list[LocalCase]:
    cases = []
    for repetition in range(1, repetitions + 1):
        for arm in arms:
            for capacity in capacities:
                for workload in workloads:
                    for rank in _ranks_for(workload, capacity, ranks):
                        token = "default" if rank == "default" else f"r{rank}"
                        case_id = f"{run_id}-{arm}-g{capacity}-{workload}-{token}-n{repetition}"
                        cases.append(
                            LocalCase(
                                case_id=case_id,
                                arm=arm,
                                capacity=capacity,
                                workload=workload,
                                shuffle_ranks=rank,
                                repetition=repetition,
                            )
                        )
    random.Random(seed).shuffle(cases)
    return cases


def build_scale_evidence_cases(
    *, run_id: str, repetitions: int, seed: int, capacity: int = 4
) -> list[LocalCase]:
    """Build the focused, paired scale campaign.

    The incident comparison gives stock Ray its best completing workaround
    (one persistent map actor per stage and a rank-one shuffle), while the
    candidate uses every GPU for both map and shuffle work.  The actor-only
    comparison holds the resource shape equal to catch ordinary-workload
    regressions independently of shuffle tuning.
    """

    blocks = []
    configurations = (
        ("stock", "incident", 1, 1, "incident-scale-speedup"),
        ("minimal", "incident", capacity, capacity, "incident-scale-speedup"),
        ("stock", "actor-only", "default", capacity // 3, "actor-only-regression"),
        (
            "minimal",
            "actor-only",
            "default",
            capacity // 3,
            "actor-only-regression",
        ),
    )
    generator = random.Random(seed)
    for repetition in range(1, repetitions + 1):
        block = []
        for arm, workload, rank, map_actors, comparison_id in configurations:
            rank_token = "default" if rank == "default" else f"r{rank}"
            case_id = (
                f"{run_id}-{comparison_id}-{arm}-g{capacity}-{rank_token}"
                f"-m{map_actors}-n{repetition}"
            )
            block.append(
                LocalCase(
                    case_id=case_id,
                    arm=arm,
                    capacity=capacity,
                    workload=workload,
                    shuffle_ranks=rank,
                    repetition=repetition,
                    map_actors_per_stage=map_actors,
                    comparison_id=comparison_id,
                )
            )
        generator.shuffle(block)
        blocks.append(block)
    generator.shuffle(blocks)
    return [case for block in blocks for case in block]


def build_dgx_scale_cases(
    *,
    run_id: str,
    repetitions: int,
    seed: int,
    strong_rows: int = SCALE_DEFAULT_ROWS,
    strong_blocks: int = SCALE_DEFAULT_BLOCKS,
    weak_rows_per_gpu: int = DGX_WEAK_ROWS_PER_GPU,
    weak_blocks_per_gpu: int = DGX_WEAK_BLOCKS_PER_GPU,
    weak_max_rows: int | None = None,
) -> list[LocalCase]:
    """Build a practical, blocked 16-GPU scaling campaign.

    Strong scaling measures every useful edge of the stock safe frontier
    ``3 * map_actors + shuffle_ranks <= GPUs`` on 4/8/16 GPUs before selecting
    the fastest completing workaround. Candidate one- and two-GPU cases anchor
    liveness and strong-scaling efficiency. Weak scaling keeps roughly one
    billion rows per GPU and runs the candidate at every capacity. A maximum-
    capacity, provably safe actor-only pair is the equal-shape non-shuffle
    regression control.

    Stock is intentionally absent at one and two GPUs: no positive uniform
    rank/map floor fits the incident's three map pools plus shuffle gang. The
    liveness profile owns those structural cases. At larger capacities the
    grid includes rank one and the maximum safe shuffle rank for every feasible
    map-pool size. Duplicate endpoints are removed, producing 1 + 4 + 9 = 14
    measured stock configurations.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    for label, value in (
        ("strong_rows", strong_rows),
        ("strong_blocks", strong_blocks),
        ("weak_rows_per_gpu", weak_rows_per_gpu),
        ("weak_blocks_per_gpu", weak_blocks_per_gpu),
    ):
        if value < 1:
            raise ValueError(f"{label} must be positive")
    if weak_max_rows is not None and weak_max_rows < 1:
        raise ValueError("weak_max_rows must be positive when set")

    generator = random.Random(seed)
    repetition_blocks: list[list[LocalCase]] = []
    for repetition in range(1, repetitions + 1):
        block: list[LocalCase] = []
        for capacity in DGX_CAPACITIES:
            comparison_id = f"dgx-strong-g{capacity}"
            if capacity in DGX_STOCK_COMPLETING_CAPACITIES:
                for map_actors, shuffle_ranks in DGX_STOCK_WORKAROUNDS[capacity]:
                    workaround_id = f"m{map_actors}-r{shuffle_ranks}"
                    if 3 * map_actors + shuffle_ranks > capacity:
                        raise AssertionError("stock workaround exceeds GPU frontier")
                    block.append(
                        LocalCase(
                            case_id=(
                                f"{run_id}-{comparison_id}-stock-{workaround_id}"
                                f"-n{repetition}"
                            ),
                            arm="stock",
                            capacity=capacity,
                            workload="incident",
                            shuffle_ranks=shuffle_ranks,
                            repetition=repetition,
                            map_actors_per_stage=map_actors,
                            comparison_id=comparison_id,
                            scaling_mode="strong",
                            rows=strong_rows,
                            blocks=strong_blocks,
                            workaround_id=workaround_id,
                        )
                    )
            block.append(
                LocalCase(
                    case_id=(
                        f"{run_id}-{comparison_id}-minimal-r{capacity}"
                        f"-m{capacity}-n{repetition}"
                    ),
                    arm="minimal",
                    capacity=capacity,
                    workload="incident",
                    shuffle_ranks=capacity,
                    repetition=repetition,
                    map_actors_per_stage=capacity,
                    comparison_id=comparison_id,
                    scaling_mode="strong",
                    rows=strong_rows,
                    blocks=strong_blocks,
                )
            )

            weak_rows = capacity * weak_rows_per_gpu
            if weak_max_rows is not None:
                weak_rows = min(weak_rows, weak_max_rows)
            weak_blocks = max(
                capacity,
                math.ceil(weak_rows * weak_blocks_per_gpu / weak_rows_per_gpu),
            )
            block.append(
                LocalCase(
                    case_id=(
                        f"{run_id}-dgx-weak-g{capacity}-minimal-r{capacity}"
                        f"-m{capacity}-n{repetition}"
                    ),
                    arm="minimal",
                    capacity=capacity,
                    workload="incident",
                    shuffle_ranks=capacity,
                    repetition=repetition,
                    map_actors_per_stage=capacity,
                    comparison_id=f"dgx-weak-g{capacity}",
                    scaling_mode="weak",
                    rows=weak_rows,
                    blocks=weak_blocks,
                )
            )

        for arm in ("stock", "minimal"):
            block.append(
                LocalCase(
                    case_id=(
                        f"{run_id}-dgx-actor-g{DGX_ACTOR_CONTROL_CAPACITY}"
                        f"-{arm}-n{repetition}"
                    ),
                    arm=arm,
                    capacity=DGX_ACTOR_CONTROL_CAPACITY,
                    workload="actor-only",
                    shuffle_ranks="default",
                    repetition=repetition,
                    map_actors_per_stage=DGX_ACTOR_CONTROL_MAP_ACTORS,
                    comparison_id=f"dgx-actor-g{DGX_ACTOR_CONTROL_CAPACITY}",
                    scaling_mode="actor-control",
                    rows=strong_rows,
                    blocks=strong_blocks,
                )
            )
        generator.shuffle(block)
        repetition_blocks.append(block)
    generator.shuffle(repetition_blocks)
    return [case for block in repetition_blocks for case in block]


def _map_actors_for_case(case: LocalCase, configured: int | None) -> int:
    if isinstance(case.map_actors_per_stage, int):
        return case.map_actors_per_stage
    if case.map_actors_per_stage != "capacity":
        raise ValueError(
            f"unknown map actor policy {case.map_actors_per_stage!r} for {case.case_id}"
        )
    return configured or case.capacity


def _manifest_source_commit(document: Mapping[str, object]) -> str | None:
    for key in ("local_commit", "commit"):
        value = document.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _manifest_base_commit(document: Mapping[str, object]) -> str | None:
    for key in ("base_commit", "commit"):
        value = document.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _installed_ray_commit(wheel: Path) -> str:
    """Read the commit that the wheel will expose as ``ray.__commit__``."""

    try:
        with zipfile.ZipFile(wheel) as archive:
            source = archive.read("ray/_version.py").decode()
    except (KeyError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise ValueError(f"wheel has no readable ray/_version.py: {error}") from error
    tree = ast.parse(source, filename=f"{wheel}!ray/_version.py")
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        names = [
            target.id for target in statement.targets if isinstance(target, ast.Name)
        ]
        if "commit" not in names:
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(
            statement.value.value, str
        ):
            return statement.value.value
    raise ValueError("wheel ray/_version.py has no constant commit assignment")


def inspect_artifacts(selected_arms: Sequence[str]) -> dict[str, object]:
    reports: dict[str, dict[str, object]] = {}
    blockers = []
    # Always inspect all four arms.  A subset is useful for a pilot, but a
    # pilot must not accidentally pronounce the final/prototype separation
    # ready for cloud merely because the prototype was omitted.
    for name in ARM_NAMES:
        arm = ARMS[name]
        manifest_path = ROOT / arm.manifest
        report: dict[str, object] = {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "wheel_layer": arm.wheel_layer,
        }
        try:
            manifest = json.loads(manifest_path.read_text())
            wheel_name = manifest.get("wheel")
            if not isinstance(wheel_name, str) or not wheel_name.endswith(".whl"):
                raise ValueError("manifest has no valid wheel filename")
            wheel = ROOT / "wheels" / arm.wheel_layer / wheel_name
            expected_sha = manifest.get("wheel_sha256")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError("manifest has no valid wheel SHA-256")
            actual_sha = _sha256_file(wheel)
            expected_size = manifest.get("wheel_size_bytes")
            if actual_sha != expected_sha:
                raise ValueError(
                    f"wheel digest mismatch: expected {expected_sha}, got {actual_sha}"
                )
            if expected_size is not None and wheel.stat().st_size != expected_size:
                raise ValueError("wheel size does not match manifest")
            source_commit = _manifest_source_commit(manifest)
            base_commit = _manifest_base_commit(manifest)
            if source_commit is None:
                raise ValueError("manifest has no source commit")
            if base_commit is None:
                raise ValueError("manifest has no base commit")
            installed_ray_commit = _installed_ray_commit(wheel)
            if installed_ray_commit != base_commit:
                raise ValueError(
                    "wheel ray.__commit__ does not match its manifest base: "
                    f"wheel={installed_ray_commit}, manifest={base_commit}"
                )
            report.update(
                status="ready",
                wheel=str(wheel.relative_to(ROOT)),
                wheel_sha256=actual_sha,
                wheel_size_bytes=wheel.stat().st_size,
                source_commit=source_commit,
                base_commit=base_commit,
                installed_ray_commit=installed_ray_commit,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            report.update(
                status="blocked",
                error={"type": type(error).__name__, "message": str(error)},
            )
            blockers.append(f"{name}: {error}")
        reports[name] = report

    candidate = reports.get("minimal") or reports.get("pg-only")
    prototype = reports.get("prototype")
    if (
        candidate
        and prototype
        and candidate.get("status") == prototype.get("status") == "ready"
        and candidate.get("wheel_sha256") == prototype.get("wheel_sha256")
    ):
        blockers.append(
            "final candidate and preserved prototype are byte-identical; rebuild the "
            "simplified candidate before execution"
        )
    return {"ready": not blockers, "arms": reports, "blockers": blockers}


def gpu_inventory() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
        devices = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 5:
                raise ValueError(f"unexpected nvidia-smi row: {line!r}")
            devices.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_mib": int(fields[3]),
                    "driver_version": fields[4],
                }
            )
        return {"status": "ready", "devices": devices, "command": command}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {
            "status": "unavailable",
            "devices": [],
            "command": command,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _overlay_environment(
    overlay: Path, gpu_indices: Sequence[int], harness_root: Path = ROOT
) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(overlay), str(harness_root), existing) if item
    )
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["RAY_USAGE_STATS_ENABLED"] = "0"
    return environment


def install_overlays(
    run_dir: Path, artifact_report: Mapping[str, object], arms: Sequence[str]
) -> dict[str, Path]:
    overlays: dict[str, Path] = {}
    installed_layers: dict[str, Path] = {}
    log_dir = run_dir / "runtime-install"
    log_dir.mkdir(parents=True, exist_ok=True)
    arm_reports = artifact_report["arms"]
    for arm_name in arms:
        definition = ARMS[arm_name]
        if definition.wheel_layer in installed_layers:
            overlays[arm_name] = installed_layers[definition.wheel_layer]
            continue
        report = arm_reports[arm_name]
        overlay = run_dir / "runtime" / definition.wheel_layer
        if overlay.exists():
            raise RuntimeError(f"isolated overlay already exists: {overlay}")
        overlay.mkdir(parents=True)
        wheel = ROOT / str(report["wheel"])
        log = log_dir / f"{definition.wheel_layer}.log"
        with log.open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--target",
                    str(overlay),
                    str(wheel),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        installed_layers[definition.wheel_layer] = overlay
        overlays[arm_name] = overlay
    return overlays


def _wait_for_json(path: Path, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        if process.poll() is not None:
            raise RuntimeError(
                f"cluster daemon exited with {process.returncode} before writing {path.name}"
            )
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


def _run_bounded(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
    terminate_grace_seconds: float = 30.0,
    cwd: Path = ROOT,
) -> dict[str, object]:
    started = time.monotonic()
    with log_path.open("w") as stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        forced_kill = False
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                forced_kill = True
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait(timeout=10)
    return {
        "command": list(command),
        "return_code": return_code,
        "timed_out": timed_out,
        "forced_kill": forced_kill,
        "elapsed_s": time.monotonic() - started,
        "log": str(log_path),
    }


def _address_is_closed(address: object) -> bool:
    if not isinstance(address, str) or ":" not in address:
        return False
    host, raw_port = address.rsplit(":", 1)
    try:
        with socket.create_connection((host.strip("[]"), int(raw_port)), timeout=1):
            return False
    except (OSError, ValueError):
        return True


def _safe_remove_case_tmp(path: Path) -> bool:
    resolved = path.resolve()
    if resolved.parent != SHM_ROOT.resolve() or not resolved.name.startswith("l"):
        raise RuntimeError(f"refusing to remove non-harness temp directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    return not resolved.exists()


def _safe_remove_case_spill(
    path: Path, *, case_dir: Path, results_root: Path = RESULTS_ROOT
) -> bool:
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("refusing spill cleanup without fd-safe shutil.rmtree")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("refusing spill cleanup without O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | nofollow
    case_fd = os.open(case_dir, flags)
    try:
        opened_case = os.fstat(case_fd)
        validated = validate_object_spilling_directory(
            path,
            case_directory=case_dir,
            results_root=results_root,
            must_exist=None,
        )
        if validated.name != "ray-spill":
            raise RuntimeError(f"refusing unexpected spill leaf: {validated}")
        current_case = os.stat(case_dir, follow_symlinks=False)
        if (opened_case.st_dev, opened_case.st_ino) != (
            current_case.st_dev,
            current_case.st_ino,
        ):
            raise RuntimeError("case directory changed during spill cleanup")
        try:
            spill_status = os.stat(
                validated.name,
                dir_fd=case_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return True
        if not stat.S_ISDIR(spill_status.st_mode):
            raise RuntimeError("refusing to remove a non-directory spill leaf")
        if spill_status.st_dev != opened_case.st_dev:
            raise RuntimeError("refusing to traverse a mounted spill leaf")
        shutil.rmtree(validated.name, dir_fd=case_fd)
        try:
            os.stat(validated.name, dir_fd=case_fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False
    finally:
        os.close(case_fd)


def _valid_spill_storage_snapshot(
    snapshot: object, *, spill_dir: Path, require_exists: bool
) -> bool:
    if not isinstance(snapshot, Mapping):
        return False
    nonnegative_integers = (
        "filesystem_device_id",
        "filesystem_device_major",
        "filesystem_device_minor",
        "available_bytes",
        "spill_file_count",
        "spill_file_bytes",
        "scan_race_count",
    )
    return (
        snapshot.get("path") == str(spill_dir)
        and snapshot.get("exists") is require_exists
        and snapshot.get("scan_complete") is True
        and all(
            isinstance(snapshot.get(key), int) and snapshot[key] >= 0
            for key in nonnegative_integers
        )
        and all(
            isinstance(snapshot.get(key), str) and bool(snapshot[key])
            for key in (
                "filesystem_device",
                "filesystem_type",
                "filesystem_mount_point",
                "filesystem_source",
            )
        )
    )


def _same_spill_filesystem(first: object, second: object) -> bool:
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return False
    return all(
        first.get(key) == second.get(key)
        for key in (
            "filesystem_device",
            "filesystem_device_id",
            "filesystem_device_major",
            "filesystem_device_minor",
            "filesystem_type",
            "filesystem_mount_point",
            "filesystem_source",
        )
    )


def _owned_process_union(*manifests: Mapping[str, object]) -> list[dict[str, object]]:
    by_pid: dict[int, dict[str, object]] = {}
    for manifest in manifests:
        processes = manifest.get("owned_processes")
        if not isinstance(processes, list):
            continue
        for process in processes:
            if not isinstance(process, Mapping):
                continue
            pid = process.get("pid")
            if isinstance(pid, int) and pid > 0:
                by_pid.setdefault(pid, dict(process))
    return [by_pid[pid] for pid in sorted(by_pid)]


def _process_group_is_gone(process_group_id: object) -> bool:
    if not isinstance(process_group_id, int) or process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return True
    except (OSError, PermissionError):
        return False
    return False


def _stop_cluster(
    *,
    daemon: subprocess.Popen,
    daemon_log: object,
    stop_request: Path,
    stopped_path: Path,
    ready: Mapping[str, object],
    tmp_dir: Path,
    spill_dir: Path,
    case_dir: Path,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, object]:
    spill_storage_before_shutdown: dict[str, object] | None = None
    spill_storage_after_shutdown: dict[str, object] | None = None
    spill_audit_errors: list[dict[str, str]] = []
    try:
        spill_storage_before_shutdown = snapshot_object_spilling_directory(
            spill_dir,
            case_directory=case_dir,
            results_root=results_root,
        )
    except BaseException as error:
        spill_audit_errors.append(
            {
                "phase": "before_shutdown",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
    stop_request.write_text("stop\n")
    escalated_signal = None
    try:
        daemon.wait(timeout=45)
    except subprocess.TimeoutExpired:
        escalated_signal = "SIGTERM"
        daemon.send_signal(signal.SIGTERM)
        try:
            daemon.wait(timeout=30)
        except subprocess.TimeoutExpired:
            escalated_signal = "SIGKILL"
            os.killpg(daemon.pid, signal.SIGKILL)
            daemon.wait(timeout=10)
    finally:
        daemon_log.close()
    stopped = (
        json.loads(stopped_path.read_text())
        if stopped_path.is_file()
        else {"status": "missing", "owned_processes": ready.get("owned_processes", [])}
    )
    owned = _owned_process_union(ready, stopped)
    alive = [
        item
        for item in owned
        if isinstance(item, dict)
        and isinstance(item.get("pid"), int)
        and Path(f"/proc/{item['pid']}").exists()
    ]
    daemon_process_group_gone = _process_group_is_gone(getattr(daemon, "pid", None))
    cluster_never_started = ready.get("status") == "error" and not owned
    address_closed = cluster_never_started or _address_is_closed(ready.get("address"))
    may_remove_owned_storage = not alive and daemon_process_group_gone
    tmp_removed = _safe_remove_case_tmp(tmp_dir) if may_remove_owned_storage else False
    spill_removed = False
    if may_remove_owned_storage:
        try:
            spill_storage_after_shutdown = snapshot_object_spilling_directory(
                spill_dir,
                case_directory=case_dir,
                results_root=results_root,
            )
            if spill_storage_after_shutdown.get("scan_complete") is not True:
                raise RuntimeError(
                    "object-spilling scan remained unstable after shutdown"
                )
            if spill_storage_before_shutdown is not None and not _same_spill_filesystem(
                spill_storage_before_shutdown, spill_storage_after_shutdown
            ):
                raise RuntimeError(
                    "object-spilling backing filesystem changed during shutdown"
                )
            spill_removed = _safe_remove_case_spill(
                spill_dir,
                case_dir=case_dir,
                results_root=results_root,
            )
        except BaseException as error:
            spill_audit_errors.append(
                {
                    "phase": "after_shutdown_or_removal",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    daemon_exited_as_expected = daemon.returncode == 0 or (
        ready.get("status") == "error" and daemon.returncode == 1
    )
    proven = (
        daemon_exited_as_expected
        and stopped.get("status") == "stopped"
        and not alive
        and daemon_process_group_gone
        and address_closed
        and tmp_removed
        and spill_removed
        and not spill_audit_errors
    )
    return {
        "cleanup_proven": proven,
        "daemon_return_code": daemon.returncode,
        "escalated_signal": escalated_signal,
        "stopped_manifest": stopped,
        "owned_processes_alive": alive,
        "daemon_process_group_gone": daemon_process_group_gone,
        "gcs_address_closed": address_closed,
        "ray_tmpdir_removed": tmp_removed,
        "ray_spill_storage_before_shutdown": spill_storage_before_shutdown,
        "ray_spill_storage_after_shutdown": spill_storage_after_shutdown,
        "ray_spill_directory_removed": spill_removed,
        "ray_spill_audit_errors": spill_audit_errors,
    }


def _load_json_if_present(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text())
        return document if isinstance(document, dict) else {}
    except json.JSONDecodeError:
        return {}


def _outcome(workload: Mapping[str, object], command: Mapping[str, object]) -> str:
    if command.get("timed_out"):
        metrics = workload.get("resource_metrics")
        closed_wait = metrics.get("closed_wait") if isinstance(metrics, dict) else None
        if isinstance(closed_wait, dict) and closed_wait.get(
            "is_structural_closed_wait"
        ):
            return "structural_closed_wait_timeout"
        return "timeout_unclassified"
    status = workload.get("status")
    if status in {"success", "expected_failure"}:
        return str(status)
    return "error"


def _finalize_timeout_metrics(
    workload: Mapping[str, object], *, max_gpus: int
) -> dict[str, object]:
    """Classify persisted samples after a blocked Ray driver is force-killed."""

    document = dict(workload)
    samples = document.get("resource_samples")
    if not isinstance(samples, list) or not samples:
        return document
    from benchmark.cloud.incident_workload import _resource_metrics

    document["resource_metrics"] = _resource_metrics(
        samples,
        max_gpus,
        target_job_id=(
            str(document.get("ray_job_id")) if document.get("ray_job_id") else None
        ),
    )
    document["timeout_metrics_finalized_by"] = "local-evidence-runner"
    return document


def execute_case(
    *,
    index: int,
    case: LocalCase,
    run_dir: Path,
    overlay: Path,
    artifact: Mapping[str, object],
    gpu_indices: Sequence[int],
    args: argparse.Namespace,
    staged_harness: Mapping[str, object],
) -> dict[str, object]:
    case_dir = run_dir / "cases" / f"{index:04d}-{case.case_id}"
    case_dir.mkdir(parents=True, exist_ok=False)
    spill_dir = case_dir / "ray-spill"
    validate_object_spilling_directory(
        spill_dir,
        case_directory=case_dir,
        results_root=RESULTS_ROOT,
        must_exist=False,
    )
    spill_storage_before_start = snapshot_object_spilling_directory(
        spill_dir,
        case_directory=case_dir,
        results_root=RESULTS_ROOT,
    )
    if not _valid_spill_storage_snapshot(
        spill_storage_before_start,
        spill_dir=spill_dir,
        require_exists=False,
    ):
        raise RuntimeError(
            f"invalid object-spilling storage preflight: {spill_storage_before_start}"
        )
    tmp_token = hashlib.sha256(case.case_id.encode()).hexdigest()[:9]
    tmp_dir = SHM_ROOT / f"l{tmp_token}"
    if tmp_dir.exists():
        raise RuntimeError(f"case Ray temp directory already exists: {tmp_dir}")
    ready_path = case_dir / "cluster-ready.json"
    stopped_path = case_dir / "cluster-stopped.json"
    stop_request = case_dir / "stop.request"
    daemon_log_path = case_dir / "cluster-daemon.log"
    harness_root = Path(str(staged_harness["directory"]))
    verified_harness = _verify_local_harness(
        harness_root, str(staged_harness["content_sha256"])
    )
    environment = _overlay_environment(overlay, gpu_indices, harness_root)
    environment["RAY_TMPDIR"] = str(tmp_dir)
    cupy_cache_dir = case_dir / "cupy-kernel-cache"
    environment["CUPY_CACHE_DIR"] = str(cupy_cache_dir)
    environment["RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL"] = (
        "1" if ARMS[case.arm].admission_enabled else "0"
    )
    object_store_memory = (
        args.spill_object_store_bytes
        if case.workload == "forced-spill"
        else (
            args.object_store_bytes * case.capacity
            if case.scaling_mode in {"strong", "weak", "actor-control"}
            else args.object_store_bytes
        )
    )
    daemon_command = [
        sys.executable,
        "-m",
        "benchmark.local.cluster_daemon",
        "--capacity",
        str(case.capacity),
        "--num-cpus",
        str(args.num_cpus),
        "--object-store-memory",
        str(object_store_memory),
        "--tmp-dir",
        str(tmp_dir),
        "--object-spilling-directory",
        str(spill_dir),
        "--case-directory",
        str(case_dir),
        "--results-root",
        str(RESULTS_ROOT),
        "--ready",
        str(ready_path),
        "--stop-request",
        str(stop_request),
        "--stopped",
        str(stopped_path),
    ]
    daemon_log = daemon_log_path.open("w")
    daemon = subprocess.Popen(
        daemon_command,
        cwd=harness_root,
        env=environment,
        stdout=daemon_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    map_actors = _map_actors_for_case(case, args.map_actors_per_stage)
    map_actors_max = args.map_actors_max_per_stage or map_actors
    case_rows = case.rows if case.rows is not None else args.rows
    case_blocks = case.blocks if case.blocks is not None else args.blocks
    record: dict[str, object] = {
        "case": {
            **asdict(case),
            "map_actors_per_stage": map_actors,
            "map_actors_max_per_stage": map_actors_max,
            "rows": case_rows,
            "blocks": case_blocks,
        },
        "physical_gpu_indices": list(gpu_indices),
        "artifact": dict(artifact),
        "overlay": str(overlay),
        "ray_tmpdir": str(tmp_dir),
        "cupy_kernel_cache_dir": str(cupy_cache_dir),
        "ray_spill_directory": str(spill_dir),
        "ray_spill_storage_before_start": spill_storage_before_start,
        "started_at": _utc_now(),
        "object_store_memory_bytes": object_store_memory,
        "staged_harness": verified_harness,
    }
    ready: dict[str, object] = {}
    try:
        ready = _wait_for_json(ready_path, daemon, args.cluster_start_timeout_seconds)
        record["cluster_ready"] = ready
        if ready.get("status") != "ready":
            raise RuntimeError(f"cluster daemon preflight failed: {ready}")
        ready_spill_storage = ready.get("object_spilling_storage")
        if (
            ready.get("object_spilling_directory") != str(spill_dir)
            or not _valid_spill_storage_snapshot(
                ready_spill_storage,
                spill_dir=spill_dir,
                require_exists=True,
            )
            or not _same_spill_filesystem(
                spill_storage_before_start, ready_spill_storage
            )
        ):
            raise RuntimeError(
                "cluster daemon did not prove its exact object-spilling directory: "
                f"{ready_spill_storage}"
            )
        environment["RAY_ADDRESS"] = str(ready["address"])

        preflight_path = case_dir / "runtime-gpu-preflight.json"
        preflight_command = [
            sys.executable,
            "-m",
            "benchmark.local.probe",
            "--mode",
            "runtime-gpu",
            "--arm",
            case.arm,
            "--capacity",
            str(case.capacity),
            "--wheel",
            str(ROOT / str(artifact["wheel"])),
            "--wheel-sha256",
            str(artifact["wheel_sha256"]),
            "--source-commit",
            str(artifact["source_commit"]),
            "--base-commit",
            str(artifact["base_commit"]),
            "--installed-ray-commit",
            str(artifact["installed_ray_commit"]),
            "--overlay",
            str(overlay),
            "--timeout-seconds",
            str(args.preflight_timeout_seconds),
            "--result",
            str(preflight_path),
        ]
        preflight_run = _run_bounded(
            preflight_command,
            environment=environment,
            log_path=case_dir / "runtime-gpu-preflight.log",
            timeout_seconds=args.preflight_timeout_seconds + 15,
            cwd=harness_root,
        )
        preflight = _load_json_if_present(preflight_path)
        record.update(preflight_command=preflight_run, preflight=preflight)
        if preflight_run["return_code"] != 0 or preflight.get("status") != "success":
            raise RuntimeError(f"runtime/GPU preflight failed: {preflight}")

        workload_path = case_dir / "workload.json"
        workload_command = [
            sys.executable,
            str(harness_root / "benchmark" / "cloud" / "incident_workload.py"),
            "--arm",
            case.arm,
            "--workload",
            case.workload,
            "--topology",
            f"local-physical-{case.capacity}",
            "--shuffle-ranks",
            str(case.shuffle_ranks),
            "--map-actors-min",
            str(map_actors),
            "--map-actors-max",
            str(map_actors_max),
            "--max-gpus",
            str(case.capacity),
            "--rows",
            str(case_rows),
            "--blocks",
            str(case_blocks),
            "--groups",
            str(args.groups),
            "--batch-size",
            str(args.batch_size),
            "--gpu-map-work-iterations",
            str(args.gpu_map_work_iterations),
            "--sample-interval-seconds",
            str(args.sample_interval_seconds),
            "--result",
            str(workload_path),
        ]
        if args.materialize_boundaries:
            workload_command.append("--materialize-boundaries")
        workload_run = _run_bounded(
            workload_command,
            environment=environment,
            log_path=case_dir / "workload.log",
            timeout_seconds=args.case_timeout_seconds,
            terminate_grace_seconds=5,
            cwd=harness_root,
        )
        workload = _load_json_if_present(workload_path)
        if workload_run["timed_out"]:
            workload = _finalize_timeout_metrics(workload, max_gpus=case.capacity)
            _atomic_json(workload_path, workload)
        record.update(workload_command=workload_run, workload=workload)

        job_id = workload.get("ray_job_id")
        if isinstance(job_id, str) and job_id:
            cleanup_path = case_dir / "job-cleanup.json"
            cleanup_command = [
                sys.executable,
                "-m",
                "benchmark.local.probe",
                "--mode",
                "cleanup",
                "--job-id",
                job_id,
                "--timeout-seconds",
                str(args.job_cleanup_timeout_seconds),
                "--result",
                str(cleanup_path),
            ]
            cleanup_run = _run_bounded(
                cleanup_command,
                environment=environment,
                log_path=case_dir / "job-cleanup.log",
                timeout_seconds=args.job_cleanup_timeout_seconds + 15,
                cwd=harness_root,
            )
            job_cleanup = _load_json_if_present(cleanup_path)
            record.update(job_cleanup_command=cleanup_run, job_cleanup=job_cleanup)
        else:
            record["job_cleanup"] = {
                "status": "error",
                "cleanup_proven": False,
                "reason": "workload did not record a Ray job ID",
            }
        record["outcome"] = _outcome(workload, workload_run)
    except BaseException as error:
        record["runner_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        record.setdefault("outcome", "harness_error")
    finally:
        try:
            record["cluster_cleanup"] = _stop_cluster(
                daemon=daemon,
                daemon_log=daemon_log,
                stop_request=stop_request,
                stopped_path=stopped_path,
                ready=ready,
                tmp_dir=tmp_dir,
                spill_dir=spill_dir,
                case_dir=case_dir,
            )
        except BaseException as error:
            if not daemon_log.closed:
                daemon_log.close()
            record["cluster_cleanup"] = {
                "cleanup_proven": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        record["finished_at"] = _utc_now()
        _atomic_json(case_dir / "execution.json", record)
    return record


def _duration_statistics(executions: Sequence[Mapping[str, object]]) -> list[dict]:
    grouped: dict[tuple[object, ...], list[float]] = {}
    for execution in executions:
        workload = execution.get("workload")
        case = execution.get("case")
        if not isinstance(workload, dict) or not isinstance(case, dict):
            continue
        elapsed = workload.get("elapsed_s")
        if execution.get("outcome") != "success" or not isinstance(
            elapsed, (int, float)
        ):
            continue
        key = (
            case.get("arm"),
            case.get("capacity"),
            case.get("workload"),
            case.get("shuffle_ranks"),
            case.get("map_actors_per_stage", case.get("capacity")),
        )
        grouped.setdefault(key, []).append(float(elapsed))
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        result.append(
            {
                "arm": key[0],
                "capacity": key[1],
                "workload": key[2],
                "shuffle_ranks": key[3],
                "map_actors_per_stage": key[4],
                "samples": len(values),
                "mean_s": statistics.fmean(values),
                "median_s": statistics.median(values),
                "min_s": min(values),
                "max_s": max(values),
            }
        )
    return result


def _correctness_groups(executions: Sequence[Mapping[str, object]]) -> list[dict]:
    grouped: dict[tuple[object, ...], set[tuple[object, object, object]]] = {}
    for execution in executions:
        if execution.get("outcome") != "success":
            continue
        case = execution.get("case")
        workload = execution.get("workload")
        if not isinstance(case, dict) or not isinstance(workload, dict):
            continue
        key = (
            case.get("capacity"),
            case.get("workload"),
            case.get("shuffle_ranks"),
            case.get("repetition"),
            case.get("map_actors_per_stage", case.get("capacity")),
        )
        grouped.setdefault(key, set()).add(
            (
                workload.get("output_rows"),
                workload.get("output_schema_hash"),
                workload.get("output_digest"),
            )
        )
    return [
        {
            "capacity": key[0],
            "workload": key[1],
            "shuffle_ranks": key[2],
            "repetition": key[3],
            "map_actors_per_stage": key[4],
            "oracles": [list(item) for item in sorted(values, key=str)],
            "agreement": len(values) == 1,
        }
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _execution_cleanup_proven(execution: Mapping[str, object]) -> bool:
    job_cleanup = execution.get("job_cleanup")
    cluster_cleanup = execution.get("cluster_cleanup")
    return (
        isinstance(job_cleanup, Mapping)
        and job_cleanup.get("cleanup_proven") is True
        and isinstance(cluster_cleanup, Mapping)
        and cluster_cleanup.get("cleanup_proven") is True
        and cluster_cleanup.get("ray_spill_directory_removed") is True
    )


def _comparison_runs(
    executions: Sequence[Mapping[str, object]],
    *,
    comparison_id: str,
    arm: str,
) -> list[dict[str, object]]:
    runs = []
    for execution in executions:
        if execution.get("outcome") != "success":
            continue
        case = execution.get("case")
        workload = execution.get("workload")
        if not isinstance(case, Mapping) or not isinstance(workload, Mapping):
            continue
        elapsed = workload.get("elapsed_s")
        if (
            case.get("comparison_id") != comparison_id
            or case.get("arm") != arm
            or not isinstance(elapsed, (int, float))
        ):
            continue
        runs.append(
            {
                "repetition": case.get("repetition"),
                "elapsed_s": float(elapsed),
                "case_id": case.get("case_id"),
            }
        )
    return runs


def _scale_correctness_groups(
    executions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[object, object],
        dict[str, set[object]],
    ] = {}
    for execution in executions:
        if execution.get("outcome") != "success":
            continue
        case = execution.get("case")
        workload = execution.get("workload")
        if not isinstance(case, Mapping) or not isinstance(workload, Mapping):
            continue
        comparison_id = case.get("comparison_id")
        if not comparison_id:
            continue
        key = (comparison_id, case.get("repetition"))
        group = grouped.setdefault(key, {"arms": set(), "oracles": set()})
        group["arms"].add(case.get("arm"))
        group["oracles"].add(
            (
                workload.get("output_rows"),
                workload.get("output_schema_hash"),
                workload.get("output_digest"),
            )
        )
    result = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        arms = sorted(str(item) for item in group["arms"])
        oracles = sorted(group["oracles"], key=str)
        complete_oracles = all(
            all(value is not None for value in oracle) for oracle in oracles
        )
        result.append(
            {
                "comparison_id": key[0],
                "repetition": key[1],
                "arms": arms,
                "oracles": [list(item) for item in oracles],
                "complete_oracles": complete_oracles,
                "agreement": (
                    arms == ["minimal", "stock"]
                    and len(oracles) == 1
                    and complete_oracles
                ),
            }
        )
    return result


def build_scale_evidence_report(
    *,
    executions: Sequence[Mapping[str, object]],
    repetitions: int,
    seed: int,
    executed: bool,
) -> dict[str, object]:
    """Evaluate the paired scale speedup and equal-shape regression gates."""

    candidate_incident = _comparison_runs(
        executions, comparison_id="incident-scale-speedup", arm="minimal"
    )
    stock_incident = _comparison_runs(
        executions, comparison_id="incident-scale-speedup", arm="stock"
    )
    candidate_actor = _comparison_runs(
        executions, comparison_id="actor-only-regression", arm="minimal"
    )
    stock_actor = _comparison_runs(
        executions, comparison_id="actor-only-regression", arm="stock"
    )
    speedup = evidence_statistics.speedup_vs_best_completing_stock(
        candidate_incident,
        {"map-actors-1_shuffle-rank-1": stock_incident},
        seed=seed,
        minimum_speedup=0.10,
        minimum_runs=5,
    )
    actor_comparison = evidence_statistics.compare_randomized_runs(
        candidate_actor,
        stock_actor,
        seed=seed + 100,
    )
    actor_regression = evidence_statistics.regression_gate(
        actor_comparison,
        max_regression=0.05,
        minimum_runs=5,
    )
    correctness = _scale_correctness_groups(executions)
    expected_runs = repetitions * 4
    all_cases_completed = len(executions) == expected_runs and all(
        item.get("outcome") == "success" for item in executions
    )
    cleanup_proven = len(executions) == expected_runs and all(
        _execution_cleanup_proven(item) for item in executions
    )
    expected_correctness_groups = repetitions * 2
    correctness_agreement = len(correctness) == expected_correctness_groups and all(
        item["agreement"] for item in correctness
    )
    gates = {
        "executed": executed,
        "all_planned_cases_completed": all_cases_completed,
        "all_job_and_cluster_cleanup_proven": cleanup_proven,
        "cross_configuration_correctness_agreement": correctness_agreement,
        "incident_at_least_10_percent_faster": speedup.get("passed") is True,
        "equal_shape_actor_only_regression_at_most_5_percent": (
            actor_regression.get("passed") is True
        ),
    }
    return {
        "schema_version": 1,
        "profile": "scale",
        "status": "pass"
        if all(gates.values())
        else ("fail" if executed else "not_run"),
        "gates": gates,
        "blockers": [name for name, passed in gates.items() if not passed],
        "planned_repetitions": repetitions,
        "expected_cases": expected_runs,
        "completion_counts": {
            "candidate_incident": len(candidate_incident),
            "stock_incident": len(stock_incident),
            "candidate_actor_only": len(candidate_actor),
            "stock_actor_only": len(stock_actor),
        },
        "incident_speedup": speedup,
        "actor_only_regression": actor_regression,
        "correctness_groups": correctness,
        "comparison_contract": {
            "incident_candidate": {
                "arm": "minimal",
                "capacity": 4,
                "map_actors_per_stage": 4,
                "shuffle_ranks": 4,
            },
            "incident_stock_workaround": {
                "arm": "stock",
                "capacity": 4,
                "map_actors_per_stage": 1,
                "shuffle_ranks": 1,
            },
            "actor_only_equal_shape": {
                "arms": ["stock", "minimal"],
                "capacity": 4,
                "map_actors_per_stage": 1,
                "shuffle_ranks": "default",
                "safety_constraint": "3 * map_actors_per_stage <= capacity",
            },
        },
    }


DGX_METRICS = (
    "elapsed_s",
    "input_rows_per_second",
    "cluster_gpu_seconds",
    "owned_gpu_seconds",
    "global_bytes_spilled",
    "global_bytes_restored",
    "premature_downstream_actor_count",
    "premature_downstream_ownership_seconds",
    "premature_downstream_gpu_seconds",
    "mean_ready_to_stage_first_input_s",
    "peak_premature_downstream_gpus",
    "premature_gpu_seconds_fraction_of_cluster",
    "premature_downstream_gpu_seconds_during_earlier_stage",
    "peak_premature_downstream_gpus_during_earlier_stage",
)


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _dgx_observations(
    executions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    observations = []
    for execution in executions:
        if execution.get("outcome") != "success":
            continue
        case = execution.get("case")
        workload = execution.get("workload")
        if not isinstance(case, Mapping) or not isinstance(workload, Mapping):
            continue
        if case.get("scaling_mode") not in {"strong", "weak", "actor-control"}:
            continue
        elapsed = _number(workload.get("elapsed_s"))
        rows = _number(case.get("rows"))
        if elapsed is None or elapsed <= 0 or rows is None or rows <= 0:
            continue
        throughput = _number(workload.get("input_rows_per_second"))
        if throughput is None:
            throughput = rows / elapsed
        resource_metrics = workload.get("resource_metrics")
        resources = resource_metrics if isinstance(resource_metrics, Mapping) else {}
        premature = resources.get("premature_downstream_ownership")
        premature_metrics = premature if isinstance(premature, Mapping) else {}
        metrics = {
            "elapsed_s": elapsed,
            "input_rows_per_second": throughput,
            "cluster_gpu_seconds": _number(resources.get("cluster_gpu_seconds")),
            "owned_gpu_seconds": _number(resources.get("owned_gpu_seconds")),
            "global_bytes_spilled": _number(workload.get("global_bytes_spilled")),
            "global_bytes_restored": _number(workload.get("global_bytes_restored")),
            "premature_downstream_actor_count": _number(
                premature_metrics.get("premature_downstream_actor_count")
            ),
            "premature_downstream_ownership_seconds": _number(
                premature_metrics.get("premature_downstream_ownership_seconds")
            ),
            "premature_downstream_gpu_seconds": _number(
                premature_metrics.get("premature_downstream_gpu_seconds")
            ),
            "mean_ready_to_stage_first_input_s": _number(
                premature_metrics.get("mean_ready_to_stage_first_input_s")
            ),
            "peak_premature_downstream_gpus": _number(
                premature_metrics.get("peak_premature_downstream_gpus")
            ),
            "premature_gpu_seconds_fraction_of_cluster": _number(
                premature_metrics.get("premature_gpu_seconds_fraction_of_cluster")
            ),
            "premature_downstream_gpu_seconds_during_earlier_stage": _number(
                premature_metrics.get(
                    "premature_downstream_gpu_seconds_during_earlier_stage"
                )
            ),
            "peak_premature_downstream_gpus_during_earlier_stage": _number(
                premature_metrics.get(
                    "peak_premature_downstream_gpus_during_earlier_stage"
                )
            ),
        }
        observations.append(
            {
                "case_id": case.get("case_id"),
                "comparison_id": case.get("comparison_id"),
                "scaling_mode": case.get("scaling_mode"),
                "arm": case.get("arm"),
                "capacity": case.get("capacity"),
                "workaround_id": case.get("workaround_id"),
                "repetition": case.get("repetition"),
                "rows": int(rows),
                "blocks": case.get("blocks"),
                "object_store_memory_bytes": execution.get("object_store_memory_bytes"),
                "metrics": metrics,
                "premature_per_stage": premature_metrics.get("per_stage", []),
                "causal_telemetry_complete": (
                    premature_metrics.get("measurement_complete") is True
                    and premature_metrics.get("gcs_corroboration_complete") is True
                    and premature_metrics.get("gcs_state_samples_valid") is True
                    and premature_metrics.get("common_clock_verified") is True
                    and premature_metrics.get("clock_basis") == "single-boot-monotonic"
                    and _number(premature_metrics.get("gcs_corroborated_actor_count"))
                    == _number(premature_metrics.get("gcs_expected_actor_count"))
                    == _number(
                        premature_metrics.get("premature_downstream_actor_count")
                    )
                    and isinstance(
                        premature_metrics.get("telemetry_quiescence"), Mapping
                    )
                    and premature_metrics["telemetry_quiescence"].get("complete")
                    is True
                    if case.get("scaling_mode") in {"strong", "weak"}
                    else True
                ),
                "telemetry_complete": all(
                    metrics[name] is not None for name in DGX_METRICS
                ),
            }
        )
    return observations


def _dgx_runs(
    observations: Sequence[Mapping[str, object]],
    *,
    scaling_mode: str,
    arm: str,
    capacity: int,
    workaround_id: str | None = None,
) -> list[dict[str, object]]:
    result = []
    for item in observations:
        metrics = item.get("metrics")
        if (
            item.get("scaling_mode") != scaling_mode
            or item.get("arm") != arm
            or item.get("capacity") != capacity
            or (
                workaround_id is not None and item.get("workaround_id") != workaround_id
            )
            or not isinstance(metrics, Mapping)
        ):
            continue
        elapsed = _number(metrics.get("elapsed_s"))
        if elapsed is None:
            continue
        result.append(
            {
                "repetition": item.get("repetition"),
                "elapsed_s": elapsed,
                "case_id": item.get("case_id"),
                "workaround_id": item.get("workaround_id"),
            }
        )
    return result


def _dgx_configuration_statistics(
    observations: Sequence[Mapping[str, object]], *, seed: int
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[object, object, object, object], list[Mapping[str, object]]
    ] = {}
    for item in observations:
        key = (
            item.get("scaling_mode"),
            item.get("arm"),
            item.get("capacity"),
            item.get("workaround_id"),
        )
        grouped.setdefault(key, []).append(item)
    rows = []
    for index, (key, items) in enumerate(
        sorted(grouped.items(), key=lambda item: str(item[0]))
    ):
        metric_summaries = {}
        for offset, name in enumerate(DGX_METRICS):
            values = []
            for item in items:
                metrics = item.get("metrics")
                value = metrics.get(name) if isinstance(metrics, Mapping) else None
                number = _number(value)
                if number is not None:
                    values.append(number)
            metric_summaries[name] = evidence_statistics.summarize_samples(
                values,
                seed=seed + index * len(DGX_METRICS) + offset,
                resamples=2_000,
            )
        rows.append(
            {
                "scaling_mode": key[0],
                "arm": key[1],
                "capacity": key[2],
                "workaround_id": key[3],
                "samples": len(items),
                "rows": sorted({item.get("rows") for item in items}),
                "blocks": sorted({item.get("blocks") for item in items}),
                "metrics": metric_summaries,
            }
        )
    return rows


def _median_metric(
    observations: Sequence[Mapping[str, object]],
    *,
    scaling_mode: str,
    arm: str,
    capacity: int,
    metric: str,
    workaround_id: str | None = None,
) -> float | None:
    values = []
    for item in observations:
        metrics = item.get("metrics")
        if (
            item.get("scaling_mode") == scaling_mode
            and item.get("arm") == arm
            and item.get("capacity") == capacity
            and (workaround_id is None or item.get("workaround_id") == workaround_id)
            and isinstance(metrics, Mapping)
        ):
            value = _number(metrics.get(metric))
            if value is not None:
                values.append(value)
    return float(statistics.median(values)) if values else None


def _strong_scaling_curve(
    observations: Sequence[Mapping[str, object]],
    *,
    arm: str,
    workaround_id: str | None = None,
) -> dict[str, object]:
    capacities = [
        capacity
        for capacity in DGX_CAPACITIES
        if _median_metric(
            observations,
            scaling_mode="strong",
            arm=arm,
            capacity=capacity,
            metric="elapsed_s",
            workaround_id=workaround_id,
        )
        is not None
    ]
    if not capacities:
        return {
            "arm": arm,
            "workaround_id": workaround_id,
            "baseline_capacity": None,
            "points": [],
        }
    baseline_capacity = min(capacities)
    baseline_elapsed = _median_metric(
        observations,
        scaling_mode="strong",
        arm=arm,
        capacity=baseline_capacity,
        metric="elapsed_s",
        workaround_id=workaround_id,
    )
    assert baseline_elapsed is not None
    points = []
    for capacity in capacities:
        elapsed = _median_metric(
            observations,
            scaling_mode="strong",
            arm=arm,
            capacity=capacity,
            metric="elapsed_s",
            workaround_id=workaround_id,
        )
        throughput = _median_metric(
            observations,
            scaling_mode="strong",
            arm=arm,
            capacity=capacity,
            metric="input_rows_per_second",
            workaround_id=workaround_id,
        )
        assert elapsed is not None
        speedup = baseline_elapsed / elapsed
        points.append(
            {
                "capacity": capacity,
                "median_elapsed_s": elapsed,
                "median_input_rows_per_second": throughput,
                "speedup_vs_baseline": speedup,
                "parallel_efficiency": speedup / (capacity / baseline_capacity),
            }
        )
    return {
        "arm": arm,
        "workaround_id": workaround_id,
        "baseline_capacity": baseline_capacity,
        "points": points,
    }


def _weak_scaling_curve(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    points = []
    baseline_elapsed = _median_metric(
        observations,
        scaling_mode="weak",
        arm="minimal",
        capacity=1,
        metric="elapsed_s",
    )
    baseline_throughput = _median_metric(
        observations,
        scaling_mode="weak",
        arm="minimal",
        capacity=1,
        metric="input_rows_per_second",
    )
    one_gpu_items = [
        item
        for item in observations
        if item.get("scaling_mode") == "weak"
        and item.get("arm") == "minimal"
        and item.get("capacity") == 1
    ]
    baseline_rows = (
        statistics.median(float(item["rows"]) for item in one_gpu_items)
        if one_gpu_items
        else None
    )
    for capacity in DGX_CAPACITIES:
        matching = [
            item
            for item in observations
            if item.get("scaling_mode") == "weak"
            and item.get("arm") == "minimal"
            and item.get("capacity") == capacity
        ]
        elapsed = _median_metric(
            observations,
            scaling_mode="weak",
            arm="minimal",
            capacity=capacity,
            metric="elapsed_s",
        )
        throughput = _median_metric(
            observations,
            scaling_mode="weak",
            arm="minimal",
            capacity=capacity,
            metric="input_rows_per_second",
        )
        if (
            not matching
            or elapsed is None
            or throughput is None
            or baseline_elapsed is None
            or baseline_throughput is None
            or baseline_rows is None
        ):
            continue
        rows = statistics.median(float(item["rows"]) for item in matching)
        work_ratio = rows / baseline_rows
        throughput_scaleup = throughput / baseline_throughput
        points.append(
            {
                "capacity": capacity,
                "median_rows": rows,
                "median_rows_per_gpu": rows / capacity,
                "median_elapsed_s": elapsed,
                "median_input_rows_per_second": throughput,
                "work_ratio_vs_one_gpu": work_ratio,
                "throughput_scaleup_vs_one_gpu": throughput_scaleup,
                "parallel_efficiency": throughput_scaleup / work_ratio,
                "elapsed_efficiency": baseline_elapsed / elapsed,
            }
        )
    return {"arm": "minimal", "baseline_capacity": 1, "points": points}


DGX_PREMATURE_METRICS = (
    "premature_downstream_actor_count",
    "premature_downstream_ownership_seconds",
    "premature_downstream_gpu_seconds",
    "mean_ready_to_stage_first_input_s",
    "peak_premature_downstream_gpus",
    "premature_gpu_seconds_fraction_of_cluster",
    "premature_downstream_gpu_seconds_during_earlier_stage",
    "peak_premature_downstream_gpus_during_earlier_stage",
)


def _dgx_premature_ownership_comparisons(
    observations: Sequence[Mapping[str, object]],
    *,
    selected_stock_workarounds: Mapping[int, str],
) -> dict[str, dict[str, object]]:
    comparisons = {}
    for capacity in DGX_STOCK_COMPLETING_CAPACITIES:
        by_arm: dict[str, dict[object, Mapping[str, object]]] = {
            "stock": {},
            "minimal": {},
        }
        for item in observations:
            if (
                item.get("scaling_mode") == "strong"
                and item.get("capacity") == capacity
                and item.get("arm") in by_arm
                and (
                    item.get("arm") == "minimal"
                    or item.get("workaround_id")
                    == selected_stock_workarounds.get(capacity)
                )
            ):
                by_arm[str(item["arm"])][item.get("repetition")] = item
        repetitions = sorted(set(by_arm["stock"]) & set(by_arm["minimal"]), key=str)
        metrics = {}
        for name in DGX_PREMATURE_METRICS:
            pairs = []
            for repetition in repetitions:
                stock_metrics = by_arm["stock"][repetition].get("metrics")
                candidate_metrics = by_arm["minimal"][repetition].get("metrics")
                stock = (
                    _number(stock_metrics.get(name))
                    if isinstance(stock_metrics, Mapping)
                    else None
                )
                candidate = (
                    _number(candidate_metrics.get(name))
                    if isinstance(candidate_metrics, Mapping)
                    else None
                )
                if stock is None or candidate is None:
                    continue
                pairs.append(
                    {
                        "repetition": repetition,
                        "stock": stock,
                        "candidate": candidate,
                        "stock_minus_candidate": stock - candidate,
                    }
                )
            stock_values = [float(item["stock"]) for item in pairs]
            candidate_values = [float(item["candidate"]) for item in pairs]
            stock_median = (
                float(statistics.median(stock_values)) if stock_values else None
            )
            candidate_median = (
                float(statistics.median(candidate_values)) if candidate_values else None
            )
            metrics[name] = {
                "paired_runs": len(pairs),
                "stock_median": stock_median,
                "candidate_median": candidate_median,
                "median_avoided": (
                    stock_median - candidate_median
                    if stock_median is not None and candidate_median is not None
                    else None
                ),
                "median_fraction_avoided": (
                    1.0 - candidate_median / stock_median
                    if stock_median is not None
                    and candidate_median is not None
                    and stock_median > 0
                    else None
                ),
                "per_repetition": pairs,
            }
        gpu_seconds = metrics["premature_downstream_gpu_seconds"]
        earlier_stage_overlap = metrics[
            "premature_downstream_gpu_seconds_during_earlier_stage"
        ]
        per_stage = {}
        for stage in ("gpu-map-groups-1", "gpu-map-final"):
            pairs = []
            for repetition in repetitions:
                values = {}
                for arm in ("stock", "minimal"):
                    item = by_arm[arm][repetition]
                    entries = item.get("premature_per_stage")
                    entry = (
                        next(
                            (
                                value
                                for value in entries
                                if isinstance(value, Mapping)
                                and value.get("stage") == stage
                            ),
                            {},
                        )
                        if isinstance(entries, list)
                        else {}
                    )
                    values[arm] = {
                        "gpu_seconds": _number(entry.get("gpu_seconds")) or 0.0,
                        "cluster_gpu_seconds_fraction": _number(
                            entry.get("cluster_gpu_seconds_fraction")
                        )
                        or 0.0,
                        "actor_count": _number(entry.get("actor_count")) or 0.0,
                    }
                pairs.append(
                    {
                        "repetition": repetition,
                        "stock": values["stock"],
                        "candidate": values["minimal"],
                        "avoided_gpu_seconds": values["stock"]["gpu_seconds"]
                        - values["minimal"]["gpu_seconds"],
                    }
                )
            per_stage[stage] = {
                "operator": (
                    "SumGroup/map_groups"
                    if stage == "gpu-map-groups-1"
                    else "Identity/map_batches"
                ),
                "paired_runs": len(pairs),
                "stock_median_gpu_seconds": (
                    statistics.median(item["stock"]["gpu_seconds"] for item in pairs)
                    if pairs
                    else None
                ),
                "candidate_median_gpu_seconds": (
                    statistics.median(
                        item["candidate"]["gpu_seconds"] for item in pairs
                    )
                    if pairs
                    else None
                ),
                "median_avoided_gpu_seconds": (
                    statistics.median(item["avoided_gpu_seconds"] for item in pairs)
                    if pairs
                    else None
                ),
                "per_repetition": pairs,
            }
        comparisons[str(capacity)] = {
            "capacity": capacity,
            "stock_workaround_id": selected_stock_workarounds.get(capacity),
            "metrics": metrics,
            "per_stage": per_stage,
            "candidate_reduced_premature_gpu_seconds": (
                gpu_seconds["paired_runs"] >= 5
                and gpu_seconds["median_avoided"] is not None
                and gpu_seconds["median_avoided"] > 0
            ),
            "candidate_reduced_premature_gpu_overlap_during_earlier_stage": (
                earlier_stage_overlap["paired_runs"] >= 5
                and earlier_stage_overlap["median_avoided"] is not None
                and earlier_stage_overlap["median_avoided"] > 0
            ),
        }
    return comparisons


def _dgx_correctness(
    executions: Sequence[Mapping[str, object]], *, repetitions: int, groups: int
) -> dict[str, object]:
    checks = []

    def successful(mode: str, repetition: int, capacity: int | None = None):
        selected = []
        for execution in executions:
            if execution.get("outcome") != "success":
                continue
            case = execution.get("case")
            workload = execution.get("workload")
            if not isinstance(case, Mapping) or not isinstance(workload, Mapping):
                continue
            if case.get("scaling_mode") != mode or case.get("repetition") != repetition:
                continue
            if capacity is not None and case.get("capacity") != capacity:
                continue
            selected.append((case, workload))
        return selected

    for repetition in range(1, repetitions + 1):
        strong = successful("strong", repetition)
        strong_oracles = {
            (
                workload.get("output_rows"),
                workload.get("output_schema_hash"),
                workload.get("output_digest"),
            )
            for _, workload in strong
        }
        strong_row_counts_match = all(
            workload.get("output_rows") == min(int(case["rows"]), groups)
            for case, workload in strong
        )
        checks.append(
            {
                "group": "strong-fixed-dataset",
                "repetition": repetition,
                "observed_cases": len(strong),
                "expected_cases": len(DGX_CAPACITIES) + DGX_STOCK_WORKAROUND_COUNT,
                "distinct_oracles": len(strong_oracles),
                "row_counts_match": strong_row_counts_match,
                "passed": len(strong)
                == len(DGX_CAPACITIES) + DGX_STOCK_WORKAROUND_COUNT
                and len(strong_oracles) == 1
                and strong_row_counts_match
                and all(value is not None for value in next(iter(strong_oracles), ())),
            }
        )
        for capacity in DGX_CAPACITIES:
            weak = successful("weak", repetition, capacity)
            weak_complete = len(weak) == 1
            if weak_complete:
                case, workload = weak[0]
                oracle_complete = all(
                    workload.get(name) is not None
                    for name in ("output_rows", "output_schema_hash", "output_digest")
                )
                row_count_matches = workload.get("output_rows") == min(
                    int(case["rows"]), groups
                )
            else:
                oracle_complete = False
                row_count_matches = False
            checks.append(
                {
                    "group": f"weak-g{capacity}",
                    "repetition": repetition,
                    "observed_cases": len(weak),
                    "expected_cases": 1,
                    "row_count_matches": row_count_matches,
                    "passed": weak_complete and oracle_complete and row_count_matches,
                }
            )
        actor = successful("actor-control", repetition)
        actor_oracles = {
            (
                workload.get("output_rows"),
                workload.get("output_schema_hash"),
                workload.get("output_digest"),
            )
            for _, workload in actor
        }
        actor_row_counts_match = all(
            workload.get("output_rows") == case.get("rows") for case, workload in actor
        )
        checks.append(
            {
                "group": f"actor-g{DGX_ACTOR_CONTROL_CAPACITY}",
                "repetition": repetition,
                "observed_cases": len(actor),
                "expected_cases": 2,
                "distinct_oracles": len(actor_oracles),
                "row_counts_match": actor_row_counts_match,
                "passed": len(actor) == 2
                and len(actor_oracles) == 1
                and actor_row_counts_match
                and all(value is not None for value in next(iter(actor_oracles), ())),
            }
        )
    return {
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "groups": checks,
    }


def build_dgx_scale_report(
    *,
    executions: Sequence[Mapping[str, object]],
    repetitions: int,
    seed: int,
    executed: bool,
    artifacts_ready: bool,
    sixteen_physical_gpus_visible: bool,
    strong_rows: int = SCALE_DEFAULT_ROWS,
    strong_blocks: int = SCALE_DEFAULT_BLOCKS,
    weak_rows_per_gpu: int = DGX_WEAK_ROWS_PER_GPU,
    weak_blocks_per_gpu: int = DGX_WEAK_BLOCKS_PER_GPU,
    weak_max_rows: int | None = None,
    groups: int = 64,
    num_cpus: int = DGX_MIN_CPUS,
    object_store_bytes_per_gpu: int = DGX_OBJECT_STORE_BYTES_PER_GPU,
) -> dict[str, object]:
    """Build the paired single-host 1/2/4/8/16-GPU evidence report."""

    observations = _dgx_observations(executions)
    comparisons = {}
    selected_stock_workarounds: dict[int, str] = {}
    for capacity in DGX_STOCK_COMPLETING_CAPACITIES:
        candidate = _dgx_runs(
            observations,
            scaling_mode="strong",
            arm="minimal",
            capacity=capacity,
        )
        stock_workarounds = {
            f"m{map_actors}-r{shuffle_ranks}": _dgx_runs(
                observations,
                scaling_mode="strong",
                arm="stock",
                capacity=capacity,
                workaround_id=f"m{map_actors}-r{shuffle_ranks}",
            )
            for map_actors, shuffle_ranks in DGX_STOCK_WORKAROUNDS[capacity]
        }
        comparison = evidence_statistics.speedup_vs_best_completing_stock(
            candidate,
            stock_workarounds,
            seed=seed + capacity,
            minimum_speedup=0.10,
            minimum_runs=5,
        )
        medians = comparison.get("stock_medians")
        ranking = (
            sorted(
                (
                    {"workaround_id": name, "median_elapsed_s": elapsed}
                    for name, elapsed in medians.items()
                ),
                key=lambda item: (
                    float(item["median_elapsed_s"]),
                    str(item["workaround_id"]),
                ),
            )
            if isinstance(medians, Mapping)
            else []
        )
        comparison["stock_workaround_ranking"] = ranking
        comparison["top_two_stock_workarounds"] = ranking[:2]
        comparisons[str(capacity)] = comparison
        selected = comparison.get("best_stock_workaround")
        if isinstance(selected, str):
            selected_stock_workarounds[capacity] = selected

    candidate_actor = _dgx_runs(
        observations,
        scaling_mode="actor-control",
        arm="minimal",
        capacity=DGX_ACTOR_CONTROL_CAPACITY,
    )
    stock_actor = _dgx_runs(
        observations,
        scaling_mode="actor-control",
        arm="stock",
        capacity=DGX_ACTOR_CONTROL_CAPACITY,
    )
    actor_comparison = evidence_statistics.compare_randomized_runs(
        candidate_actor,
        stock_actor,
        seed=seed + 1_000,
    )
    actor_regression = evidence_statistics.regression_gate(
        actor_comparison,
        max_regression=0.05,
        minimum_runs=5,
    )
    premature_comparisons = _dgx_premature_ownership_comparisons(
        observations,
        selected_stock_workarounds=selected_stock_workarounds,
    )
    expected_cases = repetitions * DGX_CASES_PER_REPETITION
    correctness = _dgx_correctness(executions, repetitions=repetitions, groups=groups)
    completion = len(executions) == expected_cases and all(
        item.get("outcome") == "success" for item in executions
    )
    cleanup = len(executions) == expected_cases and all(
        _execution_cleanup_proven(item) for item in executions
    )
    telemetry = len(observations) == expected_cases and all(
        item["telemetry_complete"] for item in observations
    )
    causal_telemetry = len(observations) == expected_cases and all(
        item["causal_telemetry_complete"] for item in observations
    )
    object_store_contract = len(observations) == expected_cases and all(
        item.get("object_store_memory_bytes")
        == object_store_bytes_per_gpu * int(item["capacity"])
        for item in observations
    )
    strong_candidate_complete = all(
        len(
            _dgx_runs(
                observations,
                scaling_mode="strong",
                arm="minimal",
                capacity=capacity,
            )
        )
        == repetitions
        for capacity in DGX_CAPACITIES
    )
    weak_candidate_complete = all(
        len(
            _dgx_runs(
                observations,
                scaling_mode="weak",
                arm="minimal",
                capacity=capacity,
            )
        )
        == repetitions
        for capacity in DGX_CAPACITIES
    )
    gates = {
        "executed": executed,
        "artifact_separation_and_hashes": artifacts_ready,
        "at_least_16_physical_gpus_visible": sixteen_physical_gpus_visible,
        "at_least_five_randomized_repetitions": repetitions >= 5,
        "at_least_64_logical_cpus": num_cpus >= DGX_MIN_CPUS,
        "object_store_is_8gib_per_gpu": (
            object_store_bytes_per_gpu == DGX_OBJECT_STORE_BYTES_PER_GPU
        ),
        "every_case_used_8gib_object_store_per_gpu": object_store_contract,
        "strong_dataset_is_4b_rows_and_1024_blocks": (
            strong_rows == SCALE_DEFAULT_ROWS and strong_blocks == SCALE_DEFAULT_BLOCKS
        ),
        "weak_dataset_is_unbounded_1b_rows_and_256_blocks_per_gpu": (
            weak_rows_per_gpu == DGX_WEAK_ROWS_PER_GPU
            and weak_blocks_per_gpu == DGX_WEAK_BLOCKS_PER_GPU
            and weak_max_rows is None
        ),
        "all_planned_cases_completed": completion,
        "all_job_and_cluster_cleanup_proven": cleanup,
        "all_throughput_gpu_seconds_and_spill_metrics_present": telemetry,
        "all_downstream_actor_stage_and_first_input_measurements_present": (
            causal_telemetry
        ),
        "all_correctness_oracles_agree": correctness["passed"],
        "candidate_strong_curve_complete_1_2_4_8_16": strong_candidate_complete,
        "candidate_weak_curve_complete_1_2_4_8_16": weak_candidate_complete,
        "candidate_at_least_10_percent_faster_at_4_8_16": all(
            result.get("passed") is True for result in comparisons.values()
        ),
        "candidate_reduces_premature_reservation_and_earlier_stage_overlap_at_4_8_16": all(
            result["candidate_reduced_premature_gpu_seconds"]
            and result["candidate_reduced_premature_gpu_overlap_during_earlier_stage"]
            for result in premature_comparisons.values()
        ),
        "equal_shape_actor_only_regression_at_most_5_percent": (
            actor_regression.get("passed") is True
        ),
    }
    configuration_statistics = _dgx_configuration_statistics(
        observations, seed=seed + 2_000
    )
    return {
        "schema_version": 1,
        "profile": "dgx-scale",
        "status": "pass"
        if all(gates.values())
        else ("fail" if executed else "not_run"),
        "gates": gates,
        "blockers": [name for name, passed in gates.items() if not passed],
        "planned_repetitions": repetitions,
        "expected_cases": expected_cases,
        "observed_cases": len(executions),
        "dataset_contract": {
            "strong_rows": strong_rows,
            "strong_blocks": strong_blocks,
            "weak_rows_per_gpu": weak_rows_per_gpu,
            "weak_blocks_per_gpu": weak_blocks_per_gpu,
            "weak_max_rows": weak_max_rows,
            "evidence_shape": (
                strong_rows == SCALE_DEFAULT_ROWS
                and strong_blocks == SCALE_DEFAULT_BLOCKS
                and weak_rows_per_gpu == DGX_WEAK_ROWS_PER_GPU
                and weak_blocks_per_gpu == DGX_WEAK_BLOCKS_PER_GPU
                and weak_max_rows is None
            ),
        },
        "resource_contract": {
            "logical_cpus": num_cpus,
            "minimum_logical_cpus": DGX_MIN_CPUS,
            "object_store_bytes_per_gpu": object_store_bytes_per_gpu,
            "object_store_bytes_by_capacity": {
                str(capacity): object_store_bytes_per_gpu * capacity
                for capacity in DGX_CAPACITIES
            },
            "weak_keyed_bytes_per_object_store_byte": (
                16 * weak_rows_per_gpu / object_store_bytes_per_gpu
                if object_store_bytes_per_gpu > 0
                else None
            ),
        },
        "strong_scaling": {
            "fixed_rows": True,
            "candidate_curve": _strong_scaling_curve(observations, arm="minimal"),
            "stock_workaround_curves": {
                workaround_id: _strong_scaling_curve(
                    observations,
                    arm="stock",
                    workaround_id=workaround_id,
                )
                for workaround_id in sorted(
                    {
                        f"m{map_actors}-r{shuffle_ranks}"
                        for values in DGX_STOCK_WORKAROUNDS.values()
                        for map_actors, shuffle_ranks in values
                    }
                )
            },
            "candidate_vs_stock_by_capacity": comparisons,
        },
        "weak_scaling": {
            "target_rows_per_gpu": weak_rows_per_gpu,
            "maximum_total_rows": weak_max_rows,
            "bounded_override": weak_max_rows is not None,
            "candidate_curve": _weak_scaling_curve(observations),
        },
        "actor_only_regression": actor_regression,
        "premature_downstream_ownership": {
            "definition": (
                "Measured lower-bound GPU actor constructor-ready intervals for post-shuffle "
                "map_groups and final map_batches before the stage receives "
                "its first usable batch, with each counted actor corroborated "
                "in-window as a restart-free GCS ALIVE GPU owner; earlier-stage "
                "overlap does not claim a pending request was displaced"
            ),
            "candidate_vs_stock_by_capacity": premature_comparisons,
        },
        "configuration_statistics": configuration_statistics,
        "correctness": correctness,
        "metric_contract": list(DGX_METRICS),
        "comparison_contract": {
            "candidate": {
                "arm": "minimal",
                "capacities": list(DGX_CAPACITIES),
                "map_actors_per_stage": "capacity",
                "shuffle_ranks": "capacity",
            },
            "stock_measured_safe_frontier": {
                "arm": "stock",
                "capacities": list(DGX_STOCK_COMPLETING_CAPACITIES),
                "constraint": "3 * map_actors_per_stage + shuffle_ranks <= capacity",
                "configurations": {
                    str(capacity): [
                        {
                            "workaround_id": f"m{map_actors}-r{shuffle_ranks}",
                            "map_actors_per_stage": map_actors,
                            "shuffle_ranks": shuffle_ranks,
                        }
                        for map_actors, shuffle_ranks in values
                    ]
                    for capacity, values in DGX_STOCK_WORKAROUNDS.items()
                },
                "selected_fastest_completing": {
                    str(capacity): selected_stock_workarounds.get(capacity)
                    for capacity in DGX_STOCK_COMPLETING_CAPACITIES
                },
                "one_and_two_gpu_exclusion": (
                    "no positive uniform floor fits three map pools plus the "
                    "shuffle gang; the liveness profile owns those structural "
                    "cases"
                ),
            },
            "actor_only_equal_shape": {
                "arms": ["stock", "minimal"],
                "capacity": DGX_ACTOR_CONTROL_CAPACITY,
                "map_actors_per_stage": DGX_ACTOR_CONTROL_MAP_ACTORS,
                "shuffle_ranks": "default",
                "safety_constraint": "3 * map_actors_per_stage <= capacity",
            },
            "focused_confirmation_after_screening": {
                "stock_shapes": "up to top two ranked shapes per capacity",
                "minimum_randomized_repetitions": 5,
                "equal_tuned_candidate_control": (
                    "same map_actors_per_stage and shuffle_ranks as each "
                    "selected stock shape"
                ),
                "object_store_bytes_per_gpu": object_store_bytes_per_gpu,
                "note": (
                    "the one-repetition default is screening only; deliberately "
                    "repeating the full grid five times satisfies the report gate"
                ),
            },
        },
    }


def build_prerequisite_report(
    *,
    plan: Mapping[str, object],
    executions: Sequence[Mapping[str, object]],
    executed: bool,
) -> dict[str, object]:
    artifact_ready = bool(dict(plan["artifact_readiness"]).get("ready"))
    inventory = dict(plan["gpu_inventory"])
    max_capacity = max(plan["capacities"])
    gpu_ready = len(inventory.get("devices", [])) >= max_capacity
    cleanup_ready = bool(executions) and all(
        _execution_cleanup_proven(item) for item in executions
    )
    correctness = _correctness_groups(executions)
    correctness_ready = bool(correctness) and all(
        item["agreement"] for item in correctness
    )

    observed_case_keys = {
        (
            item["case"].get("arm"),
            item["case"].get("capacity"),
            item["case"].get("workload"),
            item["case"].get("shuffle_ranks"),
            item["case"].get("map_actors_per_stage", item["case"].get("capacity")),
        )
        for item in executions
        if isinstance(item.get("case"), dict)
    }
    required_case_keys = set()
    for arm in ARM_NAMES:
        for capacity in (1, 2, 4):
            required_case_keys.add((arm, capacity, "actor-only", "default", capacity))
            for rank in (*range(1, capacity + 1), "default"):
                required_case_keys.add((arm, capacity, "incident", rank, capacity))
    complete_core_matrix = required_case_keys <= observed_case_keys

    def matching(
        *, arm: str, capacity: int, workload: str, rank: int | str, outcome: str
    ) -> list[Mapping[str, object]]:
        return [
            item
            for item in executions
            if item.get("outcome") == outcome
            and isinstance(item.get("case"), dict)
            and item["case"].get("arm") == arm
            and item["case"].get("capacity") == capacity
            and item["case"].get("workload") == workload
            and item["case"].get("shuffle_ranks") == rank
            and item["case"].get("map_actors_per_stage", item["case"].get("capacity"))
            == capacity
        ]

    structural = []
    final_completes = []
    for capacity in plan["capacities"]:
        structural.append(
            bool(
                matching(
                    arm="stock",
                    capacity=capacity,
                    workload="incident",
                    rank=capacity,
                    outcome="structural_closed_wait_timeout",
                )
            )
        )
        final_completes.append(
            bool(
                matching(
                    arm="minimal",
                    capacity=capacity,
                    workload="incident",
                    rank=capacity,
                    outcome="success",
                )
            )
        )
    one_gpu_rank_floor = bool(
        matching(
            arm="stock",
            capacity=1,
            workload="incident",
            rank=1,
            outcome="structural_closed_wait_timeout",
        )
    )
    actor_only_generalization = all(
        bool(
            matching(
                arm="minimal",
                capacity=capacity,
                workload="actor-only",
                rank="default",
                outcome="success",
            )
        )
        for capacity in plan["capacities"]
    )
    gates = {
        "executed": executed,
        "artifact_separation_and_hashes": artifact_ready,
        "at_least_max_capacity_physical_gpus": gpu_ready,
        "every_job_and_cluster_cleanup_proven": cleanup_ready,
        "all_completing_arms_agree_on_content_and_schema": correctness_ready,
        "all_four_arms_have_1_2_4_gpu_incident_rank_sweeps_and_actor_chain": complete_core_matrix,
        "stock_full_rank_structural_wait_each_capacity": all(structural),
        "minimal_full_rank_completes_each_capacity": all(final_completes),
        "one_gpu_rank_one_proves_rank_cannot_be_lowered": one_gpu_rank_floor,
        "minimal_actor_only_chain_completes_each_capacity": actor_only_generalization,
    }
    report = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "status": "ready_for_cloud" if all(gates.values()) else "not_ready_for_cloud",
        "gates": gates,
        "blockers": [name for name, passed in gates.items() if not passed],
        "important_limit": (
            "Single-host tests validate physical GPU scheduling, liveness, correctness, "
            "and cleanup. They do not measure EC2 launch latency, IAM/network behavior, "
            "or real multi-host autoscaling; cloud evidence remains mandatory."
        ),
        "case_count": len(executions),
        "outcome_counts": {
            outcome: sum(1 for item in executions if item.get("outcome") == outcome)
            for outcome in sorted({str(item.get("outcome")) for item in executions})
        },
        "correctness_groups": correctness,
        "duration_statistics": _duration_statistics(executions),
    }
    if plan.get("profile") == "scale":
        report["scale_evidence"] = build_scale_evidence_report(
            executions=executions,
            repetitions=int(plan["repetitions"]),
            seed=int(plan["seed"]),
            executed=executed,
        )
    elif plan.get("profile") == "dgx-scale":
        dgx_contract = dict(plan.get("dgx_scale_contract", {}))
        report["dgx_scale_evidence"] = build_dgx_scale_report(
            executions=executions,
            repetitions=int(plan["repetitions"]),
            seed=int(plan["seed"]),
            executed=executed,
            artifacts_ready=artifact_ready,
            sixteen_physical_gpus_visible=gpu_ready,
            strong_rows=int(dgx_contract.get("strong_rows", SCALE_DEFAULT_ROWS)),
            strong_blocks=int(dgx_contract.get("strong_blocks", SCALE_DEFAULT_BLOCKS)),
            weak_rows_per_gpu=int(
                dgx_contract.get("weak_rows_per_gpu", DGX_WEAK_ROWS_PER_GPU)
            ),
            weak_blocks_per_gpu=int(
                dgx_contract.get("weak_blocks_per_gpu", DGX_WEAK_BLOCKS_PER_GPU)
            ),
            weak_max_rows=dgx_contract.get("weak_max_rows"),
            groups=int(dict(plan.get("parameters", {})).get("groups", 64)),
            num_cpus=int(dict(plan.get("parameters", {})).get("num_cpus", 0)),
            object_store_bytes_per_gpu=int(
                dict(plan.get("parameters", {})).get("object_store_bytes_per_gpu", 0)
            ),
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--profile",
        choices=("matrix", "scale", "dgx-scale"),
        default="matrix",
        help=(
            "matrix preserves the general case generator; scale runs the paired "
            "4-GPU comparison; dgx-scale runs strong and weak 1/2/4/8/16-GPU "
            "curves plus a 16-GPU actor-only control"
        ),
    )
    parser.add_argument("--arms", type=_csv, default=list(ARM_NAMES))
    parser.add_argument("--capacities", type=_positive_csv, default=[1, 2, 4])
    parser.add_argument("--workloads", type=_csv, default=list(WORKLOADS))
    parser.add_argument("--ranks", type=_rank_csv, default=["all"])
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--gpu-indices",
        type=_nonnegative_csv,
        help="physical GPU indices to make available (default: lowest inventory indices)",
    )
    parser.add_argument("--rows", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument(
        "--dgx-weak-rows-per-gpu",
        type=int,
        default=DGX_WEAK_ROWS_PER_GPU,
        help="weak-scaling rows per GPU for dgx-scale",
    )
    parser.add_argument(
        "--dgx-weak-blocks-per-gpu",
        type=int,
        default=DGX_WEAK_BLOCKS_PER_GPU,
        help="weak-scaling blocks per GPU for dgx-scale",
    )
    parser.add_argument(
        "--dgx-weak-max-rows",
        type=int,
        help=(
            "optional practical cap on total weak-scaling rows; the plan records "
            "the resulting per-GPU work at every capacity"
        ),
    )
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--gpu-map-work-iterations", type=int, default=0)
    parser.add_argument("--materialize-boundaries", action="store_true")
    parser.add_argument(
        "--map-actors-per-stage",
        type=int,
        help="fixed actors per GPU map stage (default: match each case capacity)",
    )
    parser.add_argument(
        "--map-actors-max-per-stage",
        type=int,
        help=(
            "optional elastic maximum for each GPU map stage; the existing "
            "--map-actors-per-stage value is its minimum"
        ),
    )
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument(
        "--object-store-bytes",
        type=int,
        default=DGX_OBJECT_STORE_BYTES_PER_GPU,
        help=(
            "base object-store bytes; dgx-scale multiplies this by case GPU "
            "capacity to preserve bytes per GPU"
        ),
    )
    parser.add_argument("--spill-object-store-bytes", type=int, default=128 * 1024**2)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--case-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--cluster-start-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--job-cleanup-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--execute-local",
        action="store_true",
        help="install isolated wheel overlays and run local Ray/GPU cases",
    )
    args = parser.parse_args(raw_argv)
    if args.repetitions is None:
        args.repetitions = (
            5
            if args.profile == "scale"
            else (DGX_DEFAULT_REPETITIONS if args.profile == "dgx-scale" else 1)
        )
    if args.rows is None:
        args.rows = (
            SCALE_DEFAULT_ROWS if args.profile in {"scale", "dgx-scale"} else 16_000_000
        )
    if args.blocks is None:
        args.blocks = (
            SCALE_DEFAULT_BLOCKS if args.profile in {"scale", "dgx-scale"} else 32
        )
    if args.num_cpus is None:
        args.num_cpus = (
            DGX_MIN_CPUS
            if args.profile == "dgx-scale"
            else min(16, os.cpu_count() or 8)
        )
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        parser.error("run-id must contain 1-40 lowercase letters, digits, or hyphens")
    if args.execute_local and args.output_root.resolve() != RESULTS_ROOT.resolve():
        parser.error(
            f"--execute-local requires the /raid-backed results root {RESULTS_ROOT}"
        )
    try:
        _validate_selection(args.arms, ARM_NAMES, "arms")
        _validate_selection(args.workloads, WORKLOADS, "workloads")
    except ValueError as error:
        parser.error(str(error))
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    for name in (
        "rows",
        "blocks",
        "groups",
        "batch_size",
        "num_cpus",
        "object_store_bytes",
        "spill_object_store_bytes",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.gpu_map_work_iterations < 0:
        parser.error("gpu-map-work-iterations cannot be negative")
    if args.gpu_map_work_iterations > (1 << 31) - 1:
        parser.error("gpu-map-work-iterations exceeds the CUDA int32 limit")
    if args.map_actors_per_stage is not None and args.map_actors_per_stage < 1:
        parser.error("map-actors-per-stage must be positive")
    if args.map_actors_max_per_stage is not None and args.map_actors_max_per_stage < 1:
        parser.error("map-actors-max-per-stage must be positive")
    if (
        args.map_actors_per_stage is not None
        and args.map_actors_max_per_stage is not None
        and args.map_actors_max_per_stage < args.map_actors_per_stage
    ):
        parser.error("map-actors-max-per-stage cannot be smaller than the minimum")
    if args.profile in {"scale", "dgx-scale"} and (
        args.map_actors_per_stage is not None
        or args.map_actors_max_per_stage is not None
    ):
        parser.error(
            f"{args.profile} fixes map actors per arm; do not set global map actor "
            "bounds"
        )
    for name in ("dgx_weak_rows_per_gpu", "dgx_weak_blocks_per_gpu"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.dgx_weak_max_rows is not None:
        if args.dgx_weak_max_rows < args.dgx_weak_rows_per_gpu:
            parser.error("dgx-weak-max-rows must be at least one GPU's weak row target")
    if args.profile == "dgx-scale":
        owned_options = ("--arms", "--capacities", "--workloads", "--ranks")
        overridden = [
            option
            for option in owned_options
            if any(
                token == option or token.startswith(option + "=") for token in raw_argv
            )
        ]
        if overridden:
            parser.error(
                "dgx-scale owns arms, capacities, workloads, and ranks; remove "
                + ", ".join(overridden)
            )
        if args.num_cpus < DGX_MIN_CPUS:
            parser.error(
                f"dgx-scale requires at least {DGX_MIN_CPUS} logical CPUs so "
                "16-way GPU scheduling is not CPU-throttled"
            )
    if args.case_timeout_seconds < 45:
        parser.error(
            "case timeout must be at least 45 seconds for closed-wait evidence"
        )
    if args.sample_interval_seconds <= 0:
        parser.error("sample interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "execution-started.json").exists():
        raise RuntimeError(f"run has already started and is immutable: {run_dir}")
    staged_harness = stage_local_harness(run_dir)
    if args.profile == "scale":
        cases = build_scale_evidence_cases(
            run_id=args.run_id,
            repetitions=args.repetitions,
            seed=args.seed,
        )
        campaign_arms = ["stock", "minimal"]
        campaign_capacities = [4]
        campaign_workloads = ["incident", "actor-only"]
        requested_ranks: list[int | str] = [1, 4, "default"]
    elif args.profile == "dgx-scale":
        cases = build_dgx_scale_cases(
            run_id=args.run_id,
            repetitions=args.repetitions,
            seed=args.seed,
            strong_rows=args.rows,
            strong_blocks=args.blocks,
            weak_rows_per_gpu=args.dgx_weak_rows_per_gpu,
            weak_blocks_per_gpu=args.dgx_weak_blocks_per_gpu,
            weak_max_rows=args.dgx_weak_max_rows,
        )
        campaign_arms = ["stock", "minimal"]
        campaign_capacities = list(DGX_CAPACITIES)
        campaign_workloads = ["incident", "actor-only"]
        requested_ranks = [*DGX_CAPACITIES, "default"]
    else:
        cases = build_cases(
            run_id=args.run_id,
            arms=args.arms,
            capacities=args.capacities,
            workloads=args.workloads,
            ranks=args.ranks,
            repetitions=args.repetitions,
            seed=args.seed,
        )
        campaign_arms = list(args.arms)
        campaign_capacities = list(args.capacities)
        campaign_workloads = list(args.workloads)
        requested_ranks = list(args.ranks)
    artifact_readiness = inspect_artifacts(campaign_arms)
    inventory = gpu_inventory()
    available_indices = [item["index"] for item in inventory.get("devices", [])]
    selected_gpu_indices = (
        list(args.gpu_indices)
        if args.gpu_indices is not None
        else available_indices[: max(campaign_capacities)]
    )
    plan = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "mode": "execute-local" if args.execute_local else "dry-run",
        "profile": args.profile,
        "run_id": args.run_id,
        "seed": args.seed,
        "randomization": (
            "blocked-by-repetition"
            if args.profile in {"scale", "dgx-scale"}
            else "complete-case-shuffle"
        ),
        "arms": campaign_arms,
        "capacities": campaign_capacities,
        "workloads": campaign_workloads,
        "requested_ranks": requested_ranks,
        "repetitions": args.repetitions,
        "selected_gpu_indices": selected_gpu_indices,
        "gpu_inventory": inventory,
        "state_reader": "core-gcs",
        "artifact_readiness": artifact_readiness,
        "staged_harness": staged_harness,
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
        "dgx_scale_contract": (
            {
                "capacities": list(DGX_CAPACITIES),
                "strong_rows": args.rows,
                "strong_blocks": args.blocks,
                "weak_rows_per_gpu": args.dgx_weak_rows_per_gpu,
                "weak_blocks_per_gpu": args.dgx_weak_blocks_per_gpu,
                "weak_max_rows": args.dgx_weak_max_rows,
                "stock_completing_capacities": list(DGX_STOCK_COMPLETING_CAPACITIES),
                "stock_safe_frontier": {
                    str(capacity): [
                        {
                            "workaround_id": f"m{map_actors}-r{shuffle_ranks}",
                            "map_actors_per_stage": map_actors,
                            "shuffle_ranks": shuffle_ranks,
                        }
                        for map_actors, shuffle_ranks in values
                    ]
                    for capacity, values in DGX_STOCK_WORKAROUNDS.items()
                },
                "actor_control_capacity": DGX_ACTOR_CONTROL_CAPACITY,
                "actor_control_map_actors_per_stage": (DGX_ACTOR_CONTROL_MAP_ACTORS),
                "cases_per_repetition": DGX_CASES_PER_REPETITION,
                "evidence_repetitions": 5,
                "minimum_logical_cpus": DGX_MIN_CPUS,
                "object_store_bytes_per_gpu": args.object_store_bytes,
                "object_store_bytes_by_capacity": {
                    str(capacity): args.object_store_bytes * capacity
                    for capacity in DGX_CAPACITIES
                },
            }
            if args.profile == "dgx-scale"
            else None
        ),
        "parameters": {
            "rows": "per-case" if args.profile == "dgx-scale" else args.rows,
            "blocks": "per-case" if args.profile == "dgx-scale" else args.blocks,
            "groups": args.groups,
            "batch_size": args.batch_size,
            "gpu_map_work_iterations": args.gpu_map_work_iterations,
            "materialize_boundaries": args.materialize_boundaries,
            "map_actors_per_stage": (
                "per-case"
                if args.profile in {"scale", "dgx-scale"}
                else args.map_actors_per_stage or "capacity"
            ),
            "map_actors_max_per_stage": (
                "per-case"
                if args.profile in {"scale", "dgx-scale"}
                else args.map_actors_max_per_stage
                or args.map_actors_per_stage
                or "capacity"
            ),
            "num_cpus": args.num_cpus,
            "object_store_bytes": (
                "per-case" if args.profile == "dgx-scale" else args.object_store_bytes
            ),
            "object_store_bytes_per_gpu": (
                args.object_store_bytes if args.profile == "dgx-scale" else None
            ),
            "spill_object_store_bytes": args.spill_object_store_bytes,
            "sample_interval_seconds": args.sample_interval_seconds,
            "case_timeout_seconds": args.case_timeout_seconds,
        },
        "evidence_metrics": (
            [
                *DGX_METRICS,
                "strong_speedup",
                "strong_parallel_efficiency",
                "weak_parallel_efficiency",
                "correctness_oracles",
                "job_and_cluster_cleanup",
            ]
            if args.profile == "dgx-scale"
            else None
        ),
        "safety": {
            "dry_run_default": True,
            "execution_flag": "--execute-local",
            "ray_tmp_root": str(SHM_ROOT),
            "ray_object_spilling_leaf": "<case-directory>/ray-spill",
            "ray_object_spilling_results_root": str(RESULTS_ROOT),
            "spill_storage_audited": True,
            "spill_removal_required_for_cleanup": True,
            "uses_ray_stop": False,
            "one_isolated_cluster_per_case": True,
        },
    }
    _atomic_json(run_dir / "plan.json", plan)
    if not args.execute_local:
        report = build_prerequisite_report(plan=plan, executions=[], executed=False)
        _atomic_json(run_dir / "prerequisite-report.json", report)
        if args.profile == "scale":
            _atomic_json(
                run_dir / "scale-evidence-report.json", report["scale_evidence"]
            )
        elif args.profile == "dgx-scale":
            _atomic_json(
                run_dir / "dgx-scale-evidence-report.json",
                report["dgx_scale_evidence"],
            )
        print(
            f"Dry run only: wrote {len(cases)} randomized cases to {run_dir / 'plan.json'}"
        )
        if artifact_readiness["blockers"]:
            print("Artifact blockers:")
            for blocker in artifact_readiness["blockers"]:
                print(f"- {blocker}")
        print(
            "No Ray cluster was started. Add --execute-local after reviewing the plan."
        )
        return 0

    blockers = list(artifact_readiness["blockers"])
    if inventory.get("status") != "ready":
        blockers.append(f"GPU inventory unavailable: {inventory.get('error')}")
    if len(selected_gpu_indices) < max(campaign_capacities):
        blockers.append(
            f"selected {len(selected_gpu_indices)} GPUs but capacity "
            f"{max(campaign_capacities)} is required"
        )
    unknown_gpu_indices = sorted(set(selected_gpu_indices) - set(available_indices))
    if unknown_gpu_indices:
        blockers.append(f"selected GPU indices are not visible: {unknown_gpu_indices}")
    if blockers:
        report = build_prerequisite_report(plan=plan, executions=[], executed=False)
        report["execution_blockers"] = blockers
        _atomic_json(run_dir / "prerequisite-report.json", report)
        if args.profile == "scale":
            _atomic_json(
                run_dir / "scale-evidence-report.json", report["scale_evidence"]
            )
        elif args.profile == "dgx-scale":
            _atomic_json(
                run_dir / "dgx-scale-evidence-report.json",
                report["dgx_scale_evidence"],
            )
        raise RuntimeError("local execution blocked:\n- " + "\n- ".join(blockers))

    _atomic_json(
        run_dir / "execution-started.json",
        {"started_at": _utc_now(), "pid": os.getpid(), "case_count": len(cases)},
    )
    SHM_ROOT.mkdir(parents=True, exist_ok=True)
    overlays = install_overlays(run_dir, artifact_readiness, campaign_arms)
    executions = []
    arm_reports = artifact_readiness["arms"]
    for index, case in enumerate(cases, 1):
        physical = selected_gpu_indices[: case.capacity]
        execution = execute_case(
            index=index,
            case=case,
            run_dir=run_dir,
            overlay=overlays[case.arm],
            artifact=arm_reports[case.arm],
            gpu_indices=physical,
            args=args,
            staged_harness=staged_harness,
        )
        executions.append(execution)
        print(
            f"[{index}/{len(cases)}] {case.case_id}: {execution.get('outcome')} "
            f"cleanup={dict(execution.get('cluster_cleanup', {})).get('cleanup_proven')}"
        )
        report = build_prerequisite_report(
            plan=plan, executions=executions, executed=True
        )
        _atomic_json(run_dir / "prerequisite-report.json", report)
        if args.profile == "scale":
            _atomic_json(
                run_dir / "scale-evidence-report.json", report["scale_evidence"]
            )
        elif args.profile == "dgx-scale":
            _atomic_json(
                run_dir / "dgx-scale-evidence-report.json",
                report["dgx_scale_evidence"],
            )
        if not dict(execution.get("cluster_cleanup", {})).get("cleanup_proven"):
            raise RuntimeError(
                f"cluster cleanup could not be proven for {case.case_id}; stopping campaign"
            )
    report = build_prerequisite_report(plan=plan, executions=executions, executed=True)
    _atomic_json(run_dir / "prerequisite-report.json", report)
    if args.profile == "scale":
        scale_report = report["scale_evidence"]
        _atomic_json(run_dir / "scale-evidence-report.json", scale_report)
        print(f"Local scale evidence status: {scale_report['status']}")
        return 0 if scale_report["status"] == "pass" else 2
    if args.profile == "dgx-scale":
        dgx_report = report["dgx_scale_evidence"]
        _atomic_json(run_dir / "dgx-scale-evidence-report.json", dgx_report)
        print(f"Local DGX scale evidence status: {dgx_report['status']}")
        return 0 if dgx_report["status"] == "pass" else 2
    print(f"Local prerequisite status: {report['status']}")
    return 0 if report["status"] == "ready_for_cloud" else 2


if __name__ == "__main__":
    raise SystemExit(main())
