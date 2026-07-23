#!/usr/bin/env python3
"""Run and observe the physical GPU-owner chain behind resource admission.

The evidence collected here is deliberately structural.  An unchanged state
snapshot is not a deadlock: a closed wait is reported only when application
progress is stalled at the topology GPU ceiling, every logical GPU is owned,
and a GPU actor, task, or placement group is waiting for that same pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_EXPECTED_FAILURE_MARKER = "RAY_GPU_ADMISSION_EXPECTED_FAILURE"
_MIN_CLOSED_WAIT_SAMPLES = 3
_MIN_CLOSED_WAIT_SECONDS = 30.0
_STATE_RECORD_LIMIT = 2_000
_GPU_EPSILON = 0.01
_UINT64_MASK = (1 << 64) - 1
_TELEMETRY_DRAIN_TIMEOUT_SECONDS = 10.0
_TELEMETRY_DRAIN_POLL_SECONDS = 0.05
_TELEMETRY_QUIESCENCE_SECONDS = 1.0
_TELEMETRY_QUIESCENCE_POLLS = 3
_PROGRESS_EMIT_INTERVAL_SECONDS = 0.5


def _batch_rows(batch: object) -> int:
    try:
        return len(batch)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _read_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


class _ProgressUdf:
    def __init__(self, progress: object, stage: str):
        import ray

        self._progress = progress
        self._ray = ray
        self._stage = stage
        actor_id = ray.get_runtime_context().get_actor_id()
        self._actor_id = actor_id.hex() if hasattr(actor_id, "hex") else str(actor_id)
        self._sent_first_input = False
        self._last_progress_emit_monotonic_s: float | None = None
        self._boot_id = _read_boot_id()
        self._progress.register_actor.remote(
            self._stage,
            self._actor_id,
            1.0,
            time.time(),
            time.monotonic(),
            self._boot_id,
        )

    def _input(self, batch: object) -> None:
        rows = _batch_rows(batch)
        if rows <= 0:
            return
        if self._sent_first_input:
            return
        event_epoch_s = time.time()
        event_monotonic_s = time.monotonic()
        self._ray.get(
            self._progress.input.remote(
                self._stage,
                self._actor_id,
                rows,
                event_epoch_s,
                event_monotonic_s,
                self._boot_id,
            )
        )
        self._sent_first_input = True

    def _record(self, batch: object) -> None:
        # Progress remains advisory and fire-and-forget. First-input telemetry
        # alone uses one acknowledged RPC per actor so no delayed event can
        # revise the causal stage boundary after materialization completes.
        event_epoch_s = time.time()
        event_monotonic_s = time.monotonic()
        if (
            self._last_progress_emit_monotonic_s is not None
            and event_monotonic_s - self._last_progress_emit_monotonic_s
            < _PROGRESS_EMIT_INTERVAL_SECONDS
        ):
            return
        self._last_progress_emit_monotonic_s = event_monotonic_s
        self._progress.record.remote(
            self._stage,
            self._actor_id,
            _batch_rows(batch),
            event_epoch_s,
            event_monotonic_s,
            self._boot_id,
        )


class AddKey(_ProgressUdf):
    def __init__(
        self,
        groups: int,
        gpu_work_iterations: int,
        progress: object,
        stage: str,
    ):
        super().__init__(progress, stage)
        self._groups = groups
        self._gpu_work_iterations = gpu_work_iterations

    def __call__(self, batch):
        self._input(batch)
        result = batch.copy(deep=True)
        if self._gpu_work_iterations:
            import cupy as cp

            scratch = cp.asarray(result["id"].values, dtype=cp.float32)
            for _ in range(self._gpu_work_iterations):
                scratch = cp.sin(scratch * cp.float32(1e-6) + cp.float32(0.1))
            # This benchmark models synchronous GPU preprocessing or inference.
            # Synchronize so its device work cannot escape the measured UDF stage.
            cp.cuda.get_current_stream().synchronize()
        result["key"] = result["id"] % self._groups
        self._record(result)
        return result


class SumGroup(_ProgressUdf):
    def __call__(self, batch):
        self._input(batch)
        result = batch.iloc[:1].copy(deep=True)
        result["id"] = batch["id"].sum()
        result = result[["key", "id"]]
        self._record(result)
        return result


class Identity(_ProgressUdf):
    def __call__(self, batch):
        self._input(batch)
        self._record(batch)
        return batch


class AddOne(_ProgressUdf):
    def __call__(self, batch):
        self._input(batch)
        result = batch.copy(deep=True)
        result["id"] = result["id"] + 1
        self._record(result)
        return result


class RaiseInjected(_ProgressUdf):
    def __call__(self, batch):
        self._input(batch)
        self._record(batch)
        raise RuntimeError(_EXPECTED_FAILURE_MARKER)


class _ProgressTracker:
    """Zero-resource Ray actor receiving low-volume stage counters."""

    def __init__(
        self,
        origin_epoch_s: float,
        expected_actor_counts: dict[str, int],
        origin_monotonic_s: float,
        origin_boot_id: str,
    ):
        self._origin_epoch_s = float(origin_epoch_s)
        self._origin_monotonic_s = float(origin_monotonic_s)
        self._origin_boot_id = str(origin_boot_id)
        self._tracker_boot_id = _read_boot_id()
        self._expected_actor_counts = dict(expected_actor_counts)
        self._sequence = 0
        self._stages: dict[str, dict[str, object]] = {}
        self._actors: dict[str, dict[str, object]] = {}

    def _elapsed(
        self,
        event_epoch_s: float,
        event_monotonic_s: float,
        event_boot_id: str,
    ) -> tuple[float, str]:
        if event_boot_id and event_boot_id == self._origin_boot_id:
            return (
                max(0.0, float(event_monotonic_s) - self._origin_monotonic_s),
                "single-boot-monotonic",
            )
        return (
            max(0.0, float(event_epoch_s) - self._origin_epoch_s),
            "cross-boot-wall-epoch-unverified",
        )

    def _receipt(self) -> tuple[str, float]:
        received_epoch_s = time.time()
        elapsed, _ = self._elapsed(
            received_epoch_s,
            time.monotonic(),
            self._tracker_boot_id,
        )
        return (
            _utc_from_epoch(received_epoch_s),
            elapsed,
        )

    @staticmethod
    def _event_bounds(
        state: dict[str, object],
        prefix: str,
        event_epoch_s: float,
        event_monotonic_s: float,
        event_boot_id: str,
        clock_basis: str,
        elapsed: float,
    ) -> None:
        first_key = f"first_{prefix}_s"
        last_key = f"last_{prefix}_s"
        if not isinstance(state.get(first_key), (int, float)) or elapsed < float(
            state[first_key]
        ):
            state[first_key] = elapsed
            state[f"first_{prefix}_at"] = _utc_from_epoch(event_epoch_s)
            state[f"first_{prefix}_monotonic_s"] = float(event_monotonic_s)
            state[f"first_{prefix}_boot_id"] = event_boot_id
            state[f"first_{prefix}_clock_basis"] = clock_basis
        if not isinstance(state.get(last_key), (int, float)) or elapsed > float(
            state[last_key]
        ):
            state[last_key] = elapsed
            state[f"last_{prefix}_at"] = _utc_from_epoch(event_epoch_s)
            state[f"last_{prefix}_monotonic_s"] = float(event_monotonic_s)
            state[f"last_{prefix}_boot_id"] = event_boot_id
            state[f"last_{prefix}_clock_basis"] = clock_basis

    def register_actor(
        self,
        stage: str,
        actor_id: str,
        gpus: float,
        event_epoch_s: float,
        event_monotonic_s: float,
        event_boot_id: str,
    ) -> None:
        elapsed, clock_basis = self._elapsed(
            event_epoch_s, event_monotonic_s, event_boot_id
        )
        received_at, received_s = self._receipt()
        actor = self._actors.setdefault(
            actor_id,
            {
                "stage": stage,
                "gpus": float(gpus),
                "ready_at": _utc_from_epoch(event_epoch_s),
                "ready_s": elapsed,
                "ready_monotonic_s": float(event_monotonic_s),
                "ready_boot_id": event_boot_id,
                "ready_clock_basis": clock_basis,
                "ready_received_at": received_at,
                "ready_received_s": received_s,
                "registration_events": 0,
                "progress_events": 0,
                "progress_rows_sampled": 0,
            },
        )
        actor["registration_events"] = int(actor["registration_events"]) + 1
        ready_s = actor.get("ready_s")
        if not isinstance(ready_s, (int, float)) or elapsed < float(ready_s):
            actor.update(
                stage=stage,
                gpus=float(gpus),
                ready_at=_utc_from_epoch(event_epoch_s),
                ready_s=elapsed,
                ready_monotonic_s=float(event_monotonic_s),
                ready_boot_id=event_boot_id,
                ready_clock_basis=clock_basis,
                ready_received_at=received_at,
                ready_received_s=received_s,
            )
        self._sequence += 1

    def input(
        self,
        stage: str,
        actor_id: str,
        rows: int,
        event_epoch_s: float,
        event_monotonic_s: float,
        event_boot_id: str,
    ) -> None:
        elapsed, clock_basis = self._elapsed(
            event_epoch_s, event_monotonic_s, event_boot_id
        )
        received_at, received_s = self._receipt()
        state = self._stages.setdefault(stage, {})
        state.setdefault("first_input_events", 0)
        state.setdefault("first_input_rows_sampled", 0)
        self._event_bounds(
            state,
            "input",
            event_epoch_s,
            event_monotonic_s,
            event_boot_id,
            clock_basis,
            elapsed,
        )
        state["first_input_events"] = int(state["first_input_events"]) + 1
        state["first_input_rows_sampled"] = int(
            state["first_input_rows_sampled"]
        ) + int(rows)
        state["input_received_at"] = received_at
        state["input_received_s"] = received_s
        actor = self._actors.setdefault(
            actor_id,
            {
                "stage": stage,
                "gpus": 1.0,
                "ready_at": _utc_from_epoch(event_epoch_s),
                "ready_s": elapsed,
                "ready_monotonic_s": float(event_monotonic_s),
                "ready_boot_id": event_boot_id,
                "ready_clock_basis": clock_basis,
                "ready_received_at": received_at,
                "ready_received_s": received_s,
                "registration_events": 0,
                "progress_events": 0,
                "progress_rows_sampled": 0,
            },
        )
        self._event_bounds(
            actor,
            "input",
            event_epoch_s,
            event_monotonic_s,
            event_boot_id,
            clock_basis,
            elapsed,
        )
        actor["input_received_at"] = received_at
        actor["input_received_s"] = received_s
        self._sequence += 1

    def record(
        self,
        stage: str,
        actor_id: str,
        rows: int,
        event_epoch_s: float,
        event_monotonic_s: float,
        event_boot_id: str,
    ) -> None:
        elapsed, clock_basis = self._elapsed(
            event_epoch_s, event_monotonic_s, event_boot_id
        )
        received_at, received_s = self._receipt()
        state = self._stages.setdefault(stage, {})
        state.setdefault("progress_events", 0)
        state.setdefault("progress_rows_sampled", 0)
        self._event_bounds(
            state,
            "progress",
            event_epoch_s,
            event_monotonic_s,
            event_boot_id,
            clock_basis,
            elapsed,
        )
        state["progress_events"] = int(state["progress_events"]) + 1
        state["progress_rows_sampled"] = int(state["progress_rows_sampled"]) + int(rows)
        state["progress_received_at"] = received_at
        state["progress_received_s"] = received_s
        actor = self._actors.setdefault(
            actor_id,
            {
                "stage": stage,
                "gpus": 1.0,
                "ready_at": _utc_from_epoch(event_epoch_s),
                "ready_s": elapsed,
                "ready_monotonic_s": float(event_monotonic_s),
                "ready_boot_id": event_boot_id,
                "ready_clock_basis": clock_basis,
                "ready_received_at": received_at,
                "ready_received_s": received_s,
                "registration_events": 0,
                "progress_events": 0,
                "progress_rows_sampled": 0,
            },
        )
        self._event_bounds(
            actor,
            "progress",
            event_epoch_s,
            event_monotonic_s,
            event_boot_id,
            clock_basis,
            elapsed,
        )
        actor["progress_received_at"] = received_at
        actor["progress_received_s"] = received_s
        actor["progress_events"] = int(actor["progress_events"]) + 1
        actor["progress_rows_sampled"] = int(actor["progress_rows_sampled"]) + int(rows)
        self._sequence += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "sequence": self._sequence,
            "origin_epoch_s": self._origin_epoch_s,
            "origin_monotonic_s": self._origin_monotonic_s,
            "origin_boot_id": self._origin_boot_id,
            "expected_actor_counts": dict(self._expected_actor_counts),
            "stages": {
                name: dict(value) for name, value in sorted(self._stages.items())
            },
            "actors": {
                actor_id: dict(value)
                for actor_id, value in sorted(self._actors.items())
            },
            "observed_at": _utc_now(),
        }


def _progress_telemetry_coverage(snapshot: object) -> dict[str, object]:
    """Validate fixed-pool constructor and first-input telemetry."""

    if not isinstance(snapshot, dict):
        return {
            "complete": False,
            "issue": "progress snapshot is unavailable",
            "stages": [],
        }
    expected = snapshot.get("expected_actor_counts")
    actors = snapshot.get("actors")
    stages = snapshot.get("stages")
    if not isinstance(expected, dict) or not expected:
        return {
            "complete": False,
            "issue": "expected fixed-pool actor counts are unavailable",
            "stages": [],
        }
    if not isinstance(actors, dict) or not isinstance(stages, dict):
        return {
            "complete": False,
            "issue": "actor registry or stage input telemetry is unavailable",
            "stages": [],
        }
    rows = []
    for stage, expected_count_value in sorted(expected.items()):
        expected_count = int(expected_count_value)
        registered = [
            value
            for value in actors.values()
            if isinstance(value, dict)
            and value.get("stage") == stage
            and isinstance(value.get("ready_s"), (int, float))
        ]
        with_input = [
            value
            for value in registered
            if isinstance(value.get("first_input_s"), (int, float))
        ]
        registration_events = sum(
            int(value.get("registration_events", 0)) for value in registered
        )
        stage_state = stages.get(stage)
        stage_first_input = (
            stage_state.get("first_input_s") if isinstance(stage_state, dict) else None
        )
        stage_first_progress = (
            stage_state.get("first_progress_s")
            if isinstance(stage_state, dict)
            else None
        )
        first_input_rows_sampled = (
            stage_state.get("first_input_rows_sampled")
            if isinstance(stage_state, dict)
            else None
        )
        rows.append(
            {
                "stage": stage,
                "expected_actors": expected_count,
                "registered_actors": len(registered),
                "registration_events": registration_events,
                "actors_with_first_input": len(with_input),
                "stage_first_input_s": stage_first_input,
                "first_input_rows_sampled": first_input_rows_sampled,
                "stage_first_progress_s": stage_first_progress,
                "complete": len(registered) == expected_count
                and registration_events == expected_count
                and isinstance(stage_first_input, (int, float))
                and isinstance(first_input_rows_sampled, (int, float))
                and first_input_rows_sampled > 0
                and isinstance(stage_first_progress, (int, float)),
            }
        )
    complete = bool(rows) and all(bool(row["complete"]) for row in rows)
    return {
        "complete": complete,
        "issue": None if complete else "fixed-pool telemetry is incomplete",
        "stages": rows,
    }


def _drain_progress_telemetry(
    ray_module: object,
    progress: object,
    *,
    timeout_seconds: float = _TELEMETRY_DRAIN_TIMEOUT_SECONDS,
) -> tuple[dict[str, object], dict[str, object]]:
    """Boundedly wait for cross-actor telemetry before the final sample."""

    started = time.monotonic()
    deadline = started + timeout_seconds
    attempts = 0
    latest: dict[str, object] = {}
    coverage = _progress_telemetry_coverage(latest)
    stable_sequence: int | None = None
    stable_since: float | None = None
    stable_polls = 0
    quiescent = False
    while time.monotonic() < deadline:
        attempts += 1
        remaining = max(0.01, deadline - time.monotonic())
        try:
            candidate = ray_module.get(  # type: ignore[attr-defined]
                progress.snapshot.remote(),
                timeout=min(2.0, remaining),
            )
        except Exception as error:
            coverage = {
                "complete": False,
                "issue": f"snapshot failed: {type(error).__name__}: {error}",
                "stages": [],
            }
        else:
            if isinstance(candidate, dict):
                latest = candidate
            coverage = _progress_telemetry_coverage(latest)
            latest["telemetry_coverage"] = coverage
            if coverage["complete"] is True:
                sequence = latest.get("sequence")
                if isinstance(sequence, int) and sequence == stable_sequence:
                    stable_polls += 1
                elif isinstance(sequence, int):
                    stable_sequence = sequence
                    stable_since = time.monotonic()
                    stable_polls = 1
                else:
                    stable_sequence = None
                    stable_since = None
                    stable_polls = 0
                quiescent = (
                    stable_since is not None
                    and stable_polls >= _TELEMETRY_QUIESCENCE_POLLS
                    and time.monotonic() - stable_since >= _TELEMETRY_QUIESCENCE_SECONDS
                )
                if quiescent:
                    break
            else:
                stable_sequence = None
                stable_since = None
                stable_polls = 0
        time.sleep(min(_TELEMETRY_DRAIN_POLL_SECONDS, remaining))
    elapsed = time.monotonic() - started
    quiescence = {
        "complete": quiescent,
        "stable_sequence": stable_sequence,
        "stable_polls": stable_polls,
        "required_polls": _TELEMETRY_QUIESCENCE_POLLS,
        "stable_duration_s": (
            time.monotonic() - stable_since if stable_since is not None else 0.0
        ),
        "required_duration_s": _TELEMETRY_QUIESCENCE_SECONDS,
    }
    result = {
        "complete": coverage.get("complete") is True and quiescent,
        "attempts": attempts,
        "elapsed_s": elapsed,
        "timeout_s": timeout_seconds,
        "coverage": coverage,
        "quiescence": quiescence,
    }
    latest["telemetry_coverage"] = coverage
    latest["telemetry_quiescence"] = quiescence
    return latest, result


def _device_name(value: object) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _is_nvidia_l4(value: object) -> bool:
    return _device_name(value) == "NVIDIA L4"


def _pci_address(value: object) -> tuple[int, int, int]:
    """Normalize CUDA/NVML PCI bus IDs despite their domain-width difference."""

    text = _device_name(value)
    domain_bus, device_function = text.rsplit(":", 1)
    domain, bus = domain_bus.split(":", 1)
    device, _function = device_function.split(".", 1)
    return int(domain, 16), int(bus, 16), int(device, 16)


class _NodeGpuMonitor:
    """NVML sampler pinned to a node without reserving CPU or logical GPU."""

    def __init__(self, node_id: str, require_one_l4: bool):
        self._node_id = node_id
        self._pynvml = None
        self._nvml_indices: list[int] = []
        self._startup_probe = None
        self._initialization_error = None
        try:
            import cupy
            import pynvml

            pynvml.nvmlInit()
            nvml_device_count = int(pynvml.nvmlDeviceGetCount())
            visible_devices = int(cupy.cuda.runtime.getDeviceCount())
            if require_one_l4 and (visible_devices != 1 or nvml_device_count != 1):
                raise RuntimeError(
                    "G6 evidence nodes must expose exactly one CUDA and NVML GPU"
                )
            cuda_pci = {
                _pci_address(cupy.cuda.Device(index).pci_bus_id)
                for index in range(visible_devices)
            }
            self._nvml_indices = [
                index
                for index in range(nvml_device_count)
                if _pci_address(
                    pynvml.nvmlDeviceGetPciInfo(
                        pynvml.nvmlDeviceGetHandleByIndex(index)
                    ).busId
                )
                in cuda_pci
            ]
            if len(self._nvml_indices) != visible_devices:
                raise RuntimeError(
                    "could not map every CUDA-visible GPU to exactly one NVML device"
                )
            properties = cupy.cuda.runtime.getDeviceProperties(0)
            raw_name = properties.get("name", b"unknown")
            cuda_name = _device_name(raw_name)
            if require_one_l4 and not _is_nvidia_l4(cuda_name):
                raise RuntimeError(f"expected NVIDIA L4, CUDA reports {cuda_name!r}")
            nvml_name = pynvml.nvmlDeviceGetName(
                pynvml.nvmlDeviceGetHandleByIndex(self._nvml_indices[0])
            )
            nvml_name = _device_name(nvml_name)
            if require_one_l4 and not _is_nvidia_l4(nvml_name):
                raise RuntimeError(f"expected NVIDIA L4, NVML reports {nvml_name!r}")
            allocation = cupy.ones(4 * 1024 * 1024, dtype=cupy.float32)
            checksum = float(cupy.sum(allocation[:4096]).get())
            cupy.cuda.get_current_stream().synchronize()
            del allocation
            cupy.get_default_memory_pool().free_all_blocks()
            self._startup_probe = {
                "cuda_device_name": cuda_name,
                "nvml_device_name": str(nvml_name),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "visible_cuda_device_count": visible_devices,
                "monitored_nvml_indices": list(self._nvml_indices),
                "allocation_bytes": 16 * 1024 * 1024,
                "allocation_checksum": checksum,
            }
            self._pynvml = pynvml
        except Exception as error:  # pragma: no cover - exercised on cloud nodes.
            if "pynvml" in locals():
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            self._initialization_error = {
                "type": type(error).__name__,
                "message": str(error),
            }

    def sample(self) -> dict[str, object]:
        if self._pynvml is None:
            return {
                "node_id": self._node_id,
                "observed_at": _utc_now(),
                "error": self._initialization_error,
            }
        pynvml = self._pynvml
        devices = []
        for index in self._nvml_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = _device_name(name)
            if isinstance(uuid, bytes):
                uuid = uuid.decode(errors="replace")
            devices.append(
                {
                    "index": index,
                    "name": str(name),
                    "uuid": str(uuid),
                    "is_nvidia_l4": _is_nvidia_l4(name),
                    "gpu_utilization_pct": int(utilization.gpu),
                    "memory_utilization_pct": int(utilization.memory),
                    "memory_used_bytes": int(memory.used),
                    "memory_total_bytes": int(memory.total),
                }
            )
        return {
            "node_id": self._node_id,
            "observed_at": _utc_now(),
            "startup_probe": self._startup_probe,
            "devices": devices,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_from_epoch(epoch_s: float) -> str:
    return datetime.fromtimestamp(float(epoch_s), timezone.utc).isoformat()


def _atomic_json(path: Path, document: object) -> None:
    """Atomically persist JSON using a unique sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_document(
    path: Path, document: dict[str, object], lock: threading.RLock
) -> None:
    with lock:
        _atomic_json(path, document)


def _float_resources(resources: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in resources.items()}


def _state_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        state = str(record.get("state", "UNKNOWN"))
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def _is_terminal(state: str, terminal: set[str]) -> bool:
    upper = state.upper()
    return any(token in upper for token in terminal)


def _ray_state_snapshot() -> dict[str, object]:
    from benchmark.ray_core_state import read_core_state

    snapshot = read_core_state(limit=_STATE_RECORD_LIMIT)
    actors = list(snapshot["actors"])
    tasks = list(snapshot["tasks"])
    placement_groups = list(snapshot["placement_groups"])
    # Keep aggregate counts for every record, but checkpoint only records that
    # can own or request the GPU pool. Large Ray Data inputs can have thousands
    # of active CPU tasks; embedding those every five seconds creates
    # quadratic checkpoint I/O without adding evidence to the GPU wait graph.
    active_actors = [
        item
        for item in actors
        if not _is_terminal(str(item["state"]), {"DEAD"})
        and _gpu_count(item.get("required_resources")) > 0
    ]
    active_tasks = [
        item
        for item in tasks
        if not _is_terminal(str(item["state"]), {"FINISHED", "FAILED"})
        and _gpu_count(item.get("required_resources")) > 0
    ]
    active_groups = [
        item
        for item in placement_groups
        if not _is_terminal(str(item["state"]), {"REMOVED"})
        and _bundle_gpu_count(item.get("bundles")) > 0
    ]
    return {
        "counts": {
            "actors": _state_counts(actors),
            "tasks": _state_counts(tasks),
            "placement_groups": _state_counts(placement_groups),
        },
        "actors": active_actors,
        "tasks": active_tasks,
        "placement_groups": active_groups,
        "query_limit": _STATE_RECORD_LIMIT,
        "record_filter": "nonterminal GPU owners and requests",
        "possibly_truncated": any(
            len(records) >= _STATE_RECORD_LIMIT
            for records in (actors, tasks, placement_groups)
        ),
    }


def _gpu_count(resources: object) -> float:
    if not isinstance(resources, dict):
        return 0.0
    return sum(
        float(value) for key, value in resources.items() if str(key).upper() == "GPU"
    )


def _bundle_gpu_count(bundles: object) -> float:
    if not isinstance(bundles, list):
        return 0.0
    return sum(
        _gpu_count(bundle.get("resources", bundle)) if isinstance(bundle, dict) else 0.0
        for bundle in bundles
    )


def _pending_state(state: object) -> bool:
    upper = str(state).upper()
    return any(
        token in upper
        for token in ("PENDING", "SCHEDUL", "SUBMITTED", "RESTART", "RESCHEDUL")
    )


def _running_state(state: object) -> bool:
    upper = str(state).upper()
    return "ALIVE" in upper or "RUNNING" in upper or "CREATED" in upper


def _wait_graph(sample: dict[str, object]) -> dict[str, object]:
    state = sample.get("ray_state")
    if not isinstance(state, dict):
        return {
            "nodes": [],
            "edges": [],
            "owners": [],
            "requests": [],
            "observed_owner_gpus": 0.0,
        }
    cluster = dict(sample.get("cluster", {}))
    available = dict(sample.get("available", {}))
    pool_id = "logical-gpu-pool"
    nodes: list[dict[str, object]] = [
        {
            "id": pool_id,
            "kind": "resource",
            "capacity": float(cluster.get("GPU", 0.0)),
            "available": float(available.get("GPU", 0.0)),
        }
    ]
    edges: list[dict[str, str]] = []
    owners: list[str] = []
    requests: list[str] = []
    observed_owner_gpus = 0.0

    def add_owner(identifier: str, kind: str, gpu: float, record: dict) -> None:
        nonlocal observed_owner_gpus
        owners.append(identifier)
        observed_owner_gpus += gpu
        nodes.append({"id": identifier, "kind": kind, "gpu": gpu, "record": record})
        edges.append({"from": pool_id, "to": identifier, "relation": "allocated_to"})

    def add_request(identifier: str, kind: str, gpu: float, record: dict) -> None:
        requests.append(identifier)
        nodes.append({"id": identifier, "kind": kind, "gpu": gpu, "record": record})
        edges.append({"from": identifier, "to": pool_id, "relation": "waits_for"})

    actor_ids = set()
    for record in state.get("actors", []):
        if not isinstance(record, dict):
            continue
        gpu = _gpu_count(record.get("required_resources"))
        if gpu <= 0:
            continue
        actor_id = str(record.get("actor_id", "unknown"))
        actor_ids.add(actor_id)
        identifier = f"actor:{actor_id}"
        if _pending_state(record.get("state")):
            add_request(identifier, "actor_request", gpu, record)
        elif _running_state(record.get("state")):
            add_owner(identifier, "actor_owner", gpu, record)

    for record in state.get("tasks", []):
        if not isinstance(record, dict):
            continue
        gpu = _gpu_count(record.get("required_resources"))
        if gpu <= 0:
            continue
        # Actor lifetime resources are represented by the actor record.  Do
        # not double-count its currently running method as another GPU owner.
        actor_id = str(record.get("actor_id", ""))
        if actor_id and actor_id in actor_ids and _running_state(record.get("state")):
            continue
        identifier = f"task:{record.get('task_id', 'unknown')}"
        if _pending_state(record.get("state")):
            add_request(identifier, "task_request", gpu, record)
        elif _running_state(record.get("state")):
            add_owner(identifier, "task_owner", gpu, record)

    for record in state.get("placement_groups", []):
        if not isinstance(record, dict):
            continue
        gpu = _bundle_gpu_count(record.get("bundles"))
        if gpu <= 0:
            continue
        identifier = f"placement-group:{record.get('placement_group_id', 'unknown')}"
        if _pending_state(record.get("state")):
            add_request(identifier, "placement_group_request", gpu, record)
        elif _running_state(record.get("state")):
            add_owner(identifier, "placement_group_owner", gpu, record)

    return {
        "nodes": nodes,
        "edges": edges,
        "owners": owners,
        "requests": requests,
        "observed_owner_gpus": observed_owner_gpus,
    }


def _sample_wait_predicates(
    sample: dict[str, object], max_gpus: int
) -> tuple[bool, dict[str, object]]:
    cluster_gpu = float(dict(sample.get("cluster", {})).get("GPU", 0.0))
    available_gpu = float(dict(sample.get("available", {})).get("GPU", 0.0))
    graph = _wait_graph(sample)
    predicates = {
        "topology_gpu_ceiling_reached": cluster_gpu + _GPU_EPSILON >= max_gpus,
        "all_logical_gpus_unavailable": cluster_gpu > 0
        and available_gpu <= _GPU_EPSILON,
        "all_logical_gpus_have_observed_owners": float(graph["observed_owner_gpus"])
        + _GPU_EPSILON
        >= cluster_gpu,
        "pending_gpu_or_placement_group_request": bool(graph["requests"]),
    }
    return all(predicates.values()), {"predicates": predicates, "wait_graph": graph}


def _classify_closed_wait(
    samples: list[dict[str, object]], max_gpus: int
) -> dict[str, object]:
    """Classify only a sustained, resource-closed, no-progress suffix."""

    if not samples:
        return {"is_structural_closed_wait": False, "reason": "no samples"}
    latest_progress = samples[-1].get("progress")
    if not isinstance(latest_progress, dict) or not isinstance(
        latest_progress.get("sequence"), int
    ):
        return {
            "is_structural_closed_wait": False,
            "reason": "progress telemetry unavailable",
        }
    sequence = int(latest_progress["sequence"])
    qualifying = []
    latest_detail: dict[str, object] = {}
    for sample in reversed(samples):
        progress = sample.get("progress")
        if not isinstance(progress, dict) or progress.get("sequence") != sequence:
            break
        matches, detail = _sample_wait_predicates(sample, max_gpus)
        if not matches:
            break
        qualifying.append(sample)
        if not latest_detail:
            latest_detail = detail
    qualifying.reverse()
    duration = (
        float(qualifying[-1]["elapsed_s"]) - float(qualifying[0]["elapsed_s"])
        if len(qualifying) > 1
        else 0.0
    )
    sustained = (
        len(qualifying) >= _MIN_CLOSED_WAIT_SAMPLES
        and duration >= _MIN_CLOSED_WAIT_SECONDS
    )
    return {
        "is_structural_closed_wait": sustained,
        "reason": (
            "sustained no-progress GPU ownership/request cycle"
            if sustained
            else "closed resource predicates were not sustained"
        ),
        "no_progress_sequence": sequence,
        "qualifying_tail_samples": len(qualifying),
        "qualifying_tail_s": duration,
        **latest_detail,
    }


class ClusterSampler:
    def __init__(
        self,
        *,
        result_path: Path,
        interval_seconds: float,
        document: dict[str, object],
        document_lock: threading.RLock,
        progress: object,
        require_one_l4: bool,
        origin_epoch_s: float,
        origin_monotonic_s: float,
        origin_boot_id: str,
    ):
        self._result_path = result_path
        self._interval_seconds = interval_seconds
        self._document = document
        self._document_lock = document_lock
        self._progress = progress
        self._require_one_l4 = require_one_l4
        self._origin_epoch_s = float(origin_epoch_s)
        self._origin_monotonic_s = float(origin_monotonic_s)
        self._origin_boot_id = str(origin_boot_id)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._sample_lock = threading.Lock()
        self._monitors: dict[str, object] = {}
        self._stopped = False

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def stop(self, *, final_progress: dict[str, object] | None = None) -> None:
        if self._stopped:
            return
        self._stop.set()
        # Each state endpoint and telemetry RPC is explicitly bounded.  Leave
        # enough room for all of them to unwind before Ray is shut down.
        self._thread.join(timeout=15.0)
        if self._thread.is_alive():
            self._append_error(RuntimeError("sampler did not stop before shutdown"))
        if final_progress is not None and not self._thread.is_alive():
            if not self.sample(progress_override=final_progress):
                self._append_error(
                    RuntimeError("final telemetry sample was not stored")
                )
        self._stopped = True
        try:
            import ray

            for monitor in self._monitors.values():
                try:
                    ray.kill(monitor, no_restart=True)
                except Exception:
                    pass
        finally:
            self._monitors.clear()

    def _append_error(self, error: BaseException) -> None:
        with self._document_lock:
            self._document.setdefault("sampler_errors", []).append(
                {"type": type(error).__name__, "message": str(error)}
            )

    def _refresh_monitors(self, nodes: list[dict[str, object]]) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        monitor_class = ray.remote(num_cpus=0)(_NodeGpuMonitor)
        for node in nodes:
            resources = node.get("Resources", {})
            node_id = str(node.get("NodeID", ""))
            if (
                not node_id
                or not node.get("Alive")
                or _gpu_count(resources) <= 0
                or node_id in self._monitors
            ):
                continue
            self._monitors[node_id] = monitor_class.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(node_id, self._require_one_l4)

    def _gpu_samples(self) -> list[dict[str, object]]:
        import ray

        pending = {
            node_id: monitor.sample.remote()
            for node_id, monitor in self._monitors.items()
        }
        if not pending:
            return []
        ready, _ = ray.wait(
            list(pending.values()),
            num_returns=len(pending),
            timeout=min(2.0, self._interval_seconds),
        )
        ready_ids = set(ready)
        results = []
        for node_id, reference in pending.items():
            if reference not in ready_ids:
                results.append({"node_id": node_id, "error": "NVML sample timeout"})
                continue
            try:
                results.append(ray.get(reference))
            except Exception as error:
                results.append(
                    {
                        "node_id": node_id,
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
        return results

    def sample(self, *, progress_override: dict[str, object] | None = None) -> bool:
        if self._stopped or not self._sample_lock.acquire(blocking=False):
            return False
        try:
            if self._stopped:
                return False
            import ray

            cluster = _float_resources(ray.cluster_resources())
            available = _float_resources(ray.available_resources())
            nodes = list(ray.nodes())
            alive_nodes = sum(1 for node in nodes if node.get("Alive"))
            self._refresh_monitors(nodes)
            gpu_telemetry = self._gpu_samples()
            if progress_override is None:
                try:
                    progress = ray.get(
                        self._progress.snapshot.remote(),
                        timeout=min(2.0, self._interval_seconds),
                    )
                except Exception as progress_error:
                    progress = {
                        "error": {
                            "type": type(progress_error).__name__,
                            "message": str(progress_error),
                        }
                    }
            else:
                progress = progress_override
            try:
                ray_state = _ray_state_snapshot()
            except Exception as state_error:
                ray_state = {
                    "error": {
                        "type": type(state_error).__name__,
                        "message": str(state_error),
                    }
                }
            sample = {
                "elapsed_s": max(0.0, time.monotonic() - self._origin_monotonic_s),
                "origin_epoch_s": self._origin_epoch_s,
                "origin_monotonic_s": self._origin_monotonic_s,
                "origin_boot_id": self._origin_boot_id,
                "observed_at": _utc_now(),
                "alive_nodes": alive_nodes,
                "cluster": cluster,
                "available": available,
                "progress": progress,
                "gpu_telemetry": gpu_telemetry,
                "ray_state": ray_state,
            }
            with self._document_lock:
                self._document.setdefault("resource_samples", []).append(sample)
                _atomic_json(self._result_path, self._document)
            return True
        except Exception as error:
            self._append_error(error)
            return False
        finally:
            self._sample_lock.release()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.sample()


def _integrate_resource_seconds(
    samples: list[dict[str, object]], resource: str, section: str
) -> float:
    total = 0.0
    for left, right in zip(samples, samples[1:]):
        duration = float(right["elapsed_s"]) - float(left["elapsed_s"])
        left_value = float(dict(left[section]).get(resource, 0.0))
        right_value = float(dict(right[section]).get(resource, 0.0))
        total += duration * (left_value + right_value) / 2.0
    return total


def _physical_gpu_metrics(samples: list[dict[str, object]]) -> dict[str, object]:
    utilization = []
    memory = []
    device_ids = set()
    cupy_probed_nodes = set()
    l4_observations = []
    for sample in samples:
        for node in sample.get("gpu_telemetry", []):
            if not isinstance(node, dict):
                continue
            if isinstance(node.get("startup_probe"), dict):
                cupy_probed_nodes.add(str(node.get("node_id")))
            for device in node.get("devices", []):
                if not isinstance(device, dict):
                    continue
                utilization.append(float(device["gpu_utilization_pct"]))
                memory.append(int(device["memory_used_bytes"]))
                l4_observations.append(bool(device.get("is_nvidia_l4")))
                device_ids.add(f"{node.get('node_id')}:{device.get('uuid')}")
    if not utilization:
        return {"sampled_devices": 0, "samples": 0}
    return {
        "sampled_devices": len(device_ids),
        "cupy_probed_nodes": len(cupy_probed_nodes),
        "samples": len(utilization),
        "mean_gpu_utilization_pct": sum(utilization) / len(utilization),
        "peak_gpu_utilization_pct": max(utilization),
        "peak_memory_used_bytes": max(memory),
        "all_observed_devices_are_nvidia_l4": all(l4_observations),
    }


def _stage_progress_metrics(samples: list[dict[str, object]]) -> dict[str, object]:
    stages = dict(samples[-1].get("progress", {})).get("stages", {})
    if not isinstance(stages, dict):
        return {"stages": [], "handoffs": []}
    ordered = sorted(
        (
            {"stage": str(name), **dict(value)}
            for name, value in stages.items()
            if isinstance(value, dict) and "first_progress_s" in value
        ),
        key=lambda value: (float(value["first_progress_s"]), str(value["stage"])),
    )
    handoffs = [
        {
            "from_stage": left["stage"],
            "to_stage": right["stage"],
            "from_last_progress_s": left.get("last_progress_s"),
            "to_first_progress_s": right["first_progress_s"],
            "gap_s": float(right["first_progress_s"])
            - float(left.get("last_progress_s", left["first_progress_s"])),
        }
        for left, right in zip(ordered, ordered[1:])
    ]
    return {"stages": ordered, "handoffs": handoffs}


def _stage_classification(stage: str) -> tuple[int, str]:
    if stage == "gpu-map-add-key":
        return 0, "upstream-map-batches"
    if stage == "gpu-map-groups-1":
        return 1, "post-shuffle-map-groups"
    if stage == "gpu-map-groups-2":
        return 2, "post-shuffle-map-groups"
    if stage == "gpu-map-final":
        return 3, "post-shuffle-map-batches"
    match = re.fullmatch(r"gpu-map-(\d+)", stage)
    if match:
        return int(match.group(1)) - 1, "sequential-map-batches"
    return 0, "unclassified"


def _stage_operator(stage: str) -> str:
    if stage.startswith("gpu-map-groups-"):
        return "SumGroup/map_groups"
    if stage == "gpu-map-final":
        return "Identity/map_batches"
    return "map_batches"


def _stage_actor_class_fragment(stage: str) -> str:
    if stage.startswith("gpu-map-groups-"):
        return "SumGroup"
    if stage == "gpu-map-final":
        return "Identity"
    return ""


def _premature_downstream_metrics(
    samples: list[dict[str, object]],
    *,
    target_job_id: str | None = None,
) -> dict[str, object]:
    """Attribute GPUs held by post-shuffle actors before usable input arrives."""

    empty = {
        "measurement_complete": False,
        "measurement_issue": "progress actor registry or first-input stages unavailable",
        "gcs_corroboration_complete": False,
        "gcs_corroboration_issue": "progress actor registry is unavailable",
        "gcs_corroborated_actor_count": 0,
        "gcs_expected_actor_count": 0,
        "gcs_state_samples_valid": False,
        "common_clock_verified": False,
        "clock_basis": "unverified",
        "telemetry_quiescence": {"complete": False},
        "premature_downstream_actor_count": 0,
        "premature_downstream_ownership_seconds": 0.0,
        "premature_downstream_gpu_seconds": 0.0,
        "mean_ready_to_stage_first_input_s": 0.0,
        "max_ready_to_stage_first_input_s": 0.0,
        "peak_premature_downstream_gpus": 0.0,
        "peak_premature_downstream_actors": 0,
        "premature_downstream_gpu_seconds_during_earlier_stage": 0.0,
        "peak_premature_downstream_gpus_during_earlier_stage": 0.0,
        "peak_premature_downstream_actors_during_earlier_stage": 0,
        "actors": [],
        "per_stage": [],
        "stage_classification": [],
    }
    if not samples:
        return empty
    latest_progress = samples[-1].get("progress")
    if not isinstance(latest_progress, dict):
        return empty
    actor_registry = latest_progress.get("actors")
    stages = latest_progress.get("stages")
    if not isinstance(actor_registry, dict) or not isinstance(stages, dict):
        return empty
    final_elapsed = float(samples[-1]["elapsed_s"])
    stage_first_input = {
        str(stage): float(value["first_input_s"])
        for stage, value in stages.items()
        if isinstance(value, dict)
        and isinstance(value.get("first_input_s"), (int, float))
    }
    classifications = {
        stage: _stage_classification(stage) for stage in stage_first_input
    }
    coverage = _progress_telemetry_coverage(latest_progress)
    progress_boot_id = latest_progress.get("origin_boot_id")
    progress_origin_monotonic_s = latest_progress.get("origin_monotonic_s")

    def verified_monotonic_event(
        value: dict[str, object],
        *,
        elapsed_key: str,
        monotonic_key: str,
        boot_key: str,
        basis_key: str,
    ) -> bool:
        elapsed = value.get(elapsed_key)
        monotonic = value.get(monotonic_key)
        return (
            isinstance(progress_origin_monotonic_s, (int, float))
            and isinstance(elapsed, (int, float))
            and isinstance(monotonic, (int, float))
            and math.isfinite(float(elapsed))
            and math.isfinite(float(monotonic))
            and float(monotonic) >= float(progress_origin_monotonic_s)
            and abs(
                float(elapsed) - (float(monotonic) - float(progress_origin_monotonic_s))
            )
            < 1e-9
            and value.get(boot_key) == progress_boot_id
            and value.get(basis_key) == "single-boot-monotonic"
        )

    rendered_stage_states = [
        value
        for value in stages.values()
        if isinstance(value, dict)
        and isinstance(value.get("first_input_s"), (int, float))
    ]
    downstream_actor_states = [
        value
        for value in actor_registry.values()
        if isinstance(value, dict)
        and _stage_classification(str(value.get("stage", "")))[1]
        in {"post-shuffle-map-groups", "post-shuffle-map-batches"}
    ]
    common_clock_verified = (
        isinstance(progress_boot_id, str)
        and bool(progress_boot_id)
        and isinstance(progress_origin_monotonic_s, (int, float))
        and math.isfinite(float(progress_origin_monotonic_s))
        and all(sample.get("origin_boot_id") == progress_boot_id for sample in samples)
        and all(
            isinstance(sample.get("origin_monotonic_s"), (int, float))
            and math.isfinite(float(sample["origin_monotonic_s"]))
            and abs(
                float(sample["origin_monotonic_s"]) - float(progress_origin_monotonic_s)
            )
            < 1e-9
            for sample in samples
        )
        and all(
            verified_monotonic_event(
                value,
                elapsed_key="ready_s",
                monotonic_key="ready_monotonic_s",
                boot_key="ready_boot_id",
                basis_key="ready_clock_basis",
            )
            for value in downstream_actor_states
        )
        and all(
            verified_monotonic_event(
                value,
                elapsed_key="first_input_s",
                monotonic_key="first_input_monotonic_s",
                boot_key="first_input_boot_id",
                basis_key="first_input_clock_basis",
            )
            and verified_monotonic_event(
                value,
                elapsed_key="first_progress_s",
                monotonic_key="first_progress_monotonic_s",
                boot_key="first_progress_boot_id",
                basis_key="first_progress_clock_basis",
            )
            for value in rendered_stage_states
        )
    )
    quiescence = latest_progress.get("telemetry_quiescence")
    quiescence_complete = (
        isinstance(quiescence, dict) and quiescence.get("complete") is True
    )
    premature = []
    downstream_roles = {"post-shuffle-map-groups", "post-shuffle-map-batches"}
    for actor_id, value in actor_registry.items():
        if not isinstance(value, dict):
            continue
        stage = str(value.get("stage", ""))
        order, role = _stage_classification(stage)
        if role not in downstream_roles:
            continue
        ready_s = value.get("ready_s")
        if not isinstance(ready_s, (int, float)):
            continue
        stage_input_s = stage_first_input.get(stage, final_elapsed)
        ready_s = float(ready_s)
        if ready_s >= stage_input_s:
            continue
        actor_first_input_s = value.get("first_input_s")
        actor_first_input = (
            float(actor_first_input_s)
            if isinstance(actor_first_input_s, (int, float))
            else None
        )
        gpus = float(value.get("gpus", 1.0))
        premature.append(
            {
                "actor_id": str(actor_id),
                "stage": stage,
                "stage_order": order,
                "stage_role": role,
                "operator": _stage_operator(stage),
                "gpus": gpus,
                "ready_s": ready_s,
                "stage_first_input_s": stage_input_s,
                "actor_first_input_s": actor_first_input,
                "ready_to_stage_first_input_s": stage_input_s - ready_s,
                "ready_to_actor_first_input_s": (
                    actor_first_input - ready_s
                    if actor_first_input is not None
                    else None
                ),
                "received_usable_input": actor_first_input is not None,
            }
        )

    state_samples_valid = all(
        isinstance(sample.get("ray_state"), dict)
        and sample["ray_state"].get("error") is None
        and sample["ray_state"].get("possibly_truncated") is False
        for sample in samples
    )
    gcs_issues = []
    if not target_job_id:
        gcs_issues.append("target Ray job ID is unavailable")
    if not state_samples_valid:
        gcs_issues.append("one or more GCS state samples are invalid or truncated")
    for actor in premature:
        actor_id = str(actor["actor_id"])
        expected_gpus = float(actor["gpus"])
        expected_class = _stage_actor_class_fragment(str(actor["stage"]))
        alive_observations = 0
        signature: tuple[str, str, float] | None = None
        for sample in samples:
            state = sample.get("ray_state")
            records = state.get("actors", []) if isinstance(state, dict) else []
            matching = [
                record
                for record in records
                if isinstance(record, dict) and str(record.get("actor_id")) == actor_id
            ]
            if len(matching) > 1:
                gcs_issues.append(f"{actor_id}: duplicate GCS actor records")
            for record in matching:
                gpus = _gpu_count(record.get("required_resources"))
                restarts = record.get("num_restarts")
                class_name = str(record.get("class_name", ""))
                job_id = str(record.get("job_id", ""))
                valid = True
                if (
                    not isinstance(restarts, int)
                    or isinstance(restarts, bool)
                    or restarts != 0
                ):
                    gcs_issues.append(f"{actor_id}: nonzero or missing restart count")
                    valid = False
                if abs(gpus - expected_gpus) > 1e-9:
                    gcs_issues.append(f"{actor_id}: conflicting GPU resources")
                    valid = False
                if expected_class not in class_name:
                    gcs_issues.append(f"{actor_id}: conflicting operator class")
                    valid = False
                if not target_job_id or job_id != target_job_id:
                    gcs_issues.append(f"{actor_id}: conflicting Ray job")
                    valid = False
                current_signature = (job_id, class_name, gpus)
                if signature is None:
                    signature = current_signature
                elif signature != current_signature:
                    gcs_issues.append(f"{actor_id}: conflicting GCS signatures")
                    valid = False
                elapsed = float(sample["elapsed_s"])
                in_interval = (
                    float(actor["ready_s"])
                    <= elapsed
                    < float(actor["stage_first_input_s"])
                )
                if (
                    valid
                    and str(record.get("state", "")).upper() == "ALIVE"
                    and in_interval
                ):
                    alive_observations += 1
        actor["gcs_alive_observation_count"] = alive_observations
        actor["gcs_corroborated"] = alive_observations > 0
        if alive_observations < 1:
            gcs_issues.append(f"{actor_id}: no in-window GCS ALIVE observation")
    unique_gcs_issues = sorted(set(gcs_issues))
    corroborated_count = sum(
        1 for actor in premature if actor.get("gcs_corroborated") is True
    )
    gcs_corroboration_complete = (
        bool(target_job_id)
        and not unique_gcs_issues
        and corroborated_count == len(premature)
    )
    base_telemetry_complete = (
        coverage.get("complete") is True
        and quiescence_complete
        and common_clock_verified
        and state_samples_valid
    )
    measurement_complete = base_telemetry_complete and gcs_corroboration_complete

    earlier_stage_overlap_samples = []
    for sample in samples:
        elapsed = float(sample["elapsed_s"])
        started_orders = [
            _stage_classification(stage)[0]
            for stage, first_input in stage_first_input.items()
            if first_input <= elapsed
        ]
        active_order = max(started_orders, default=0)
        state = sample.get("ray_state")
        actor_records = state.get("actors", []) if isinstance(state, dict) else []
        owners = {
            str(record.get("actor_id")): _gpu_count(record.get("required_resources"))
            for record in actor_records
            if isinstance(record, dict) and _running_state(record.get("state"))
        }
        premature_owned = [
            actor
            for actor in premature
            if actor["ready_s"] <= elapsed < actor["stage_first_input_s"]
            and actor["actor_id"] in owners
        ]
        earlier_stage_overlap = [
            actor for actor in premature_owned if actor["stage_order"] > active_order
        ]
        earlier_stage_overlap_samples.append(
            {
                "elapsed_s": elapsed,
                "premature_actors": len(premature_owned),
                "premature_gpus": sum(
                    owners[str(actor["actor_id"])] for actor in premature_owned
                ),
                "actors": len(earlier_stage_overlap),
                "gpus": sum(
                    owners[str(actor["actor_id"])] for actor in earlier_stage_overlap
                ),
            }
        )
    earlier_stage_overlap_gpu_seconds = 0.0
    for left, right in zip(
        earlier_stage_overlap_samples, earlier_stage_overlap_samples[1:]
    ):
        duration = float(right["elapsed_s"]) - float(left["elapsed_s"])
        earlier_stage_overlap_gpu_seconds += (
            duration * (float(left["gpus"]) + float(right["gpus"])) / 2.0
        )
    intervals = [float(actor["ready_to_stage_first_input_s"]) for actor in premature]
    interval_boundaries = sorted(
        {
            float(value)
            for actor in premature
            for value in (actor["ready_s"], actor["stage_first_input_s"])
        }
    )
    exact_peaks = [
        {
            "actors": sum(
                1
                for actor in premature
                if float(actor["ready_s"])
                <= boundary
                < float(actor["stage_first_input_s"])
            ),
            "gpus": sum(
                float(actor["gpus"])
                for actor in premature
                if float(actor["ready_s"])
                <= boundary
                < float(actor["stage_first_input_s"])
            ),
        }
        for boundary in interval_boundaries
    ]
    per_stage = []
    for stage in sorted({str(actor["stage"]) for actor in premature}):
        actors = [actor for actor in premature if actor["stage"] == stage]
        stage_intervals = [
            float(actor["ready_to_stage_first_input_s"]) for actor in actors
        ]
        per_stage.append(
            {
                "stage": stage,
                "stage_role": actors[0]["stage_role"],
                "operator": actors[0]["operator"],
                "actor_count": len(actors),
                "ownership_seconds": sum(stage_intervals),
                "gpu_seconds": sum(
                    interval * float(actor["gpus"])
                    for interval, actor in zip(stage_intervals, actors)
                ),
                "mean_ready_to_stage_first_input_s": statistics.fmean(stage_intervals),
                "max_ready_to_stage_first_input_s": max(stage_intervals),
            }
        )
    return {
        "measurement_complete": measurement_complete,
        "measurement_issue": (
            None
            if measurement_complete
            else (
                coverage.get("issue")
                if coverage.get("complete") is not True
                else (
                    "telemetry did not become quiescent before finalization"
                    if not quiescence_complete
                    else (
                        "events do not share one verified boot/monotonic clock"
                        if not common_clock_verified
                        else "GCS lifetime reservation corroboration is incomplete"
                    )
                )
            )
        ),
        "telemetry_coverage": coverage,
        "telemetry_quiescence": quiescence,
        "common_clock_verified": common_clock_verified,
        "clock_basis": (
            "single-boot-monotonic" if common_clock_verified else "unverified"
        ),
        "gcs_corroboration_complete": gcs_corroboration_complete,
        "gcs_corroboration_issue": (
            None if not unique_gcs_issues else unique_gcs_issues
        ),
        "gcs_corroborated_actor_count": corroborated_count,
        "gcs_expected_actor_count": len(premature),
        "gcs_state_samples_valid": state_samples_valid,
        "reservation_definition": (
            "measured lower bound from actor constructor-ready to stage first "
            "usable input for a lifetime-resource reservation, corroborated by "
            "in-window GCS ALIVE GPU ownership"
        ),
        "zero_waste_explanation": (
            "no downstream actor was observed constructor-ready before its "
            "stage's first usable input under this lower-bound measure"
            if measurement_complete and not premature
            else None
        ),
        "premature_downstream_actor_count": len(premature),
        "premature_downstream_ownership_seconds": sum(intervals),
        "premature_downstream_gpu_seconds": sum(
            interval * float(actor["gpus"])
            for interval, actor in zip(intervals, premature)
        ),
        "mean_ready_to_stage_first_input_s": (
            statistics.fmean(intervals) if intervals else 0.0
        ),
        "max_ready_to_stage_first_input_s": max(intervals, default=0.0),
        "peak_premature_downstream_gpus": max(
            (float(item["gpus"]) for item in exact_peaks),
            default=0.0,
        ),
        "peak_premature_downstream_actors": max(
            (int(item["actors"]) for item in exact_peaks),
            default=0,
        ),
        "sampled_peak_premature_gpus_joined_to_alive_state": max(
            (float(item["premature_gpus"]) for item in earlier_stage_overlap_samples),
            default=0.0,
        ),
        "premature_downstream_gpu_seconds_during_earlier_stage": (
            earlier_stage_overlap_gpu_seconds
        ),
        "peak_premature_downstream_gpus_during_earlier_stage": max(
            (float(item["gpus"]) for item in earlier_stage_overlap_samples),
            default=0.0,
        ),
        "peak_premature_downstream_actors_during_earlier_stage": max(
            (int(item["actors"]) for item in earlier_stage_overlap_samples),
            default=0,
        ),
        "actors": premature,
        "per_stage": per_stage,
        "stage_classification": [
            {
                "stage": stage,
                "order": order,
                "role": role,
                "operator": _stage_operator(stage),
            }
            for stage, (order, role) in sorted(
                classifications.items(), key=lambda item: item[1]
            )
        ],
    }


def _resource_metrics(
    samples: list[dict[str, object]],
    max_gpus: int,
    *,
    target_job_id: str | None = None,
) -> dict:
    if not samples:
        return {}
    cluster_gpus = [float(dict(item["cluster"]).get("GPU", 0.0)) for item in samples]
    available_gpus = [
        float(dict(item["available"]).get("GPU", 0.0)) for item in samples
    ]
    first_gpu = next(
        (
            float(item["elapsed_s"])
            for item, value in zip(samples, cluster_gpus)
            if value
        ),
        None,
    )
    peak_gpu = max(cluster_gpus)
    time_to_peak = next(
        float(item["elapsed_s"])
        for item, value in zip(samples, cluster_gpus)
        if value == peak_gpu
    )
    time_to_requested_peak = next(
        (
            float(item["elapsed_s"])
            for item, value in zip(samples, cluster_gpus)
            if value >= max_gpus
        ),
        None,
    )
    first_gpu_sample = next(
        (item for item, value in zip(samples, cluster_gpus) if value), None
    )
    observed_peak_sample = next(
        item for item, value in zip(samples, cluster_gpus) if value == peak_gpu
    )
    requested_peak_sample = next(
        (item for item, value in zip(samples, cluster_gpus) if value >= max_gpus),
        None,
    )
    stage_progress = _stage_progress_metrics(samples)
    premature_downstream = _premature_downstream_metrics(
        samples, target_job_id=target_job_id
    )
    first_progress = next(iter(stage_progress["stages"]), None)
    cluster_gpu_seconds = _integrate_resource_seconds(samples, "GPU", "cluster")
    available_gpu_seconds = _integrate_resource_seconds(samples, "GPU", "available")
    premature_downstream["premature_gpu_seconds_fraction_of_cluster"] = (
        float(premature_downstream["premature_downstream_gpu_seconds"])
        / cluster_gpu_seconds
        if cluster_gpu_seconds > 0
        else None
    )
    for stage in premature_downstream["per_stage"]:
        stage["cluster_gpu_seconds_fraction"] = (
            float(stage["gpu_seconds"]) / cluster_gpu_seconds
            if cluster_gpu_seconds > 0
            else None
        )
    closed_wait = _classify_closed_wait(samples, max_gpus)
    return {
        "first_gpu_visible_s": first_gpu,
        "first_gpu_visible_at": (
            first_gpu_sample.get("observed_at") if first_gpu_sample else None
        ),
        "peak_cluster_gpus": peak_gpu,
        "time_to_observed_peak_gpus_s": time_to_peak,
        "observed_peak_gpus_visible_at": observed_peak_sample.get("observed_at"),
        "time_to_topology_max_gpus_s": time_to_requested_peak,
        "topology_max_gpus_visible_at": (
            requested_peak_sample.get("observed_at") if requested_peak_sample else None
        ),
        "cluster_gpu_seconds": cluster_gpu_seconds,
        "available_gpu_seconds": available_gpu_seconds,
        "owned_gpu_seconds": max(0.0, cluster_gpu_seconds - available_gpu_seconds),
        "min_available_gpus": min(available_gpus),
        "max_alive_nodes": max(int(item["alive_nodes"]) for item in samples),
        "physical_gpu": _physical_gpu_metrics(samples),
        "stage_progress": stage_progress,
        "premature_downstream_ownership": premature_downstream,
        "first_useful_progress_at": (
            first_progress.get("first_progress_at") if first_progress else None
        ),
        "structural_closed_wait": closed_wait["is_structural_closed_wait"],
        "closed_wait": closed_wait,
    }


def _expected_sums(rows: int, groups: int) -> dict[int, int]:
    expected = {}
    for key in range(min(rows, groups)):
        count = ((rows - 1 - key) // groups) + 1
        expected[key] = count * (2 * key + (count - 1) * groups) // 2
    return expected


def _output_digest(rows: list[dict[str, object]]) -> str:
    normalized = sorted(
        ({"key": int(row["key"]), "id": int(row["id"])} for row in rows),
        key=lambda row: row["key"],
    )
    payload = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _schema_document(dataset: object) -> dict[str, object]:
    schema = dataset.schema()
    if schema is None:
        raise AssertionError("materialized dataset has no schema")
    fields = [
        {"name": str(name), "type": str(field_type)}
        for name, field_type in zip(schema.names, schema.types)
    ]
    payload = json.dumps(fields, separators=(",", ":"), sort_keys=True).encode()
    return {"fields": fields, "sha256": hashlib.sha256(payload).hexdigest()}


def _empty_fingerprint() -> dict[str, int | None]:
    return {
        "count": 0,
        "minimum": None,
        "maximum": None,
        "sum_u64": 0,
        "sum_mix_a_u64": 0,
        "sum_mix_b_u64": 0,
        "xor_mix_u64": 0,
    }


def _update_fingerprint(summary: dict[str, int | None], values: object) -> None:
    import numpy as np

    array = np.asarray(values, dtype=np.uint64).reshape(-1)
    if not len(array):
        return

    def splitmix(data, salt):
        mixed = data + np.uint64(salt)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return mixed ^ (mixed >> np.uint64(31))

    mixed_a = splitmix(array, 0x9E3779B97F4A7C15)
    mixed_b = splitmix(array, 0xD1B54A32D192ED03)
    summary["count"] = int(summary["count"]) + len(array)
    minimum = int(array.min())
    maximum = int(array.max())
    summary["minimum"] = (
        minimum if summary["minimum"] is None else min(int(summary["minimum"]), minimum)
    )
    summary["maximum"] = (
        maximum if summary["maximum"] is None else max(int(summary["maximum"]), maximum)
    )
    summary["sum_u64"] = (
        int(summary["sum_u64"]) + int(array.sum(dtype=np.uint64))
    ) & _UINT64_MASK
    summary["sum_mix_a_u64"] = (
        int(summary["sum_mix_a_u64"]) + int(mixed_a.sum(dtype=np.uint64))
    ) & _UINT64_MASK
    summary["sum_mix_b_u64"] = (
        int(summary["sum_mix_b_u64"]) + int(mixed_b.sum(dtype=np.uint64))
    ) & _UINT64_MASK
    summary["xor_mix_u64"] = int(summary["xor_mix_u64"]) ^ int(
        np.bitwise_xor.reduce(mixed_a)
    )


def _fingerprint_digest(summary: dict[str, int | None]) -> str:
    payload = json.dumps(summary, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _actual_row_fingerprint(dataset: object) -> dict[str, object]:
    summary = _empty_fingerprint()
    for batch in dataset.iter_batches(batch_format="numpy", batch_size=1_000_000):
        if set(batch) != {"id"}:
            raise AssertionError(
                f"row workload produced unexpected columns: {set(batch)}"
            )
        _update_fingerprint(summary, batch["id"])
    return {"summary": summary, "sha256": _fingerprint_digest(summary)}


def _expected_row_fingerprint(ranges: list[tuple[int, int]]) -> dict[str, object]:
    import numpy as np

    summary = _empty_fingerprint()
    chunk_size = 1_000_000
    for start, count in ranges:
        for offset in range(0, count, chunk_size):
            length = min(chunk_size, count - offset)
            values = np.arange(start + offset, start + offset + length, dtype=np.uint64)
            _update_fingerprint(summary, values)
    return {"summary": summary, "sha256": _fingerprint_digest(summary)}


def _shuffle_rank(value: str) -> int | None:
    if value == "default":
        return None
    try:
        rank = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer or 'default'"
        ) from error
    if rank < 1:
        raise argparse.ArgumentTypeError("must be a positive integer or 'default'")
    return rank


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, choices=("stock", "pg-only", "minimal", "prototype")
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=(
            "incident",
            "actor-only",
            "map-heavy",
            "shuffle-heavy",
            "forced-spill",
            "fan-in",
            "failure-cleanup",
        ),
    )
    parser.add_argument("--topology", required=True)
    parser.add_argument("--shuffle-ranks", required=True, type=_shuffle_rank)
    parser.add_argument(
        "--map-actors",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--map-actors-min", type=int)
    parser.add_argument("--map-actors-max", type=int)
    parser.add_argument("--max-gpus", required=True, type=int)
    parser.add_argument("--rows", type=int, default=16_000_000)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--gpu-map-work-iterations", type=int, default=0)
    parser.add_argument("--materialize-boundaries", action="store_true")
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--require-one-l4-per-gpu-node", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.map_actors is not None:
        if args.map_actors_min is not None or args.map_actors_max is not None:
            parser.error("--map-actors cannot be combined with map min/max")
        args.map_actors_min = args.map_actors
        args.map_actors_max = args.map_actors
    elif args.map_actors_min is None or args.map_actors_max is None:
        parser.error("both --map-actors-min and --map-actors-max are required")
    for name in (
        "map_actors_min",
        "map_actors_max",
        "max_gpus",
        "rows",
        "blocks",
        "groups",
        "batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.map_actors_min > args.map_actors_max:
        parser.error("map actor minimum cannot exceed its maximum")
    if args.shuffle_ranks is not None and args.shuffle_ranks > args.max_gpus:
        parser.error("shuffle ranks cannot exceed topology GPU capacity")
    if args.sample_interval_seconds <= 0:
        parser.error("--sample-interval-seconds must be positive")
    if args.gpu_map_work_iterations < 0:
        parser.error("--gpu-map-work-iterations cannot be negative")
    return args


def _configure_context(args: argparse.Namespace):
    import ray.data
    from ray.data._internal.execution.interfaces import ExecutionResources
    from ray.data.context import ShuffleStrategy

    context = ray.data.DataContext.get_current()
    context.shuffle_strategy = ShuffleStrategy.GPU_SHUFFLE
    if args.shuffle_ranks is not None:
        context.gpu_shuffle_num_actors = args.shuffle_ranks
    context.execution_options.resource_limits = ExecutionResources.for_limits(
        gpu=args.max_gpus
    )
    admission_enabled = args.arm in {"minimal", "prototype"}
    if args.arm != "stock":
        if not hasattr(context, "_enable_resource_admission_control"):
            raise RuntimeError("candidate arm lacks the admission rollback field")
        context._enable_resource_admission_control = admission_enabled
    return context


def _actor_pool_strategy(args: argparse.Namespace):
    from ray.data import ActorPoolStrategy

    return (
        ActorPoolStrategy(size=args.map_actors_min)
        if args.map_actors_min == args.map_actors_max
        else ActorPoolStrategy(
            min_size=args.map_actors_min,
            max_size=args.map_actors_max,
        )
    )


def _expected_downstream_actor_counts(
    args: argparse.Namespace,
) -> dict[str, int]:
    """Return the fixed GPU pools required for causal downstream evidence."""

    if args.map_actors_min != args.map_actors_max:
        return {}
    if args.workload not in {
        "incident",
        "shuffle-heavy",
        "forced-spill",
        "failure-cleanup",
    }:
        return {}
    count = int(args.map_actors_min)
    expected = {
        "gpu-map-groups-1": count,
        "gpu-map-final": count,
    }
    if args.workload == "shuffle-heavy":
        expected["gpu-map-groups-2"] = count
    return expected


def _build_dataset(args: argparse.Namespace, progress: object):
    import ray.data

    strategy = _actor_pool_strategy(args)
    materialization_phases: list[dict[str, object]] = []

    def materialize_boundary(dataset, stage: str):
        if not args.materialize_boundaries:
            return dataset
        started = time.monotonic()
        materialized = dataset.materialize()
        materialization_phases.append(
            {"stage": stage, "materialize_s": time.monotonic() - started}
        )
        return materialized

    def gpu_map(dataset, udf, stage: str, *, constructor_args=()):
        return dataset.map_batches(
            udf,
            fn_constructor_args=(*constructor_args, progress, stage),
            batch_format="cudf",
            batch_size=args.batch_size,
            compute=strategy,
            num_cpus=0,
            num_gpus=1,
        )

    source = ray.data.range(args.rows, override_num_blocks=args.blocks)
    if args.workload == "actor-only":
        dataset = gpu_map(source, AddOne, "gpu-map-1")
        dataset = gpu_map(dataset, AddOne, "gpu-map-2")
        return (
            gpu_map(dataset, Identity, "gpu-map-3"),
            "row-count",
            materialization_phases,
        )
    if args.workload == "map-heavy":
        dataset = source
        for index in range(6):
            dataset = gpu_map(dataset, AddOne, f"gpu-map-{index + 1}")
        return dataset, "row-count", materialization_phases
    if args.workload == "fan-in":
        left_rows = args.rows // 2
        right_rows = args.rows - left_rows
        left = gpu_map(
            ray.data.range(left_rows, override_num_blocks=max(1, args.blocks // 2)),
            AddOne,
            "gpu-map-left",
        )
        right = gpu_map(
            ray.data.range(right_rows, override_num_blocks=max(1, args.blocks // 2)),
            AddOne,
            "gpu-map-right",
        )
        return (
            gpu_map(left.union(right), Identity, "gpu-map-fan-in"),
            "row-count",
            materialization_phases,
        )

    dataset = gpu_map(
        source,
        AddKey,
        "gpu-map-add-key",
        constructor_args=(args.groups, args.gpu_map_work_iterations),
    )
    dataset = materialize_boundary(dataset, "upstream-gpu-map")
    groupby_kwargs = (
        {} if args.shuffle_ranks is None else {"num_partitions": args.shuffle_ranks}
    )
    dataset = dataset.groupby("key", **groupby_kwargs).map_groups(
        SumGroup,
        fn_constructor_args=(progress, "gpu-map-groups-1"),
        batch_format="cudf",
        compute=strategy,
        num_cpus=0,
        num_gpus=1,
    )
    dataset = materialize_boundary(dataset, "shuffle-and-map-groups")
    if args.workload == "shuffle-heavy":
        dataset = dataset.groupby("key", **groupby_kwargs).map_groups(
            SumGroup,
            fn_constructor_args=(progress, "gpu-map-groups-2"),
            batch_format="cudf",
            compute=strategy,
            num_cpus=0,
            num_gpus=1,
        )
    final_udf = RaiseInjected if args.workload == "failure-cleanup" else Identity
    return (
        gpu_map(dataset, final_udf, "gpu-map-final"),
        "group-sum",
        materialization_phases,
    )


def _expected_row_ranges(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.workload == "actor-only":
        return [(2, args.rows)]
    if args.workload == "map-heavy":
        return [(6, args.rows)]
    if args.workload == "fan-in":
        left_rows = args.rows // 2
        right_rows = args.rows - left_rows
        return [(1, left_rows), (1, right_rows)]
    raise AssertionError(f"no row oracle for {args.workload}")


def _is_expected_injected_failure(
    workload: str, error: BaseException, error_traceback: str
) -> bool:
    return (
        workload == "failure-cleanup"
        and _EXPECTED_FAILURE_MARKER in f"{error}\n{error_traceback}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault(
        "RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL",
        "1" if args.arm in {"minimal", "prototype"} else "0",
    )
    document_lock = threading.RLock()
    document: dict[str, object] = {
        "schema_version": 2,
        "status": "starting",
        "started_at": _utc_now(),
        "arm": args.arm,
        "workload": args.workload,
        "topology": args.topology,
        "shuffle_ranks": (
            "default" if args.shuffle_ranks is None else args.shuffle_ranks
        ),
        "map_actors_per_stage": (
            args.map_actors_min if args.map_actors_min == args.map_actors_max else None
        ),
        "map_actor_pool": {
            "min_size": args.map_actors_min,
            "max_size": args.map_actors_max,
            "constructor": (
                "size"
                if args.map_actors_min == args.map_actors_max
                else "min_size,max_size"
            ),
        },
        "topology_max_gpus": args.max_gpus,
        "rows": args.rows,
        "blocks": args.blocks,
        "groups": args.groups,
        "batch_size": args.batch_size,
        "gpu_map_work_iterations": args.gpu_map_work_iterations,
        "materialize_boundaries": args.materialize_boundaries,
        "forced_spill_requires_cluster_object_store_cap": args.workload
        == "forced-spill",
        "injected_failure_stage": (
            "gpu-map-final" if args.workload == "failure-cleanup" else None
        ),
        "pid": os.getpid(),
        "resource_samples": [],
    }
    _write_document(args.result, document, document_lock)
    sampler = None
    ray_module = None
    workload_started = time.monotonic()
    termination_signal = None

    class TerminationRequested(BaseException):
        pass

    def terminate(signum: int, _frame: object) -> None:
        nonlocal termination_signal
        termination_signal = signum
        raise TerminationRequested()

    # Ray installs its own termination handlers while importing/initializing.
    # Capture the driver's original handlers now, then install ours only after
    # ray.init() so a bounded evidence timeout checkpoints telemetry instead of
    # being rewritten into Ray's generic SystemExit(1).
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }

    try:
        import ray

        ray_module = ray
        ray.init(address="auto")
        with document_lock:
            document["ray_connected_at"] = _utc_now()
            document["ray_connected_s"] = time.monotonic() - workload_started
        for signum in previous_handlers:
            signal.signal(signum, terminate)
        job_id = ray.get_runtime_context().get_job_id()
        document["ray_job_id"] = job_id if isinstance(job_id, str) else job_id.hex()
        document["ray_job_id_repr"] = str(job_id)
        _configure_context(args)
        telemetry_origin_epoch_s = time.time()
        telemetry_origin_monotonic_s = time.monotonic()
        telemetry_origin_boot_id = _read_boot_id()
        expected_actor_counts = _expected_downstream_actor_counts(args)
        with document_lock:
            document["telemetry_origin_epoch_s"] = telemetry_origin_epoch_s
            document["telemetry_origin_monotonic_s"] = telemetry_origin_monotonic_s
            document["telemetry_origin_boot_id"] = telemetry_origin_boot_id
            document["expected_downstream_actor_counts"] = expected_actor_counts
        progress = ray.remote(num_cpus=0)(_ProgressTracker).remote(
            telemetry_origin_epoch_s,
            expected_actor_counts,
            telemetry_origin_monotonic_s,
            telemetry_origin_boot_id,
        )
        sampler = ClusterSampler(
            result_path=args.result,
            interval_seconds=args.sample_interval_seconds,
            document=document,
            document_lock=document_lock,
            progress=progress,
            require_one_l4=args.require_one_l4_per_gpu_node,
            origin_epoch_s=telemetry_origin_epoch_s,
            origin_monotonic_s=telemetry_origin_monotonic_s,
            origin_boot_id=telemetry_origin_boot_id,
        )
        sampler.start()
        with document_lock:
            document["status"] = "building"
        build_started = time.monotonic()
        dataset, oracle, materialization_phases = _build_dataset(args, progress)
        demand_started_s = time.monotonic() - workload_started
        demand_started_at = _utc_now()
        with document_lock:
            document["build_s"] = time.monotonic() - build_started
            document["logical_dataset"] = repr(dataset)
            document["materialization_phases"] = materialization_phases
            document["status"] = "materializing"
            document["workload_demand_started_s"] = demand_started_s
            document["workload_demand_started_at"] = demand_started_at
            _atomic_json(args.result, document)

        materialize_started = time.monotonic()
        materialized = dataset.materialize()
        materialize_finished = time.monotonic()
        materialization_phases.append(
            {
                "stage": "terminal-gpu-map",
                "materialize_s": materialize_finished - materialize_started,
            }
        )
        completion_s = materialize_finished - workload_started
        with document_lock:
            document["materialize_s"] = materialize_finished - materialize_started
            document["pipeline_materialize_s"] = materialize_finished - build_started
            document["materialization_phases"] = materialization_phases
        if expected_actor_counts:
            final_progress, telemetry_drain = _drain_progress_telemetry(ray, progress)
        else:
            final_progress = ray.get(progress.snapshot.remote(), timeout=2.0)
            telemetry_drain = {
                "complete": True,
                "attempts": 1,
                "elapsed_s": 0.0,
                "timeout_s": 0.0,
                "coverage": {"complete": True, "not_applicable": True},
            }
        with document_lock:
            document["telemetry_drain"] = telemetry_drain
            document["final_progress"] = final_progress
        sampler.stop(final_progress=final_progress)
        samples = list(document["resource_samples"])
        peak_cluster_gpus = max(
            float(dict(sample["cluster"]).get("GPU", 0.0)) for sample in samples
        )
        physical_gpu = _physical_gpu_metrics(samples)
        if args.require_one_l4_per_gpu_node and (
            physical_gpu.get("sampled_devices", 0) != peak_cluster_gpus
            or physical_gpu.get("cupy_probed_nodes", 0) != peak_cluster_gpus
            or physical_gpu.get("all_observed_devices_are_nvidia_l4") is not True
        ):
            raise AssertionError(
                "GPU telemetry did not validate every observed Ray GPU node: "
                f"peak_cluster_gpus={peak_cluster_gpus}, physical={physical_gpu}"
            )
        validation_started = time.monotonic()
        schema = _schema_document(materialized)
        if oracle == "group-sum":
            output = materialized.take_all()
            actual = {int(row["key"]): int(row["id"]) for row in output}
            expected = _expected_sums(args.rows, args.groups)
            if actual != expected:
                raise AssertionError(
                    f"aggregate oracle mismatch: expected {expected}, got {actual}"
                )
            output_rows = len(output)
            output_digest = _output_digest(output)
            content_oracle = {"kind": "exact-group-sum", "sha256": output_digest}
        else:
            actual_fingerprint = _actual_row_fingerprint(materialized)
            expected_fingerprint = _expected_row_fingerprint(_expected_row_ranges(args))
            if actual_fingerprint != expected_fingerprint:
                raise AssertionError(
                    "row content oracle mismatch: "
                    f"expected {expected_fingerprint}, got {actual_fingerprint}"
                )
            output_rows = int(actual_fingerprint["summary"]["count"])
            output_digest = str(actual_fingerprint["sha256"])
            content_oracle = {
                "kind": "commutative-row-fingerprint",
                "actual": actual_fingerprint,
                "expected": expected_fingerprint,
            }
        summary = materialized.get_stats_summary()
        global_bytes_spilled = int(getattr(summary, "global_bytes_spilled", 0))
        global_bytes_restored = int(getattr(summary, "global_bytes_restored", 0))
        if args.workload == "forced-spill" and global_bytes_spilled <= 0:
            raise AssertionError(
                "forced-spill workload completed without observed object spilling; "
                "the cloud runner must impose a bounded object-store capacity"
            )
        validation_s = time.monotonic() - validation_started
        total_driver_s = time.monotonic() - workload_started
        with document_lock:
            document.update(
                status="success",
                finished_at=_utc_now(),
                elapsed_s=completion_s,
                total_driver_s=total_driver_s,
                validation_s=validation_s,
                output_rows=output_rows,
                output_schema=schema,
                output_schema_hash=schema["sha256"],
                output_digest=output_digest,
                content_oracle=content_oracle,
                input_rows_per_second=args.rows / completion_s,
                global_bytes_spilled=global_bytes_spilled,
                global_bytes_restored=global_bytes_restored,
                dataset_stats=materialized.stats(),
            )
        return 0
    except TerminationRequested:
        with document_lock:
            document.update(
                status="terminated",
                termination_signal=termination_signal,
                finished_at=_utc_now(),
                elapsed_s=time.monotonic() - workload_started,
            )
        return 124
    except BaseException as error:
        error_traceback = traceback.format_exc()
        expected_failure = _is_expected_injected_failure(
            args.workload, error, error_traceback
        )
        with document_lock:
            document.update(
                status="expected-failure" if expected_failure else "error",
                expected_injected_failure=expected_failure,
                expected_failure_marker=(
                    _EXPECTED_FAILURE_MARKER if expected_failure else None
                ),
                finished_at=_utc_now(),
                elapsed_s=time.monotonic() - workload_started,
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": error_traceback,
                },
            )
        # An injected failure is a successful cleanup experiment.  The runner
        # uses ray_job_id above to audit actors and placement groups afterward.
        return 0 if expected_failure else 1
    finally:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        if sampler is not None:
            if termination_signal is None:
                sampler.sample()
            sampler.stop()
        with document_lock:
            document["resource_metrics"] = _resource_metrics(
                list(document["resource_samples"]),
                args.max_gpus,
                target_job_id=(
                    str(document.get("ray_job_id"))
                    if document.get("ray_job_id")
                    else None
                ),
            )
            _atomic_json(args.result, document)
        if ray_module is not None:
            ray_module.shutdown()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
