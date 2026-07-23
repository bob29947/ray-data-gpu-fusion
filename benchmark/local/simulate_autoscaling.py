#!/usr/bin/env python3
"""Simulate Ray Data reacting to added logical GPU capacity on one host.

The executable path runs the candidate resource-admission implementation, not
a synthetic placement group.  A fixed two-actor GPU map starts with only one
logical-GPU raylet, exposes pending scheduler demand without making data-path
progress, and completes after a second logical-GPU raylet is added.  A warm
replay then runs on the already-expanded cluster.

All raylets share one physical host and the map UDF never calls CUDA.  This is
only scheduler/admission evidence; it is not EC2 autoscaling, multi-host, or
GPU-throughput evidence.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHM_ROOT = Path("/dev/shm/ray-admission")
CANDIDATE_MANIFEST = ROOT / "pins" / "pr-candidate.json"
LOGICAL_GPU_CEILING = 2
_GPU_EPSILON = 0.01


class _LogicalProgress:
    """Zero-resource telemetry actor used by the logical-only map UDF."""

    def __init__(self):
        self._phases: dict[str, dict[str, int | float]] = {}

    def record(self, phase: str, rows: int) -> None:
        state = self._phases.setdefault(
            phase, {"batches": 0, "rows": 0, "first_progress_wall_s": time.time()}
        )
        state["batches"] = int(state["batches"]) + 1
        state["rows"] = int(state["rows"]) + int(rows)

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        return {phase: dict(state) for phase, state in self._phases.items()}


class _LogicalIdentity:
    """CPU-only identity UDF whose actor reserves one *logical* Ray GPU."""

    def __init__(self, progress: object, phase: str):
        self._progress = progress
        self._phase = phase

    def __call__(self, batch):
        import ray

        # Synchronous telemetry makes "no progress before capacity" a strong
        # assertion.  The telemetry actor reserves no CPU or GPU.
        ray.get(self._progress.record.remote(self._phase, _batch_rows(batch)))
        return batch


def _batch_rows(batch: object) -> int:
    if isinstance(batch, Mapping):
        first_column = next(iter(batch.values()), ())
        return len(first_column)  # type: ignore[arg-type]
    return len(batch)  # type: ignore[arg-type]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_artifact(explicit_wheel: Path | None) -> dict[str, object]:
    manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    wheel_name = manifest.get("wheel")
    if explicit_wheel is None:
        if not isinstance(wheel_name, str) or not wheel_name.endswith(".whl"):
            raise RuntimeError("candidate manifest has no wheel filename")
        wheel = ROOT / "wheels" / "pr-candidate" / wheel_name
    else:
        wheel = explicit_wheel.expanduser().resolve()
    if not wheel.is_file():
        raise RuntimeError(f"candidate wheel does not exist: {wheel}")
    actual_sha256 = _sha256_file(wheel)
    expected_sha256 = manifest.get("wheel_sha256")
    if explicit_wheel is None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "candidate wheel SHA-256 does not match pins/pr-candidate.json"
        )
    matches_pin = actual_sha256 == expected_sha256
    return {
        "wheel": str(wheel),
        "wheel_sha256": actual_sha256,
        "source_commit": (
            manifest.get("local_commit") or manifest.get("commit")
            if matches_pin
            else None
        ),
        "base_commit": (
            manifest.get("base_commit") or manifest.get("commit")
            if matches_pin
            else None
        ),
        "matches_pinned_sha256": matches_pin,
    }


def _install_candidate_overlay(wheel: Path, parent: Path) -> tuple[Path, float]:
    parent.mkdir(parents=True, exist_ok=True)
    overlay = Path(tempfile.mkdtemp(prefix=".logical-ray-overlay-", dir=parent))
    started = time.monotonic()
    try:
        completed = subprocess.run(
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
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode:
            detail = (completed.stdout + completed.stderr)[-4_000:]
            raise RuntimeError(f"candidate wheel overlay install failed:\n{detail}")
    except BaseException:
        shutil.rmtree(overlay, ignore_errors=True)
        raise
    return overlay, time.monotonic() - started


def _node_processes(node: object) -> list[int]:
    result = []
    for entries in getattr(node, "all_processes", {}).values():
        for entry in entries:
            pid = getattr(getattr(entry, "process", None), "pid", None)
            if isinstance(pid, int):
                result.append(pid)
    return sorted(set(result))


def _active_gpu_actor_demand(
    actors: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    gpu_actors = []
    for actor in actors:
        resources = actor.get("required_resources")
        gpu = (
            float(resources.get("GPU", 0.0)) if isinstance(resources, Mapping) else 0.0
        )
        state = str(actor.get("state", "UNKNOWN")).upper()
        if gpu >= 1 - _GPU_EPSILON and state != "DEAD":
            gpu_actors.append(actor)
    states: dict[str, int] = {}
    for actor in gpu_actors:
        state = str(actor.get("state", "UNKNOWN"))
        states[state] = states.get(state, 0) + 1
    return {
        "active_gpu_actor_requests": len(gpu_actors),
        "pending_gpu_actor_requests": sum(
            count for state, count in states.items() if state.upper() != "ALIVE"
        ),
        "states": dict(sorted(states.items())),
        "actors": [
            {
                "actor_id": actor.get("actor_id"),
                "class_name": actor.get("class_name"),
                "state": actor.get("state"),
                "node_id": actor.get("node_id"),
                "required_resources": actor.get("required_resources"),
            }
            for actor in gpu_actors
        ],
    }


def _phase_batches(progress: Mapping[str, object], phase: str) -> int:
    state = progress.get(phase)
    return int(state.get("batches", 0)) if isinstance(state, Mapping) else 0


def _pending_without_progress(sample: Mapping[str, object], phase: str) -> bool:
    demand = sample.get("gpu_actor_demand")
    cluster = sample.get("cluster_resources")
    available = sample.get("available_resources")
    progress = sample.get("progress")
    return (
        isinstance(demand, Mapping)
        and int(demand.get("active_gpu_actor_requests", 0)) >= LOGICAL_GPU_CEILING
        and int(demand.get("pending_gpu_actor_requests", 0)) >= 1
        and isinstance(cluster, Mapping)
        and float(cluster.get("GPU", 0.0)) == 1.0
        and isinstance(available, Mapping)
        and float(available.get("GPU", 0.0)) <= _GPU_EPSILON
        and isinstance(progress, Mapping)
        and _phase_batches(progress, phase) == 0
    )


def _timing_metrics(marks: Mapping[str, float]) -> dict[str, float]:
    metrics = {
        "resource_request_to_node_visible_s": (
            marks["node_visible"] - marks["cold_request"]
        ),
        "pending_demand_to_node_visible_s": (
            marks["node_visible"] - marks["pending_demand"]
        ),
        "capacity_add_request_to_node_visible_s": (
            marks["node_visible"] - marks["capacity_add_request"]
        ),
        "node_visible_to_first_progress_s": (
            marks["cold_first_progress"] - marks["node_visible"]
        ),
        "node_visible_to_completion_s": (
            marks["cold_completion"] - marks["node_visible"]
        ),
        "cold_request_to_completion_s": (
            marks["cold_completion"] - marks["cold_request"]
        ),
        "warm_request_to_first_progress_s": (
            marks["warm_first_progress"] - marks["warm_request"]
        ),
        "warm_request_to_completion_s": (
            marks["warm_completion"] - marks["warm_request"]
        ),
    }
    if "both_gpu_actors_alive" in marks:
        metrics.update(
            node_visible_to_two_actor_ready_s=(
                marks["both_gpu_actors_alive"] - marks["node_visible"]
            ),
            two_actor_ready_to_first_progress_s=(
                marks["cold_first_progress"] - marks["both_gpu_actors_alive"]
            ),
        )
    return metrics


def _build_actor_map(args: argparse.Namespace, progress: object, phase: str):
    import ray.data
    from ray.data import ActorPoolStrategy

    strategy = ActorPoolStrategy(
        min_size=LOGICAL_GPU_CEILING, max_size=LOGICAL_GPU_CEILING
    )
    return ray.data.range(args.rows, override_num_blocks=args.blocks).map_batches(
        _LogicalIdentity,
        fn_constructor_args=(progress, phase),
        batch_format="numpy",
        batch_size=args.batch_size,
        compute=strategy,
        num_cpus=0,
        num_gpus=1,
    )


def _start_materialization(
    dataset: object,
) -> tuple[threading.Thread, threading.Event, dict]:
    finished = threading.Event()
    result: dict[str, object] = {}

    def materialize() -> None:
        try:
            materialized = dataset.materialize()
            result["rows"] = int(materialized.count())
            result["materialized"] = materialized
        except BaseException as error:
            result["error"] = error
        finally:
            result["finished_monotonic"] = time.monotonic()
            finished.set()

    thread = threading.Thread(target=materialize, daemon=True)
    thread.start()
    return thread, finished, result


def _raise_materialization_error(result: Mapping[str, object]) -> None:
    error = result.get("error")
    if isinstance(error, BaseException):
        raise RuntimeError("logical Ray Data materialization failed") from error


def _wait_for_owned_processes(
    pids: Sequence[int], timeout_s: float = 15.0
) -> list[int]:
    deadline = time.monotonic() + timeout_s
    while True:
        alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.1)


def _safe_remove_ray_tmp(path: Path) -> bool:
    resolved = path.resolve()
    if resolved.parent != SHM_ROOT.resolve() or not resolved.name.startswith("s"):
        raise RuntimeError(f"refusing to remove unexpected Ray temp path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    return not resolved.exists()


def run_simulation(args: argparse.Namespace) -> dict[str, object]:
    artifact = _candidate_artifact(args.candidate_wheel)
    SHM_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="s", dir=SHM_ROOT))
    # RAY_TMPDIR is a root under which Ray writes its own ``ray`` directory,
    # while Cluster.add_node also receives the short outer directory.
    (tmp_dir / "ray").mkdir()
    try:
        overlay, overlay_install_s = _install_candidate_overlay(
            Path(str(artifact["wheel"])), args.result.resolve().parent
        )
    except BaseException:
        _safe_remove_ray_tmp(tmp_dir)
        raise
    sys.path.insert(0, str(overlay))
    old_tmpdir = os.environ.get("RAY_TMPDIR")
    old_admission = os.environ.get("RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL")
    os.environ["RAY_TMPDIR"] = str(tmp_dir)
    os.environ["RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL"] = "1"

    ray = None
    cluster = None
    nodes = []
    threads: list[threading.Thread] = []
    samples: list[dict[str, object]] = []
    marks: dict[str, float] = {}
    cleanup: dict[str, object] = {}
    started_at = _utc_now()
    started = time.monotonic()
    wall_to_monotonic = started - time.time()
    document: dict[str, object] | None = None
    try:
        import ray as ray_module
        from benchmark.ray_core_state import CoreGcsStateReader
        from ray.cluster_utils import Cluster
        from ray.data._internal.execution.interfaces import ExecutionResources

        ray = ray_module
        context = ray.data.DataContext.get_current()
        if not hasattr(context, "_enable_resource_admission_control"):
            raise RuntimeError("candidate wheel lacks resource-admission control")
        context._enable_resource_admission_control = True
        context.execution_options.resource_limits = ExecutionResources.for_limits(
            gpu=LOGICAL_GPU_CEILING
        )

        cluster = Cluster(shutdown_at_exit=False)
        head = cluster.add_node(
            num_cpus=args.cpus_per_node,
            num_gpus=1,
            object_store_memory=args.object_store_bytes,
            temp_dir=str(tmp_dir),
            include_dashboard=False,
        )
        nodes.append(head)
        ray.init(address=cluster.address, log_to_driver=False)
        reader = CoreGcsStateReader()
        progress = ray.remote(num_cpus=0)(_LogicalProgress).remote()

        def sample(event: str) -> dict[str, object]:
            progress_snapshot = ray.get(
                progress.snapshot.remote(), timeout=args.sample_timeout_seconds
            )
            actor_records = reader.actors(limit=1_000)
            value = {
                "elapsed_s": time.monotonic() - started,
                "observed_at": _utc_now(),
                "event": event,
                "alive_raylets": sum(1 for item in ray.nodes() if item["Alive"]),
                "cluster_resources": {
                    key: float(amount)
                    for key, amount in ray.cluster_resources().items()
                },
                "available_resources": {
                    key: float(amount)
                    for key, amount in ray.available_resources().items()
                },
                "gpu_actor_demand": _active_gpu_actor_demand(actor_records),
                "progress": progress_snapshot,
            }
            samples.append(value)
            return value

        def record_first_progress(
            sample_value: Mapping[str, object], phase: str, mark: str
        ) -> None:
            if mark in marks:
                return
            progress_value = sample_value.get("progress")
            phase_value = (
                progress_value.get(phase)
                if isinstance(progress_value, Mapping)
                else None
            )
            if (
                isinstance(phase_value, Mapping)
                and phase_value.get("first_progress_wall_s") is not None
            ):
                marks[mark] = (
                    float(phase_value["first_progress_wall_s"]) + wall_to_monotonic
                )

        cold = _build_actor_map(args, progress, "cold")
        marks["cold_request"] = time.monotonic()
        cold_thread, cold_finished, cold_result = _start_materialization(cold)
        threads.append(cold_thread)

        demand_deadline = time.monotonic() + args.ready_timeout_seconds
        while True:
            if cold_finished.is_set():
                _raise_materialization_error(cold_result)
                raise RuntimeError(
                    "cold map completed before logical capacity was added"
                )
            pending_sample = sample("waiting-for-pending-demand")
            if _pending_without_progress(pending_sample, "cold"):
                marks["pending_demand"] = time.monotonic()
                break
            if time.monotonic() >= demand_deadline:
                raise TimeoutError(
                    "candidate did not expose two-actor GPU demand on one logical GPU"
                )
            time.sleep(args.sample_interval_seconds)

        observation_deadline = (
            marks["pending_demand"] + args.pending_observation_seconds
        )
        pending_observations = 1
        while time.monotonic() < observation_deadline:
            time.sleep(
                min(
                    args.sample_interval_seconds,
                    max(0.0, observation_deadline - time.monotonic()),
                )
            )
            if cold_finished.is_set():
                _raise_materialization_error(cold_result)
                raise RuntimeError(
                    "cold map progressed before logical capacity was added"
                )
            pending_sample = sample("pending-demand-no-progress")
            if not _pending_without_progress(pending_sample, "cold"):
                raise RuntimeError(
                    "pending two-actor demand was not stable with one logical GPU"
                )
            pending_observations += 1

        marks["capacity_add_request"] = time.monotonic()
        worker = cluster.add_node(
            num_cpus=args.cpus_per_node,
            num_gpus=1,
            object_store_memory=args.object_store_bytes,
        )
        nodes.append(worker)
        node_deadline = time.monotonic() + args.ready_timeout_seconds
        while True:
            node_sample = sample("waiting-for-second-logical-raylet")
            cluster_gpu = float(node_sample["cluster_resources"].get("GPU", 0.0))
            if (
                cluster_gpu >= LOGICAL_GPU_CEILING
                and int(node_sample["alive_raylets"]) >= LOGICAL_GPU_CEILING
            ):
                marks["node_visible"] = time.monotonic()
                break
            if time.monotonic() >= node_deadline:
                raise TimeoutError("second logical-GPU raylet did not become visible")
            time.sleep(args.sample_interval_seconds)

        completion_deadline = time.monotonic() + args.ready_timeout_seconds
        while not cold_finished.is_set():
            cold_sample = sample("cold-running-after-capacity")
            cold_demand = cold_sample["gpu_actor_demand"]
            if (
                "both_gpu_actors_alive" not in marks
                and int(cold_demand["active_gpu_actor_requests"]) >= LOGICAL_GPU_CEILING
                and int(cold_demand["pending_gpu_actor_requests"]) == 0
            ):
                marks["both_gpu_actors_alive"] = time.monotonic()
            record_first_progress(cold_sample, "cold", "cold_first_progress")
            if time.monotonic() >= completion_deadline:
                raise TimeoutError("cold logical actor map did not complete")
            time.sleep(args.sample_interval_seconds)
        cold_thread.join(timeout=1)
        _raise_materialization_error(cold_result)
        marks["cold_completion"] = float(cold_result["finished_monotonic"])
        final_cold_sample = sample("cold-complete")
        record_first_progress(final_cold_sample, "cold", "cold_first_progress")
        if "cold_first_progress" not in marks:
            raise RuntimeError("cold map completed without progress telemetry")
        if int(cold_result.get("rows", -1)) != args.rows:
            raise RuntimeError("cold map row-count oracle failed")

        # Wait for the cold actor pool to release its logical GPUs.  The warm
        # replay starts only after both raylets remain visible and idle.
        idle_deadline = time.monotonic() + args.ready_timeout_seconds
        while True:
            idle_sample = sample("waiting-for-warm-idle-capacity")
            if float(idle_sample["available_resources"].get("GPU", 0.0)) >= (
                LOGICAL_GPU_CEILING - _GPU_EPSILON
            ):
                break
            if time.monotonic() >= idle_deadline:
                raise TimeoutError("cold actor pool did not release logical GPUs")
            time.sleep(args.sample_interval_seconds)

        warm = _build_actor_map(args, progress, "warm")
        marks["warm_request"] = time.monotonic()
        warm_thread, warm_finished, warm_result = _start_materialization(warm)
        threads.append(warm_thread)
        warm_deadline = time.monotonic() + args.ready_timeout_seconds
        while not warm_finished.is_set():
            warm_sample = sample("warm-running-on-expanded-cluster")
            record_first_progress(warm_sample, "warm", "warm_first_progress")
            if float(warm_sample["cluster_resources"].get("GPU", 0.0)) < (
                LOGICAL_GPU_CEILING - _GPU_EPSILON
            ):
                raise RuntimeError("logical capacity shrank during warm replay")
            if time.monotonic() >= warm_deadline:
                raise TimeoutError("warm logical actor map did not complete")
            time.sleep(args.sample_interval_seconds)
        warm_thread.join(timeout=1)
        _raise_materialization_error(warm_result)
        marks["warm_completion"] = float(warm_result["finished_monotonic"])
        final_warm_sample = sample("warm-complete")
        record_first_progress(final_warm_sample, "warm", "warm_first_progress")
        if "warm_first_progress" not in marks:
            raise RuntimeError("warm map completed without progress telemetry")
        if int(warm_result.get("rows", -1)) != args.rows:
            raise RuntimeError("warm map row-count oracle failed")

        document = {
            "schema_version": 2,
            "status": "success",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "mode": "local-ray-data-logical-autoscaling-simulation",
            "operator_shape": {
                "kind": "Ray Data actor map",
                "actor_pool_strategy": {"min_size": 2, "max_size": 2},
                "per_actor_resources": {"CPU": 0, "GPU": 1},
                "explicit_execution_gpu_ceiling": 2,
                "resource_admission_enabled": True,
                "ray_data_cluster_autoscaler": os.environ.get(
                    "RAY_DATA_CLUSTER_AUTOSCALER", "V2"
                ),
            },
            "candidate_artifact": artifact,
            "candidate_overlay_install_s": overlay_install_s,
            "ray_version": ray.__version__,
            "installed_ray_commit": str(getattr(ray, "__commit__", "")),
            "pending_evidence": {
                "observations": pending_observations,
                "observation_window_s": args.pending_observation_seconds,
                "two_actor_requests_visible": True,
                "at_least_one_gpu_actor_pending": True,
                "data_path_batches_before_capacity": 0,
                "one_logical_gpu_fully_owned": True,
            },
            "cold_output_rows": int(cold_result["rows"]),
            "warm_output_rows": int(warm_result["rows"]),
            "timing": _timing_metrics(marks),
            "samples": samples,
            "claims": {
                "all_raylets_share_one_physical_host": True,
                "simulation_udf_calls_cuda": False,
                "measures_ec2_provisioning": False,
                "measures_multi_host_networking": False,
                "measures_gpu_throughput": False,
            },
            "disclaimer": (
                "This is a logical Ray scheduler and Ray Data admission simulation "
                "on one host. It cannot replace G6 autoscaling or throughput evidence."
            ),
        }
    finally:
        process_ids = sorted({pid for node in nodes for pid in _node_processes(node)})
        shutdown_warnings = []
        fallback_errors = []
        try:
            if ray is not None:
                if ray.is_initialized():
                    ray.shutdown()
                # The candidate files can be deleted only after Ray is fully
                # stopped.  Its default atexit hook dynamically imports
                # ``ray.dag`` and would otherwise read the removed overlay.
                import ray._private.worker

                atexit.unregister(ray._private.worker.shutdown)
        except BaseException as error:
            shutdown_warnings.append(
                {
                    "step": "driver_disconnect",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        try:
            if cluster is not None:
                cluster.shutdown()
        except BaseException as error:
            shutdown_warnings.append(
                {
                    "step": "cluster_shutdown",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            for node in nodes:
                try:
                    node.kill_all_processes(
                        check_alive=False, allow_graceful=True, wait=True
                    )
                except BaseException as fallback_error:
                    fallback_errors.append(
                        {
                            "type": type(fallback_error).__name__,
                            "message": str(fallback_error),
                        }
                    )
        for thread in threads:
            thread.join(timeout=5)
        alive = _wait_for_owned_processes(process_ids)
        tmp_removed = _safe_remove_ray_tmp(tmp_dir) if not alive else False
        if str(overlay) in sys.path:
            sys.path.remove(str(overlay))
        shutil.rmtree(overlay, ignore_errors=True)
        overlay_removed = not overlay.exists()
        if old_tmpdir is None:
            os.environ.pop("RAY_TMPDIR", None)
        else:
            os.environ["RAY_TMPDIR"] = old_tmpdir
        if old_admission is None:
            os.environ.pop("RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL", None)
        else:
            os.environ["RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL"] = old_admission
        cleanup = {
            "owned_process_ids": process_ids,
            "owned_processes_alive_after_shutdown": alive,
            "ray_tmpdir": str(tmp_dir),
            "ray_tmpdir_removed": tmp_removed,
            "candidate_overlay_removed": overlay_removed,
            "materialization_threads_alive": [
                thread.name for thread in threads if thread.is_alive()
            ],
            "shutdown_warnings": shutdown_warnings,
            "fallback_errors": fallback_errors,
        }
        if (
            alive
            or not tmp_removed
            or not overlay_removed
            or cleanup["materialization_threads_alive"]
            or fallback_errors
        ):
            raise RuntimeError(f"simulation cleanup failed: {cleanup}")

    if document is None:
        raise AssertionError("simulation produced no result")
    document["cleanup"] = cleanup
    document["cleanup_proven"] = True
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-logical-gpus", type=int, default=2)
    parser.add_argument(
        "--node-interval-seconds",
        "--pending-observation-seconds",
        dest="pending_observation_seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument("--ready-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.2)
    parser.add_argument("--sample-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--object-store-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--cpus-per-node", type=int, default=2)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--execute-local", action="store_true")
    args = parser.parse_args(argv)
    if args.max_logical_gpus != LOGICAL_GPU_CEILING:
        parser.error("this controlled simulation requires exactly 2 logical GPUs")
    for name in (
        "pending_observation_seconds",
        "ready_timeout_seconds",
        "sample_interval_seconds",
        "sample_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("object_store_bytes", "cpus_per_node", "rows", "blocks", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = _candidate_artifact(args.candidate_wheel)
    if not args.execute_local:
        document = {
            "schema_version": 2,
            "status": "dry_run",
            "mode": "local-ray-data-logical-autoscaling-simulation",
            "candidate_artifact": artifact,
            "operator_shape": {
                "kind": "Ray Data actor map",
                "actor_pool_strategy": {"min_size": 2, "max_size": 2},
                "per_actor_resources": {"CPU": 0, "GPU": 1},
                "explicit_execution_gpu_ceiling": 2,
                "resource_admission_enabled": True,
                "ray_data_cluster_autoscaler": os.environ.get(
                    "RAY_DATA_CLUSTER_AUTOSCALER", "V2"
                ),
            },
            "experiment_steps": [
                "start one logical-GPU raylet",
                "request a fixed two-actor GPU map through candidate admission",
                "prove pending demand and zero map progress",
                "add the second logical-GPU raylet",
                "measure capacity visibility, first progress, and completion",
                "replay warm on the unchanged two-raylet cluster",
            ],
            "will_start_ray": False,
            "uses_ray_stop": False,
            "disclaimer": (
                "This is a logical scheduler simulation on one host. It cannot replace "
                "the G6 autoscaling experiments."
            ),
        }
    else:
        document = run_simulation(args)
    _atomic_json(args.result.resolve(), document)
    print(f"{document['status']}: {args.result.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
