#!/usr/bin/env python3
"""Run local runtime/GPU preflight and post-job Ray-state cleanup audits."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


EXPECTED_GRANT_FIELDS = {
    "stock": None,
    "pg-only": ["max_units", "may_submit"],
    "minimal": ["max_units", "may_submit"],
    "prototype": ["max_units"],
}
TERMINAL_ACTOR_STATES = {"DEAD"}
TERMINAL_PLACEMENT_GROUP_STATES = {"REMOVED"}


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


def _grant_fields() -> list[str] | None:
    try:
        module = importlib.import_module(
            "ray.data._internal.execution.resource_admission"
        )
    except ModuleNotFoundError as error:
        if error.name == "ray.data._internal.execution.resource_admission":
            return None
        raise
    return sorted(module.ResourceAdmissionGrant.__dataclass_fields__)


def _gpu_actor_report(allocation_bytes: int) -> dict[str, object]:
    import cupy
    import ray

    count = int(cupy.cuda.runtime.getDeviceCount())
    if count != 1:
        raise RuntimeError(f"one-GPU Ray actor sees {count} CUDA devices")
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    raw_name = properties.get("name", b"unknown")
    name = (
        raw_name.decode(errors="replace")
        if isinstance(raw_name, bytes)
        else str(raw_name)
    )
    elements = max(1, allocation_bytes // cupy.dtype(cupy.float32).itemsize)
    allocation = cupy.empty(elements, dtype=cupy.float32)
    allocation.fill(1.0)
    checksum = float(cupy.sum(allocation[: min(elements, 4096)]).get())
    cupy.cuda.get_current_stream().synchronize()
    del allocation
    cupy.get_default_memory_pool().free_all_blocks()
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": count,
        "device_name": name,
        "allocation_bytes": allocation_bytes,
        "allocation_checksum": checksum,
    }


class _LocalGpuProbe:
    def report(self, allocation_bytes: int) -> dict[str, object]:
        return _gpu_actor_report(allocation_bytes)


def runtime_gpu_probe(args: argparse.Namespace) -> dict[str, object]:
    import ray

    wheel = args.wheel.resolve()
    actual_sha = _sha256_file(wheel)
    if actual_sha != args.wheel_sha256:
        raise RuntimeError(
            f"wheel digest mismatch: expected {args.wheel_sha256}, got {actual_sha}"
        )
    if str(getattr(ray, "__commit__", "")) != args.installed_ray_commit:
        raise RuntimeError(
            "imported Ray commit mismatch: "
            f"expected {args.installed_ray_commit}, got {getattr(ray, '__commit__', '')}"
        )
    if args.installed_ray_commit != args.base_commit:
        raise RuntimeError(
            "derived wheel must expose its Ray base commit: "
            f"installed={args.installed_ray_commit}, base={args.base_commit}"
        )
    fields = _grant_fields()
    if fields != EXPECTED_GRANT_FIELDS[args.arm]:
        raise RuntimeError(
            f"{args.arm} grant fields must be {EXPECTED_GRANT_FIELDS[args.arm]!r}, "
            f"got {fields!r}"
        )
    if args.overlay.resolve() not in Path(ray.__file__).resolve().parents:
        raise RuntimeError(
            f"Ray was not imported from the isolated overlay: {ray.__file__}"
        )
    ray.init(address="auto")
    try:
        actor_type = ray.remote(num_cpus=0, num_gpus=1)(_LocalGpuProbe)
        actors = [actor_type.remote() for _ in range(args.capacity)]
        try:
            reports = ray.get(
                [actor.report.remote(args.allocation_bytes) for actor in actors],
                timeout=args.timeout_seconds,
            )
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
        assignments = [str(item["cuda_visible_devices"]) for item in reports]
        if len(set(assignments)) != args.capacity:
            raise RuntimeError(
                "concurrent GPU actors did not receive unique visible devices: "
                f"{assignments}"
            )
        cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))
        if cluster_gpus != args.capacity:
            raise RuntimeError(
                f"Ray advertises {cluster_gpus} GPUs, expected {args.capacity}"
            )
        return {
            "status": "success",
            "observed_at": _utc_now(),
            "arm": args.arm,
            "python": platform.python_version(),
            "ray_version": ray.__version__,
            "source_commit": args.source_commit,
            "base_commit": args.base_commit,
            "installed_ray_commit": str(getattr(ray, "__commit__", "")),
            "ray_file": str(Path(ray.__file__).resolve()),
            "grant_fields": fields,
            "wheel_path": str(wheel),
            "wheel_sha256": actual_sha,
            "cluster_resources": {
                key: float(value) for key, value in ray.cluster_resources().items()
            },
            "gpu_actor_reports": reports,
        }
    finally:
        ray.shutdown()


def _active_job_resources(job_id: str) -> dict[str, list[dict[str, object]]]:
    from benchmark.ray_core_state import (
        CoreGcsStateReader,
        is_permitted_detached_ray_data_service,
    )

    reader = CoreGcsStateReader()
    actor_records = [
        record
        for record in reader.actors(job_id=job_id)
        if str(record["state"]) not in TERMINAL_ACTOR_STATES
    ]
    permitted_services = [
        record
        for record in actor_records
        if is_permitted_detached_ray_data_service(record)
    ]
    actors = [
        {
            "actor_id": str(record["actor_id"]),
            "state": str(record["state"]),
            "class_name": str(record["class_name"]),
            "name": str(record["name"]),
            "pid": record["pid"],
            "required_resources": record["required_resources"],
            "is_detached": record["is_detached"],
        }
        for record in actor_records
        if record not in permitted_services
    ]
    groups = [
        {
            "placement_group_id": str(record["placement_group_id"]),
            "state": str(record["state"]),
            "name": str(record["name"]),
        }
        for record in reader.placement_groups(job_id=job_id)
        if str(record["state"]) not in TERMINAL_PLACEMENT_GROUP_STATES
    ]
    return {
        "active_actors": actors,
        "active_placement_groups": groups,
        "permitted_detached_ray_data_services": [
            {
                "actor_id": str(record["actor_id"]),
                "class_name": str(record["class_name"]),
                "name": str(record["name"]),
                "state": str(record["state"]),
            }
            for record in permitted_services
        ],
    }


def cleanup_probe(args: argparse.Namespace) -> dict[str, object]:
    import ray

    ray.init(address="auto")
    samples = []
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while True:
            resources = _active_job_resources(args.job_id)
            sample = {"observed_at": _utc_now(), **resources}
            samples.append(sample)
            if (
                not resources["active_actors"]
                and not resources["active_placement_groups"]
            ):
                return {
                    "status": "success",
                    "job_id": args.job_id,
                    "cleanup_proven": True,
                    "samples": samples,
                }
            if time.monotonic() >= deadline:
                return {
                    "status": "error",
                    "job_id": args.job_id,
                    "cleanup_proven": False,
                    "samples": samples,
                }
            time.sleep(1.0)
    finally:
        ray.shutdown()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("runtime-gpu", "cleanup"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--arm", choices=tuple(EXPECTED_GRANT_FIELDS))
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--allocation-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--source-commit")
    parser.add_argument("--base-commit")
    parser.add_argument("--installed-ray-commit")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args(argv)
    required = (
        (
            "arm",
            "capacity",
            "wheel",
            "wheel_sha256",
            "source_commit",
            "base_commit",
            "installed_ray_commit",
            "overlay",
        )
        if args.mode == "runtime-gpu"
        else ("job_id",)
    )
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        parser.error(f"{args.mode} requires: {', '.join(missing)}")
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = (
            runtime_gpu_probe(args)
            if args.mode == "runtime-gpu"
            else cleanup_probe(args)
        )
    except BaseException as error:
        document = {
            "status": "error",
            "observed_at": _utc_now(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    _atomic_json(args.result, document)
    return 0 if document.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
