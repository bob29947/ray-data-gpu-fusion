#!/usr/bin/env python3
"""Validate the pinned runtime, Ray artifact, GPU nodes, and job cleanup."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_DISTRIBUTIONS = {
    "boto3": "1.42.90",
    "botocore": "1.42.90",
    "cudf": "25.12",
    "pyarrow": "19.",
    "rapidsmpf": "25.12",
    "ucxx": "0.47",
}
EXPECTED_GRANT_FIELDS = {
    "stock": None,
    "pg-only": ["max_units", "may_submit"],
    "minimal": ["max_units", "may_submit"],
    "prototype": ["max_units"],
}
DEFAULT_GPU_PROBE_BYTES = 256 * 1024 * 1024
MINIMUM_GPU_PROBE_BYTES = 64 * 1024 * 1024
TERMINAL_ACTOR_STATES = {"DEAD"}
TERMINAL_PLACEMENT_GROUP_STATES = {"REMOVED"}
STAGED_HARNESS_MANIFEST = "MANIFEST.json"


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required distribution is missing: {name}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(document: object) -> bytes:
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode()


def staged_harness_report(
    *, expected_harness_sha256: str, root: Path = PROJECT_ROOT
) -> dict[str, object]:
    """Verify the content-addressed harness before importing workload code."""
    manifest_path = root / STAGED_HARNESS_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid staged harness manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "files",
        "content_sha256",
    }:
        raise RuntimeError("staged harness manifest has an invalid schema")
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict):
        raise RuntimeError("staged harness manifest has an invalid schema")
    content_sha256 = hashlib.sha256(
        _canonical_json_bytes({"schema_version": 1, "files": files})
    ).hexdigest()
    if (
        content_sha256 != expected_harness_sha256
        or manifest.get("content_sha256") != expected_harness_sha256
    ):
        raise RuntimeError("staged harness content digest does not match its manifest")

    total_size = manifest_path.stat().st_size
    for relative_name, entry in files.items():
        relative = Path(relative_name)
        if (
            not relative_name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_name
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "size_bytes"}
        ):
            raise RuntimeError(f"unsafe staged harness entry: {relative_name!r}")
        path = root / relative
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not path.is_file()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise RuntimeError(f"staged harness file failed verification: {relative_name}")
        total_size += size
    return {
        "root": str(root.resolve()),
        "content_sha256": content_sha256,
        "manifest_sha256": _sha256_file(manifest_path),
        "file_count": len(files),
        "total_size_bytes": total_size,
    }


def _resource_admission_fields() -> list[str] | None:
    try:
        admission = importlib.import_module(
            "ray.data._internal.execution.resource_admission"
        )
    except ModuleNotFoundError as error:
        if error.name == "ray.data._internal.execution.resource_admission":
            return None
        raise
    return sorted(admission.ResourceAdmissionGrant.__dataclass_fields__)


def _ray_direct_url() -> dict[str, object]:
    document = metadata.distribution("ray").read_text("direct_url.json")
    if document is None:
        raise RuntimeError("installed Ray has no PEP 610 direct_url.json provenance")
    try:
        parsed = json.loads(document)
    except json.JSONDecodeError as error:
        raise RuntimeError("installed Ray direct_url.json is malformed") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("installed Ray direct_url.json must be an object")
    return parsed


def ray_install_report(
    *,
    arm: str,
    expected_wheel_path: Path,
    expected_wheel_sha256: str,
    expected_ray_commit: str,
    expected_source_provenance: str,
) -> dict[str, object]:
    """Prove that this process imported the intended, mounted Ray wheel."""
    import ray

    wheel_path = expected_wheel_path.expanduser().resolve()
    if not wheel_path.is_file():
        raise RuntimeError(f"expected mounted Ray wheel is missing: {wheel_path}")
    mounted_sha256 = _sha256_file(wheel_path)
    if mounted_sha256 != expected_wheel_sha256:
        raise RuntimeError(
            "mounted Ray wheel digest mismatch: "
            f"expected {expected_wheel_sha256}, got {mounted_sha256}"
        )

    installed_commit = str(getattr(ray, "__commit__", ""))
    if installed_commit != expected_ray_commit:
        raise RuntimeError(
            "installed Ray commit mismatch: "
            f"expected {expected_ray_commit}, got {installed_commit or '<missing>'}"
        )

    fields = _resource_admission_fields()
    expected_fields = EXPECTED_GRANT_FIELDS[arm]
    if fields != expected_fields:
        raise RuntimeError(
            f"{arm} requires ResourceAdmissionGrant fields {expected_fields!r}, "
            f"got {fields!r}"
        )

    direct_url = _ray_direct_url()
    parsed_url = urlparse(str(direct_url.get("url", "")))
    if parsed_url.scheme != "file":
        raise RuntimeError(
            "installed Ray provenance is not a mounted file URL: "
            f"{direct_url.get('url')!r}"
        )
    installed_source = Path(unquote(parsed_url.path)).resolve()
    if installed_source != wheel_path:
        raise RuntimeError(
            "installed Ray came from a different wheel: "
            f"expected {wheel_path}, got {installed_source}"
        )
    archive_info = direct_url.get("archive_info") or {}
    archive_hash = archive_info.get("hash") if isinstance(archive_info, dict) else None
    if archive_hash and archive_hash != f"sha256={mounted_sha256}":
        raise RuntimeError(
            "installed Ray direct-URL digest does not match the mounted wheel: "
            f"{archive_hash!r}"
        )

    return {
        "arm": arm,
        "version": ray.__version__,
        "commit": installed_commit,
        "grant_fields": fields,
        "mounted_wheel_path": str(wheel_path),
        "mounted_wheel_sha256": mounted_sha256,
        "direct_url": direct_url,
        "expected_source_provenance": expected_source_provenance,
    }


def local_node_report(
    *,
    arm: str,
    expected_wheel_path: Path,
    expected_wheel_sha256: str,
    expected_ray_commit: str,
    expected_source_provenance: str,
    expected_harness_sha256: str,
) -> dict[str, object]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"evidence runtime requires CPython 3.11, got {platform.python_version()}"
        )
    versions = {name: _distribution_version(name) for name in EXPECTED_DISTRIBUTIONS}
    for name, prefix in EXPECTED_DISTRIBUTIONS.items():
        if not versions[name].startswith(prefix):
            raise RuntimeError(
                f"{name} must match the pinned {prefix} series, got {versions[name]}"
            )
    for module_name in ("cudf", "cupy", "rapidsmpf", "ucxx"):
        importlib.import_module(module_name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "distributions": versions,
        "state_reader": "core-gcs",
        "staged_harness": staged_harness_report(
            expected_harness_sha256=expected_harness_sha256
        ),
        "ray_install": ray_install_report(
            arm=arm,
            expected_wheel_path=expected_wheel_path,
            expected_wheel_sha256=expected_wheel_sha256,
            expected_ray_commit=expected_ray_commit,
            expected_source_provenance=expected_source_provenance,
        ),
    }


def _decode_nvml(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _gpu_probe(
    *,
    allocation_bytes: int,
    arm: str,
    expected_wheel_path: str,
    expected_wheel_sha256: str,
    expected_ray_commit: str,
    expected_source_provenance: str,
    expected_harness_sha256: str,
) -> dict[str, object]:
    import cupy
    import pynvml
    import ray

    visible_count = int(cupy.cuda.runtime.getDeviceCount())
    if visible_count != 1:
        raise RuntimeError(f"Ray assigned one GPU but the task sees {visible_count}")
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    cuda_name = _decode_nvml(properties.get("name", b"unknown"))
    if cuda_name != "NVIDIA L4":
        raise RuntimeError(f"expected NVIDIA L4, CUDA reports {cuda_name!r}")

    pynvml.nvmlInit()
    try:
        if pynvml.nvmlDeviceGetCount() != 1:
            raise RuntimeError("the G6 node must expose exactly one physical GPU")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        nvml_name = _decode_nvml(pynvml.nvmlDeviceGetName(handle))
        if nvml_name != "NVIDIA L4":
            raise RuntimeError(f"expected NVIDIA L4, NVML reports {nvml_name!r}")
        uuid = _decode_nvml(pynvml.nvmlDeviceGetUUID(handle))
        driver_version = _decode_nvml(pynvml.nvmlSystemGetDriverVersion())
        before = pynvml.nvmlDeviceGetMemoryInfo(handle)

        element_count = allocation_bytes // cupy.dtype(cupy.float32).itemsize
        allocation = cupy.empty(element_count, dtype=cupy.float32)
        allocation.fill(1.0)
        checksum = float(cupy.sum(allocation[: min(element_count, 4096)]).get())
        cupy.cuda.get_current_stream().synchronize()
        after = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if int(after.used) <= int(before.used):
            raise RuntimeError(
                "CuPy allocation did not increase observed GPU memory use"
            )
        del allocation
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.cuda.get_current_stream().synchronize()

        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": nvml_name,
            "device_uuid": uuid,
            "driver_version": driver_version,
            "total_memory_bytes": int(after.total),
            "free_memory_before_bytes": int(before.free),
            "used_memory_before_bytes": int(before.used),
            "used_memory_after_bytes": int(after.used),
            "allocation_bytes": allocation_bytes,
            "allocation_checksum": checksum,
            "runtime": local_node_report(
                arm=arm,
                expected_wheel_path=Path(expected_wheel_path),
                expected_wheel_sha256=expected_wheel_sha256,
                expected_ray_commit=expected_ray_commit,
                expected_source_provenance=expected_source_provenance,
                expected_harness_sha256=expected_harness_sha256,
            ),
        }
    finally:
        pynvml.nvmlShutdown()


def _alive_gpu_nodes(ray_module: object) -> list[dict[str, object]]:
    nodes = []
    for node in ray_module.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources") or {}
        gpu_count = float(resources.get("GPU", 0))
        if gpu_count <= 0:
            continue
        if gpu_count != 1:
            raise RuntimeError(
                f"G6 node {node.get('NodeID')} advertises {gpu_count} logical GPUs"
            )
        nodes.append(node)
    return sorted(nodes, key=lambda node: str(node.get("NodeID")))


def _wait_for_gpu_nodes(
    ray_module: object,
    *,
    minimum: int,
    timeout_seconds: float,
    poll_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    started = monotonic()
    observations = []
    while True:
        nodes = _alive_gpu_nodes(ray_module)
        elapsed = monotonic() - started
        observations.append(
            {
                "elapsed_seconds": round(elapsed, 3),
                "gpu_node_ids": [str(node.get("NodeID")) for node in nodes],
            }
        )
        if len(nodes) >= minimum:
            return nodes, observations
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise RuntimeError(
                "timed out waiting for fixed minimum GPU capacity: "
                f"expected {minimum} nodes, saw {len(nodes)}"
            )
        sleep(min(poll_seconds, remaining))


def cluster_report(
    *,
    arm: str,
    expected_min_gpus: int,
    expected_max_gpus: int,
    capacity_timeout_seconds: float,
    capacity_poll_seconds: float,
    gpu_probe_bytes: int,
    expected_wheel_path: Path,
    expected_wheel_sha256: str,
    expected_ray_commit: str,
    expected_source_provenance: str,
    expected_harness_sha256: str,
) -> dict[str, object]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address="auto")
    gpu_nodes, capacity_observations = _wait_for_gpu_nodes(
        ray,
        minimum=expected_min_gpus,
        timeout_seconds=capacity_timeout_seconds,
        poll_seconds=capacity_poll_seconds,
    )
    if len(gpu_nodes) > expected_max_gpus:
        raise RuntimeError(
            "cluster GPU capacity exceeds topology maximum: "
            f"expected at most {expected_max_gpus}, got {len(gpu_nodes)}"
        )

    remote_probe = ray.remote(num_cpus=0, num_gpus=1)(_gpu_probe)
    probe_refs = []
    for node in gpu_nodes:
        node_id = str(node["NodeID"])
        probe_refs.append(
            remote_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(
                allocation_bytes=gpu_probe_bytes,
                arm=arm,
                expected_wheel_path=str(expected_wheel_path),
                expected_wheel_sha256=expected_wheel_sha256,
                expected_ray_commit=expected_ray_commit,
                expected_source_provenance=expected_source_provenance,
                expected_harness_sha256=expected_harness_sha256,
            )
        )
    probes = ray.get(probe_refs)
    expected_node_ids = {str(node["NodeID"]) for node in gpu_nodes}
    probed_node_ids = {str(probe["node_id"]) for probe in probes}
    if probed_node_ids != expected_node_ids:
        raise RuntimeError(
            "node-affinity GPU probes did not cover every available GPU node: "
            f"expected {sorted(expected_node_ids)}, got {sorted(probed_node_ids)}"
        )

    resources = ray.cluster_resources()
    return {
        "arm": arm,
        "ray_version": ray.__version__,
        "ray_commit": getattr(ray, "__commit__", None),
        "resource_admission_grant_fields": _resource_admission_fields(),
        "cluster_resources": {key: float(value) for key, value in resources.items()},
        "alive_nodes": sum(1 for node in ray.nodes() if node.get("Alive")),
        "capacity_wait_observations": capacity_observations,
        "gpu_probes": probes,
        "coverage": {
            "gpu_nodes_present": len(gpu_nodes),
            "gpu_nodes_probed": len(probes),
            "topology_min_gpus": expected_min_gpus,
            "topology_max_gpus": expected_max_gpus,
            "future_autoscaled_nodes": (
                "validated by setup-command preflight when each node joins; "
                "this cluster preflight intentionally does not warm them"
            ),
        },
    }


def _nonterminal_records(
    records: Sequence[Mapping[str, object]], *, terminal_states: set[str]
) -> list[dict[str, object]]:
    return [
        dict(document)
        for document in records
        if str(document.get("state")) not in terminal_states
    ]


def _select_fields(
    documents: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> list[dict[str, object]]:
    """Keep leak evidence useful without copying runtime environments to logs."""
    return [
        {field: document.get(field) for field in fields if field in document}
        for document in documents
    ]


def audit_job_resources(
    *,
    job_id: str,
    timeout_seconds: float,
    poll_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Poll job-scoped actors and placement groups without requesting resources."""
    import ray
    from benchmark.ray_core_state import (
        CoreGcsStateReader,
        is_permitted_detached_ray_data_service,
    )

    ray.init(address="auto")
    reader = CoreGcsStateReader()
    started = monotonic()
    observations = []
    while True:
        actor_records = _nonterminal_records(
            reader.actors(job_id=job_id),
            terminal_states=TERMINAL_ACTOR_STATES,
        )
        permitted_services = _select_fields(
            [
                record
                for record in actor_records
                if is_permitted_detached_ray_data_service(record)
            ],
            ("actor_id", "class_name", "name", "state", "job_id", "node_id", "pid"),
        )
        actors = _select_fields(
            [
                record
                for record in actor_records
                if not is_permitted_detached_ray_data_service(record)
            ],
            (
                "actor_id",
                "class_name",
                "name",
                "state",
                "job_id",
                "node_id",
                "pid",
                "required_resources",
                "placement_group_id",
                "is_detached",
            ),
        )
        placement_groups = _select_fields(
            _nonterminal_records(
                reader.placement_groups(job_id=job_id),
                terminal_states=TERMINAL_PLACEMENT_GROUP_STATES,
            ),
            (
                "placement_group_id",
                "name",
                "state",
                "creator_job_id",
                "bundles",
                "is_detached",
                "stats",
            ),
        )
        elapsed = monotonic() - started
        observation = {
            "elapsed_seconds": round(elapsed, 3),
            "nonterminal_actor_count": len(actors),
            "nonterminal_placement_group_count": len(placement_groups),
            "permitted_detached_service_count": len(permitted_services),
        }
        observations.append(observation)
        if not actors and not placement_groups:
            return {
                "job_id": job_id,
                "clean": True,
                "elapsed_seconds": round(elapsed, 3),
                "observations": observations,
                "nonterminal_actors": [],
                "nonterminal_placement_groups": [],
                "permitted_detached_ray_data_services": permitted_services,
            }
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            return {
                "job_id": job_id,
                "clean": False,
                "elapsed_seconds": round(elapsed, 3),
                "observations": observations,
                "nonterminal_actors": actors,
                "nonterminal_placement_groups": placement_groups,
                "permitted_detached_ray_data_services": permitted_services,
            }
        sleep(min(poll_seconds, remaining))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local-node-only", action="store_true")
    mode.add_argument("--check-job-id")
    parser.add_argument("--arm", choices=("stock", "pg-only", "minimal", "prototype"))
    parser.add_argument("--expected-min-gpus", type=int)
    parser.add_argument("--expected-max-gpus", type=int)
    parser.add_argument("--capacity-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--capacity-poll-seconds", type=float, default=5.0)
    parser.add_argument("--gpu-probe-bytes", type=int, default=DEFAULT_GPU_PROBE_BYTES)
    parser.add_argument("--expected-wheel-path", type=Path)
    parser.add_argument("--expected-wheel-sha256")
    parser.add_argument("--expected-ray-commit")
    parser.add_argument("--expected-source-provenance")
    parser.add_argument("--expected-harness-sha256")
    parser.add_argument("--leak-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--leak-poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    positive = (
        "capacity_timeout_seconds",
        "capacity_poll_seconds",
        "gpu_probe_bytes",
        "leak_timeout_seconds",
        "leak_poll_seconds",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.check_job_id:
        return args

    provenance = (
        "arm",
        "expected_wheel_path",
        "expected_wheel_sha256",
        "expected_ray_commit",
        "expected_source_provenance",
        "expected_harness_sha256",
    )
    missing = [name for name in provenance if not getattr(args, name)]
    if missing:
        parser.error(
            "runtime preflight requires provenance arguments: " + ", ".join(missing)
        )
    if len(args.expected_wheel_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF"
        for character in args.expected_wheel_sha256
    ):
        parser.error("--expected-wheel-sha256 must be a SHA-256 hex digest")
    if len(args.expected_harness_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.expected_harness_sha256
    ):
        parser.error("--expected-harness-sha256 must be a lowercase SHA-256 digest")
    if args.gpu_probe_bytes < MINIMUM_GPU_PROBE_BYTES:
        parser.error(
            f"--gpu-probe-bytes must be at least {MINIMUM_GPU_PROBE_BYTES} bytes"
        )
    if not args.local_node_only:
        if args.expected_min_gpus is None or args.expected_max_gpus is None:
            parser.error("cluster preflight requires both expected GPU bounds")
        if not 0 <= args.expected_min_gpus <= args.expected_max_gpus:
            parser.error("expected GPU bounds must satisfy 0 <= min <= max")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_job_id:
        leak_audit = audit_job_resources(
            job_id=args.check_job_id,
            timeout_seconds=args.leak_timeout_seconds,
            poll_seconds=args.leak_poll_seconds,
        )
        print("PREFLIGHT=" + json.dumps({"leak_audit": leak_audit}, sort_keys=True))
        return 0 if leak_audit["clean"] else 1

    report = local_node_report(
        arm=args.arm,
        expected_wheel_path=args.expected_wheel_path,
        expected_wheel_sha256=args.expected_wheel_sha256,
        expected_ray_commit=args.expected_ray_commit,
        expected_source_provenance=args.expected_source_provenance,
        expected_harness_sha256=args.expected_harness_sha256,
    )
    if not args.local_node_only:
        report["cluster"] = cluster_report(
            arm=args.arm,
            expected_min_gpus=args.expected_min_gpus,
            expected_max_gpus=args.expected_max_gpus,
            capacity_timeout_seconds=args.capacity_timeout_seconds,
            capacity_poll_seconds=args.capacity_poll_seconds,
            gpu_probe_bytes=args.gpu_probe_bytes,
            expected_wheel_path=args.expected_wheel_path,
            expected_wheel_sha256=args.expected_wheel_sha256,
            expected_ray_commit=args.expected_ray_commit,
            expected_source_provenance=args.expected_source_provenance,
            expected_harness_sha256=args.expected_harness_sha256,
        )
    print("PREFLIGHT=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
