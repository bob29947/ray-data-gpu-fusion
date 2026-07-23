#!/usr/bin/env python3
"""Own one isolated local Ray cluster until the runner requests shutdown.

The daemon is deliberately separate from the workload driver.  This lets the
workload time out (the expected stock-Ray deadlock outcome) while preserving
the cluster long enough to audit job actors and placement groups.  The daemon
starts Ray with ``address="local"`` and only tears down processes it owns; it
never invokes the process-wide ``ray stop`` command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import stat
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


_MOUNT_ESCAPE_RE = re.compile(r"\\([0-7]{3})")


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


def _owned_processes(node: object) -> list[dict[str, object]]:
    result = []
    for process_type, entries in getattr(node, "all_processes", {}).items():
        for entry in entries:
            process = getattr(entry, "process", None)
            pid = getattr(process, "pid", None)
            if isinstance(pid, int):
                result.append({"kind": str(process_type), "pid": pid})
    return sorted(result, key=lambda item: (str(item["kind"]), int(item["pid"])))


def _unescape_mount_field(value: str) -> str:
    return _MOUNT_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _backing_filesystem(path: Path) -> dict[str, object]:
    """Describe the mounted filesystem backing ``path`` without shelling out."""

    resolved = path.resolve(strict=True)
    best: tuple[int, dict[str, str]] | None = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        trailing = after.split()
        if len(fields) < 6 or len(trailing) < 2:
            continue
        mount_point = Path(_unescape_mount_field(fields[4])).resolve(strict=False)
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidate = {
            "device": fields[2],
            "mount_point": str(mount_point),
            "source": _unescape_mount_field(trailing[1]),
            "type": trailing[0],
        }
        score = len(mount_point.parts)
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None:
        raise RuntimeError(f"cannot identify backing filesystem for {resolved}")

    status = resolved.stat()
    capacity = os.statvfs(resolved)
    return {
        **best[1],
        "device_id": status.st_dev,
        "device_major": os.major(status.st_dev),
        "device_minor": os.minor(status.st_dev),
        "available_bytes": capacity.f_bavail * capacity.f_frsize,
    }


def validate_object_spilling_directory(
    spilling_directory: Path,
    *,
    case_directory: Path,
    results_root: Path,
    must_exist: bool | None,
) -> Path:
    """Require the exact ``<case>/ray-spill`` path under the local results root."""

    supplied = {
        "object-spilling-directory": spilling_directory,
        "case-directory": case_directory,
        "results-root": results_root,
    }
    for label, path in supplied.items():
        if not path.is_absolute():
            raise ValueError(f"{label} must be absolute: {path}")

    resolved_root = results_root.resolve(strict=True)
    resolved_case = case_directory.resolve(strict=True)
    resolved_spill = spilling_directory.resolve(strict=False)
    if results_root != resolved_root or case_directory != resolved_case:
        raise ValueError("results-root and case-directory must be canonical paths")
    if tuple(resolved_root.parts[-3:]) != ("benchmark", "results", "local"):
        raise ValueError("results-root must be the benchmark/results/local directory")
    if (
        resolved_case.parent.name != "cases"
        or len(resolved_case.parents) < 3
        or resolved_case.parents[2] != resolved_root
    ):
        raise ValueError(
            f"case-directory must be an exact case leaf under {resolved_root}"
        )
    expected = resolved_case / "ray-spill"
    if spilling_directory != expected or resolved_spill != expected:
        raise ValueError(f"object-spilling-directory must be exactly {expected}")
    if spilling_directory.is_symlink():
        raise ValueError("object-spilling-directory must not be a symlink")
    if must_exist is True and not spilling_directory.is_dir():
        raise ValueError(
            f"object-spilling-directory does not exist: {spilling_directory}"
        )
    if must_exist is False and spilling_directory.exists():
        raise ValueError(
            f"object-spilling-directory must not preexist: {spilling_directory}"
        )
    if spilling_directory.exists() and not spilling_directory.is_dir():
        raise ValueError(
            f"object-spilling-directory is not a directory: {spilling_directory}"
        )
    if (
        spilling_directory.exists()
        and spilling_directory.stat().st_dev != resolved_case.stat().st_dev
    ):
        raise ValueError(
            "object-spilling-directory must use the case directory's backing device"
        )
    return expected


def snapshot_object_spilling_directory(
    spilling_directory: Path,
    *,
    case_directory: Path,
    results_root: Path,
) -> dict[str, object]:
    """Record the spill leaf and its backing storage, including physical files."""

    validated = validate_object_spilling_directory(
        spilling_directory,
        case_directory=case_directory,
        results_root=results_root,
        must_exist=None,
    )
    exists = validated.is_dir()
    file_count = 0
    file_bytes = 0
    scan_race_count = 0
    scan_errors: list[OSError] = []

    def record_scan_error(error: OSError) -> None:
        nonlocal scan_race_count
        if isinstance(error, FileNotFoundError):
            scan_race_count += 1
        else:
            scan_errors.append(error)

    if exists:
        for directory, _subdirectories, files in os.walk(
            validated,
            followlinks=False,
            onerror=record_scan_error,
        ):
            for name in files:
                try:
                    status = os.stat(
                        Path(directory) / name,
                        dir_fd=None,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    scan_race_count += 1
                    continue
                if stat.S_ISREG(status.st_mode):
                    file_count += 1
                    file_bytes += status.st_size
        if not validated.is_dir():
            scan_race_count += 1
    if scan_errors:
        raise RuntimeError(
            "failed to scan object-spilling directory: "
            + "; ".join(str(error) for error in scan_errors)
        )
    filesystem = _backing_filesystem(
        validated if validated.is_dir() else case_directory
    )
    return {
        "path": str(validated),
        "exists": exists,
        "filesystem_device": filesystem["device"],
        "filesystem_device_id": filesystem["device_id"],
        "filesystem_device_major": filesystem["device_major"],
        "filesystem_device_minor": filesystem["device_minor"],
        "filesystem_type": filesystem["type"],
        "filesystem_mount_point": filesystem["mount_point"],
        "filesystem_source": filesystem["source"],
        "available_bytes": filesystem["available_bytes"],
        "spill_file_count": file_count,
        "spill_file_bytes": file_bytes,
        "scan_race_count": scan_race_count,
        "scan_complete": scan_race_count == 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--num-cpus", type=int, required=True)
    parser.add_argument("--object-store-memory", type=int, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--object-spilling-directory", type=Path, required=True)
    parser.add_argument("--case-directory", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--stop-request", type=Path, required=True)
    parser.add_argument("--stopped", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.capacity < 1 or args.num_cpus < 1:
        parser.error("capacity and num-cpus must be positive")
    if args.object_store_memory < 75 * 1024**2:
        parser.error("object-store-memory must be at least 75 MiB")
    if not args.tmp_dir.is_absolute() or not str(args.tmp_dir).startswith(
        "/dev/shm/ray-admission/"
    ):
        parser.error("tmp-dir must be a child of /dev/shm/ray-admission")
    try:
        validate_object_spilling_directory(
            args.object_spilling_directory,
            case_directory=args.case_directory,
            results_root=args.results_root,
            must_exist=False,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_object_spilling_directory(
        args.object_spilling_directory,
        case_directory=args.case_directory,
        results_root=args.results_root,
        must_exist=False,
    )
    args.tmp_dir.mkdir(parents=True, exist_ok=False)
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    ray_node = None
    owned_processes: list[dict[str, object]] = []
    ready_document: dict[str, object] = {}
    try:
        import ray
        import ray._private.worker

        context = ray.init(
            address="local",
            num_cpus=args.num_cpus,
            num_gpus=args.capacity,
            object_store_memory=args.object_store_memory,
            include_dashboard=False,
            _temp_dir=str(args.tmp_dir),
            object_spilling_directory=str(args.object_spilling_directory),
            log_to_driver=False,
        )
        ray_node = ray._private.worker._global_node
        if ray_node is None:
            raise RuntimeError("ray.init did not create an owned local node")
        owned_processes = _owned_processes(ray_node)
        ready_document = {
            "status": "ready",
            "started_at": _utc_now(),
            "daemon_pid": os.getpid(),
            "address": context.address_info["address"],
            "capacity": args.capacity,
            "cluster_resources": {
                key: float(value) for key, value in ray.cluster_resources().items()
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ray_tmpdir": str(args.tmp_dir),
            "object_spilling_directory": str(args.object_spilling_directory),
            "object_spilling_storage": snapshot_object_spilling_directory(
                args.object_spilling_directory,
                case_directory=args.case_directory,
                results_root=args.results_root,
            ),
            "owned_processes": owned_processes,
            "ray_version": ray.__version__,
            "installed_ray_commit": str(getattr(ray, "__commit__", "")),
        }
        _atomic_json(args.ready, ready_document)
        while not stopping and not args.stop_request.exists():
            time.sleep(0.2)
        return 0
    except BaseException as error:
        _atomic_json(
            args.ready,
            {
                "status": "error",
                "observed_at": _utc_now(),
                "daemon_pid": os.getpid(),
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return 1
    finally:
        shutdown_error = None
        if ray_node is None:
            try:
                import ray._private.worker

                ray_node = ray._private.worker._global_node
                if ray_node is not None:
                    owned_processes = _owned_processes(ray_node)
            except BaseException:
                ray_node = None
        if ray_node is not None:
            try:
                import ray

                if ray.is_initialized():
                    ray.shutdown(wait_for_processes=True)
                else:
                    # ray.init can fail after spawning some local processes but
                    # before marking the driver initialized.
                    ray_node.kill_all_processes(
                        check_alive=False, allow_graceful=True, wait=True
                    )
            except BaseException as error:  # retain evidence even during teardown
                shutdown_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
        alive_after = [
            item for item in owned_processes if Path(f"/proc/{item['pid']}").exists()
        ]
        _atomic_json(
            args.stopped,
            {
                "status": "stopped"
                if not shutdown_error and not alive_after
                else "error",
                "stopped_at": _utc_now(),
                "daemon_pid": os.getpid(),
                "address": ready_document.get("address"),
                "owned_processes": owned_processes,
                "owned_processes_alive_after_shutdown": alive_after,
                "shutdown_error": shutdown_error,
            },
        )
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
