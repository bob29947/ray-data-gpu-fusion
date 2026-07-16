#!/usr/bin/env python3
"""Verify the pinned Ray source, derived wheel seams, and plugin installation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / "pins" / "stock-ray.json").read_text())


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    stock = ROOT / "ray-stock"
    head = _run("git", "-C", str(stock), "rev-parse", "HEAD")
    if head != PIN["commit"]:
        raise RuntimeError(f"ray-stock HEAD {head} != {PIN['commit']}")
    if _run("git", "-C", str(stock), "status", "--porcelain"):
        raise RuntimeError("ray-stock must remain byte-for-byte clean")

    stock_wheel = ROOT / "wheels" / "stock" / PIN["wheel"]
    if _sha256(stock_wheel) != PIN["wheel_sha256"]:
        raise RuntimeError("stock Ray wheel hash mismatch")

    import ray

    if ray.__version__ != PIN["version"] or ray.__commit__ != PIN["commit"]:
        raise RuntimeError(
            f"installed Ray identity mismatch: {ray.__version__} {ray.__commit__}"
        )
    installed_ray = Path(ray.__file__).resolve()
    if stock in installed_ray.parents:
        raise RuntimeError("Ray is imported from ray-stock instead of the wheel")
    expected_prefix = (ROOT / ".venv").resolve()
    if expected_prefix not in Path(sys.executable).resolve().parents:
        raise RuntimeError(
            f"interpreter is outside {expected_prefix}: {sys.executable}"
        )

    from ray.data.context import DataContext
    from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
    from ray.data._internal.execution.operators.actor_pool_map_operator import (
        ActorPoolMapOperator,
    )

    if not hasattr(DataContext(), "custom_physical_optimizer_rule_classes"):
        raise RuntimeError("plan-local physical optimizer plugin seam is absent")
    if not hasattr(ParquetDatasource, "get_external_scan_descriptor"):
        raise RuntimeError("Parquet external scan descriptor seam is absent")
    actor_parameters = inspect.signature(ActorPoolMapOperator.__init__).parameters
    required_actor_seams = {
        "defer_actor_start",
        "wait_for_upstream_deferred_operators",
        "release_idle_actors_on_completion",
    }
    if not required_actor_seams.issubset(actor_parameters):
        raise RuntimeError(
            "demand-driven actor seams are absent: "
            f"{sorted(required_actor_seams - set(actor_parameters))}"
        )

    import ray_data_gpu_fusion as rgf

    compatibility = rgf.compatibility()
    if compatibility.ray_commit != PIN["commit"] or not compatibility.supported:
        raise RuntimeError(f"plugin compatibility check failed: {compatibility}")

    if os.environ.get("PYTHONHOME"):
        raise RuntimeError("PYTHONHOME may redirect the pinned interpreter")
    python_path = os.environ.get("PYTHONPATH", "")
    if str(stock) in python_path:
        raise RuntimeError("PYTHONPATH redirects imports into ray-stock")

    # RAPIDS is installed from the explicit conda lock.  ``pip check`` cannot
    # validate that environment: several RAPIDS conda packages intentionally
    # expose Python distributions whose native dependencies are satisfied by
    # conda rather than PyPI metadata.  Verify the imported runtime and exact
    # pinned versions instead.
    import cudf
    import cupy
    import pyarrow
    import rmm

    runtime_versions = {
        "cudf": cudf.__version__,
        "cupy": cupy.__version__,
        "pyarrow": pyarrow.__version__,
        "rmm": rmm.__version__,
    }
    expected_versions = {
        "cudf": "25.12.00",
        "cupy": "13.6.0",
        "pyarrow": "19.0.1",
        "rmm": "25.12.00",
    }
    if runtime_versions != expected_versions:
        raise RuntimeError(
            "RAPIDS runtime version mismatch: "
            f"installed={runtime_versions}, expected={expected_versions}"
        )
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "ray": ray.__version__,
                "ray_commit": ray.__commit__,
                "ray_path": str(installed_ray),
                "plugin": rgf.__version__,
                "adapter": compatibility.adapter,
                "runtime": runtime_versions,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
