"""Bootstrap the wheel-backed Ray candidate tests in drivers and workers.

This module is loaded automatically because the acceptance runner prepends this
directory to ``PYTHONPATH``.  It deliberately mounts only Ray's *test* packages
from the candidate worktree.  Production ``ray`` modules must already come from
the project's ``.venv`` wheel installation, and their origins are checked here
in every Python process, including Ray workers.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path

_TEST_ROOT_ENV = "RAY_GPU_ACCEPTANCE_TEST_ROOT"
_VENV_ENV = "RAY_GPU_ACCEPTANCE_VENV"
_LAYER_ENV = "RAY_GPU_ACCEPTANCE_LAYER"


def _mount_test_package(name: str, directory: Path) -> None:
    if not directory.is_dir():
        raise RuntimeError(f"Ray acceptance test package is missing: {directory}")

    package = types.ModuleType(name)
    package.__file__ = str(directory / "__init__.py")
    package.__package__ = name
    package.__path__ = [str(directory)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[name] = package

    parent_name, attribute = name.rsplit(".", 1)
    setattr(importlib.import_module(parent_name), attribute, package)


def _assert_wheel_backed_production(venv: Path, layer: str) -> None:
    executable = Path(sys.executable).resolve()
    if venv not in executable.parents:
        raise RuntimeError(
            f"Ray acceptance process uses {executable}, not interpreter under {venv}"
        )
    production_modules = (
        "ray",
        "ray.data.context",
        "ray.data._internal.actor_autoscaler.default_actor_autoscaler",
        "ray.data._internal.execution.operators.actor_pool_map_operator",
        "ray.data._internal.execution.interfaces.physical_operator",
        "ray.data._internal.execution.resource_admission",
        "ray.data._internal.execution.resource_manager",
        "ray.data._internal.execution.streaming_executor",
        "ray.data._internal.execution.streaming_executor_state",
        "ray.data._internal.gpu_shuffle.hash_aggregate",
        "ray.data._internal.gpu_shuffle.hash_shuffle",
        "ray.data._internal.logical.optimizers",
        "ray.data._internal.datasource.parquet_datasource",
    )
    imported = {name: importlib.import_module(name) for name in production_modules}
    for name, module in imported.items():
        origin_value = getattr(module, "__file__", None)
        if not origin_value:
            raise RuntimeError(f"production module has no file origin: {name}")
        origin = Path(origin_value).resolve()
        if venv not in origin.parents:
            raise RuntimeError(
                f"production module {name} came from {origin}, not {venv}"
            )

    context_module = imported["ray.data.context"]
    resource_manager = imported["ray.data._internal.execution.resource_manager"]
    actor_pool = imported[
        "ray.data._internal.execution.operators.actor_pool_map_operator"
    ]
    physical_operator = imported[
        "ray.data._internal.execution.interfaces.physical_operator"
    ]
    resource_admission = imported["ray.data._internal.execution.resource_admission"]
    gpu_shuffle = imported["ray.data._internal.gpu_shuffle.hash_shuffle"]
    parquet = imported["ray.data._internal.datasource.parquet_datasource"]

    if getattr(resource_manager, "RESOURCE_ADMISSION_CONTROL_VERSION", None) != 1:
        raise RuntimeError("installed Ray lacks resource admission capability v1")
    if not hasattr(resource_admission, "AdmissionKind"):
        raise RuntimeError("installed Ray lacks generic resource admission types")
    if {kind.value for kind in resource_admission.AdmissionKind} != {
        "transient",
        "elastic_pool",
        "fixed_gang",
    }:
        raise RuntimeError("installed Ray has unexpected resource admission kinds")
    for hook in (
        "resource_admission_spec",
        "has_internal_admission_demand",
        "apply_resource_admission_grant",
        "can_release_resource_admission",
    ):
        if not callable(getattr(physical_operator.PhysicalOperator, hook, None)):
            raise RuntimeError(f"installed Ray lacks PhysicalOperator.{hook}()")
    if not callable(
        actor_pool.ActorPoolMapOperator.__dict__.get("resource_admission_spec")
    ):
        raise RuntimeError("installed Ray lacks actor-pool admission adapter")
    if not callable(
        gpu_shuffle.GPUShuffleOperator.__dict__.get("resource_admission_spec")
    ):
        raise RuntimeError("installed Ray lacks GPU-shuffle gang admission adapter")
    if not isinstance(
        getattr(
            context_module.DataContext(), "_enable_resource_admission_control", None
        ),
        bool,
    ):
        raise RuntimeError("installed Ray lacks the resource admission rollback flag")

    context_has_h1 = hasattr(
        context_module.DataContext(), "custom_physical_optimizer_rule_classes"
    )
    parquet_has_h2 = hasattr(parquet.ParquetDatasource, "get_external_scan_descriptor")
    expected_hooks = layer == "hooked"
    if context_has_h1 is not expected_hooks or parquet_has_h2 is not expected_hooks:
        raise RuntimeError(
            f"installed {layer} layer has wrong hooks: "
            f"H1={context_has_h1}, H2={parquet_has_h2}"
        )


def _activate() -> None:
    test_root_value = os.environ.get(_TEST_ROOT_ENV)
    if test_root_value is None:
        return
    venv_value = os.environ.get(_VENV_ENV)
    layer = os.environ.get(_LAYER_ENV)
    if not venv_value or layer not in {"candidate", "hooked"}:
        raise RuntimeError("Ray acceptance bootstrap environment is incomplete")

    test_root = Path(test_root_value).resolve()
    venv = Path(venv_value).resolve()
    _assert_wheel_backed_production(venv, layer)
    _mount_test_package("ray.tests", test_root / "tests")
    _mount_test_package("ray.data.tests", test_root / "data" / "tests")


_activate()
