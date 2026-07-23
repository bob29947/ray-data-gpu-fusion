"""All imports and structural assumptions tied to the pinned Ray revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Type

import ray
from ray.data.block import Block, BlockAccessor, CallableClass
from ray.data.context import DataContext
from ray.data.exceptions import UserCodeException
from ray.runtime_env import RuntimeEnv

# Keep every import from ``ray.data._internal`` in this one module.  The rest of
# the plugin imports aliases from here, making a future Ray-version shim local.
from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
from ray.data._internal.execution.resource_admission import (
    ResourceAdmissionGrant,
    ResourceAdmissionSpec,
)
from ray.data._internal.execution.interfaces import (
    BlockEntry,
    PhysicalOperator,
    RefBundle,
    TaskContext,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators import MapBatches, Read
from ray.data._internal.logical.optimizers import (
    LogicalOptimizer,
    PhysicalOptimizer,
)
from ray.data._internal.logical.rules.operator_fusion import FuseOperators
from ray.data._internal.logical.rules.set_read_parallelism import (
    SetReadParallelismRule,
)
from ray.data._internal.planner import create_planner
from ray.data._internal.util import explain_plan as _ray_explain_plan
from ray.data._internal.util import iterate_with_retry
from ray.util.rpdb import _is_ray_debugger_post_mortem_enabled

PINNED_RAY_COMMIT = "2741c6461d2bd3e5ff114af67be7a1190453dadd"
PHYSICAL_RULE_CLASSES_ATTR = "custom_physical_optimizer_rule_classes"
RESOURCE_ADMISSION_CONTEXT_ATTR = "_enable_resource_admission_control"
RESOURCE_ADMISSION_SPEC_FIELDS = (
    "minimum_resources",
    "unit_resources",
    "min_units",
    "max_units",
)
RESOURCE_ADMISSION_GRANT_FIELDS = ("max_units", "may_submit")


class RayCompatibilityError(RuntimeError):
    """The installed Ray does not provide the pinned Phase-0 contract."""


@dataclass(frozen=True)
class CompatibilityInfo:
    adapter: str
    ray_version: str
    ray_commit: str | None
    supported: bool
    missing_seams: tuple[str, ...] = ()


def compatibility(context: DataContext | None = None) -> CompatibilityInfo:
    """Return compatibility facts without raising or changing Ray state."""

    missing: list[str] = []
    commit = getattr(ray, "__commit__", None)
    if commit != PINNED_RAY_COMMIT:
        missing.append("pinned_ray_commit")
    selected = DataContext.get_current() if context is None else context
    if not hasattr(selected, PHYSICAL_RULE_CLASSES_ATTR):
        missing.append("plan_local_physical_rules")
    if (
        tuple(ResourceAdmissionSpec.__dataclass_fields__)
        != RESOURCE_ADMISSION_SPEC_FIELDS
    ):
        missing.append("aggregate_resource_admission_spec")
    if (
        tuple(ResourceAdmissionGrant.__dataclass_fields__)
        != RESOURCE_ADMISSION_GRANT_FIELDS
    ):
        missing.append("aggregate_resource_admission_grant")
    if not callable(ActorPoolMapOperator.__dict__.get("resource_admission_spec")):
        missing.append("actor_pool_resource_admission_spec")
    if not hasattr(selected, RESOURCE_ADMISSION_CONTEXT_ATTR):
        missing.append("resource_admission_context")
    elif getattr(selected, RESOURCE_ADMISSION_CONTEXT_ATTR) is not True:
        missing.append("resource_admission_disabled")
    if not hasattr(ParquetDatasource, "get_external_scan_descriptor"):
        missing.append("parquet_external_scan_descriptor")
    return CompatibilityInfo(
        adapter="ray-2741c646-phase0-aggregate-admission",
        ray_version=str(getattr(ray, "__version__", "unknown")),
        ray_commit=commit,
        supported=not missing,
        missing_seams=tuple(missing),
    )


def verify_compatibility(context: DataContext | None = None) -> None:
    """Fail before mutating a context when Ray or either patch is missing."""

    commit = getattr(ray, "__commit__", None)
    if commit != PINNED_RAY_COMMIT:
        raise RayCompatibilityError(
            "ray-data-gpu-fusion 0.1 requires Ray commit "
            f"{PINNED_RAY_COMMIT}, but the installed Ray reports {commit!r}"
        )

    selected = DataContext.get_current() if context is None else context
    if not hasattr(selected, PHYSICAL_RULE_CLASSES_ATTR):
        raise RayCompatibilityError(
            "Ray is at the pinned commit but lacks the plan-local physical-rule "
            "patch (DataContext.custom_physical_optimizer_rule_classes)"
        )

    spec_fields = tuple(ResourceAdmissionSpec.__dataclass_fields__)
    if spec_fields != RESOURCE_ADMISSION_SPEC_FIELDS:
        raise RayCompatibilityError(
            "Ray lacks the aggregate resource admission specification; "
            f"expected {RESOURCE_ADMISSION_SPEC_FIELDS!r}, found {spec_fields!r}"
        )

    grant_fields = tuple(ResourceAdmissionGrant.__dataclass_fields__)
    if grant_fields != RESOURCE_ADMISSION_GRANT_FIELDS:
        raise RayCompatibilityError(
            "Ray lacks the aggregate resource admission grant; "
            f"expected {RESOURCE_ADMISSION_GRANT_FIELDS!r}, found {grant_fields!r}"
        )
    if not callable(ActorPoolMapOperator.__dict__.get("resource_admission_spec")):
        raise RayCompatibilityError(
            "Ray lacks ActorPoolMapOperator.resource_admission_spec()"
        )

    if not hasattr(selected, RESOURCE_ADMISSION_CONTEXT_ATTR):
        raise RayCompatibilityError(
            "Ray lacks DataContext._enable_resource_admission_control"
        )
    if getattr(selected, RESOURCE_ADMISSION_CONTEXT_ATTR) is not True:
        raise RayCompatibilityError(
            "DataContext._enable_resource_admission_control must be True"
        )

    if not hasattr(ParquetDatasource, "get_external_scan_descriptor"):
        raise RayCompatibilityError(
            "Ray lacks ParquetDatasource.get_external_scan_descriptor()"
        )


def require_elastic_resource_admission(operator: PhysicalOperator) -> None:
    """Require a plugin GPU region to participate as an elastic resource pool."""

    spec = operator.resource_admission_spec()
    if spec is None or spec.unit_resources is None:
        unit_resources = None if spec is None else spec.unit_resources
        raise RayCompatibilityError(
            f"{operator.name} must provide elastic resource admission; "
            f"found unit_resources={unit_resources!r}"
        )


def get_rule_classes(context: DataContext) -> list[Type[Rule]]:
    value = getattr(context, PHYSICAL_RULE_CLASSES_ATTR, None)
    if not isinstance(value, list):
        raise RayCompatibilityError(
            f"DataContext.{PHYSICAL_RULE_CLASSES_ATTR} must be a list"
        )
    return value


def set_rule_classes(context: DataContext, classes: Iterable[Type[Rule]]) -> None:
    setattr(context, PHYSICAL_RULE_CLASSES_ATTR, list(classes))


def stock_map_transformer(operator: PhysicalOperator) -> MapTransformer:
    transformer = getattr(operator, "_map_transformer", None)
    if not isinstance(transformer, MapTransformer):
        raise RayCompatibilityError(
            "the pinned stock MapBatches physical operator has no MapTransformer"
        )
    return transformer


def set_input_dependencies(
    operator: PhysicalOperator, inputs: Iterable[PhysicalOperator]
) -> None:
    operator._input_dependencies = list(inputs)


def set_output_dependencies(
    operator: PhysicalOperator, outputs: Iterable[PhysicalOperator]
) -> None:
    operator._output_dependencies = list(outputs)


def logical_operators(operator: PhysicalOperator) -> tuple[Any, ...]:
    return tuple(getattr(operator, "_logical_operators", ()))


def explain_plan(logical_plan: Any) -> str:
    return _ray_explain_plan(logical_plan)


def optimized_physical_plan(logical_plan: Any) -> PhysicalPlan:
    """Construct the same optimized physical plan Ray uses for execution."""

    optimized_logical = LogicalOptimizer().optimize(logical_plan)
    physical, _ = create_planner().plan(optimized_logical)
    return PhysicalOptimizer().optimize(physical)


__all__ = [
    "ActorPoolMapOperator",
    "ActorPoolStrategy",
    "Block",
    "BlockAccessor",
    "BlockEntry",
    "BlockMapTransformFn",
    "CallableClass",
    "CompatibilityInfo",
    "DataContext",
    "FuseOperators",
    "RESOURCE_ADMISSION_CONTEXT_ATTR",
    "RESOURCE_ADMISSION_GRANT_FIELDS",
    "RESOURCE_ADMISSION_SPEC_FIELDS",
    "InputDataBuffer",
    "LogicalOptimizer",
    "MapBatches",
    "MapTransformer",
    "ParquetDatasource",
    "PhysicalOperator",
    "PhysicalOptimizer",
    "PhysicalPlan",
    "RayCompatibilityError",
    "Read",
    "RefBundle",
    "Rule",
    "RuntimeEnv",
    "SetReadParallelismRule",
    "TaskContext",
    "TaskPoolStrategy",
    "UserCodeException",
    "_is_ray_debugger_post_mortem_enabled",
    "create_planner",
    "compatibility",
    "explain_plan",
    "get_rule_classes",
    "iterate_with_retry",
    "logical_operators",
    "optimized_physical_plan",
    "ray",
    "require_elastic_resource_admission",
    "set_input_dependencies",
    "set_output_dependencies",
    "set_rule_classes",
    "stock_map_transformer",
    "verify_compatibility",
]
