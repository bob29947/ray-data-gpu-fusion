"""Plan-local lowering and opportunistic fusion inside Ray's physical optimizer."""

from __future__ import annotations

import functools
import inspect
import math
import numbers
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional, Sequence

from ray_data_gpu_fusion._compat import (
    ActorPoolMapOperator,
    ActorPoolStrategy,
    DataContext,
    FuseOperators,
    InputDataBuffer,
    MapBatches,
    PhysicalOperator,
    PhysicalPlan,
    Read,
    Rule,
    RuntimeEnv,
    SetReadParallelismRule,
    TaskPoolStrategy,
    logical_operators,
    ray,
    set_input_dependencies,
    set_output_dependencies,
)
from ray_data_gpu_fusion.config import settings
from ray_data_gpu_fusion.operators import (
    BackendRefusal,
    CreationOptions,
    ExecutableGPUMapBatchesOperator,
    ExecutableGPUOperator,
    backend_can_materialize,
    materialize_operator,
)
from ray_data_gpu_fusion.parquet import (
    ParquetPlanningError,
    ReadParquetConfig,
    build_plan,
    descriptor_bundles,
    read_footer_with_retry,
    recognize_scan,
)
from ray_data_gpu_fusion.runtime import MAP_BATCHES, MapBatchesConfig
from ray_data_gpu_fusion.parquet import READ_PARQUET
from ray_data_gpu_fusion.specs import (
    FRAME_STREAM,
    PARQUET_WORK,
    CompositionError,
    ExecutionProfile,
    ExecutionRequirements,
    OperatorSpec,
    RetryPolicy,
    TransformSpec,
    compile_region,
)


@dataclass(frozen=True)
class Eligibility:
    accepted: bool
    reason: str = ""
    spec: Optional[OperatorSpec] = None
    creation_options: Optional[CreationOptions] = None
    source_factory: Any = None

    def __post_init__(self) -> None:
        if self.accepted:
            if self.reason or self.spec is None or self.creation_options is None:
                raise ValueError("accepted eligibility requires a spec and options")
        elif not self.reason:
            raise ValueError("declined eligibility requires a reason")


def _decline(reason: str) -> Eligibility:
    return Eligibility(False, reason)


def _is_sync_callable(value: Any) -> bool:
    if not callable(value):
        return False
    target = (
        value
        if inspect.isroutine(value) or isinstance(value, functools.partial)
        else value.__call__
    )
    return not inspect.iscoroutinefunction(target) and not inspect.isasyncgenfunction(
        target
    )


def _runtime_env_serialized(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, RuntimeEnv):
        runtime_env = value
    elif isinstance(value, dict):
        runtime_env = RuntimeEnv(**value)
    else:
        raise ValueError("runtime_env must be a dict or RuntimeEnv")
    return runtime_env.serialize()


def _cluster_gpu_count() -> int:
    resources = ray.cluster_resources() if ray.is_initialized() else {}
    return int(resources.get("GPU", 0) or 0)


def _retry_policy(context: DataContext, resources: dict[str, Any]) -> RetryPolicy:
    restarts = resources.pop("max_restarts", -1)
    task_retries = resources.pop("max_task_retries", -1 if restarts != 0 else 0)
    actor_errors = context.actor_task_retry_on_errors
    if not isinstance(actor_errors, bool):
        actor_errors = tuple(actor_errors)
    map_errors = context.retried_map_errors
    if not isinstance(map_errors, bool):
        map_errors = tuple(map_errors)
    return RetryPolicy(
        actor_max_restarts=int(restarts),
        actor_max_task_retries=int(task_retries),
        actor_task_retry_on_errors=actor_errors,
        actor_init_retry_on_errors=bool(context.actor_init_retry_on_errors),
        actor_init_max_retries=(
            int(context.actor_init_max_retries)
            if context.actor_init_retry_on_errors
            else 0
        ),
        retried_map_errors=map_errors,
        max_map_retries=(
            int(context.max_map_retries) if map_errors is not False else 0
        ),
        retried_io_errors=tuple(context.retried_io_errors),
    )


def _profile(
    compute: Any,
    raw_resources: dict[str, Any],
    context: DataContext,
    *,
    consumer: str,
) -> ExecutionRequirements:
    if not isinstance(compute, ActorPoolStrategy):
        raise ValueError(f"{consumer} requires ActorPoolStrategy")
    if compute.enable_true_multi_threading:
        raise ValueError(f"{consumer} does not support true multi-threading")
    if compute.max_tasks_in_flight_per_actor not in (None, 1):
        raise ValueError(f"{consumer} requires one task in flight per actor")

    maximum = compute.max_size
    if maximum == float("inf"):
        maximum = _cluster_gpu_count()
        if maximum <= 0:
            raise ValueError(f"{consumer} found no GPUs in Ray cluster resources")
    if (
        not isinstance(maximum, numbers.Integral)
        or isinstance(maximum, bool)
        or maximum <= 0
    ):
        raise ValueError(f"{consumer} requires a finite positive maximum pool size")
    minimum = min(int(compute.min_size), int(maximum))
    initial = min(int(compute.initial_size), int(maximum))

    resources = dict(raw_resources)
    max_concurrency = resources.pop("max_concurrency", 1)
    num_gpus = resources.pop("num_gpus", None)
    num_cpus = resources.pop("num_cpus", 1)
    memory = resources.pop("memory", None)
    custom = resources.pop("resources", None) or {}
    accelerator = resources.pop("accelerator_type", None)
    runtime_env = resources.pop("runtime_env", None)
    scheduling = resources.pop("scheduling_strategy", None)
    labels = resources.pop("label_selector", None) or {}
    retry = _retry_policy(context, resources)

    if max_concurrency != 1:
        raise ValueError(f"{consumer} requires max_concurrency=1")
    if (
        not isinstance(num_gpus, numbers.Real)
        or isinstance(num_gpus, bool)
        or float(num_gpus) != 1.0
    ):
        raise ValueError(f"{consumer} requires exactly num_gpus=1")
    if (
        not isinstance(num_cpus, numbers.Real)
        or isinstance(num_cpus, bool)
        or not math.isfinite(float(num_cpus))
        or num_cpus < 0
    ):
        raise ValueError(f"{consumer} num_cpus must be finite and nonnegative")
    if memory is not None and (
        not isinstance(memory, numbers.Real)
        or isinstance(memory, bool)
        or not math.isfinite(float(memory))
        or memory < 0
    ):
        raise ValueError(f"{consumer} memory must be finite and nonnegative")
    if not isinstance(custom, dict) or any(
        not isinstance(name, str)
        or not isinstance(value, numbers.Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
        for name, value in custom.items()
    ):
        raise ValueError(f"{consumer} has invalid custom resources")
    if accelerator is not None and (
        not isinstance(accelerator, str) or not accelerator
    ):
        raise ValueError(f"{consumer} accelerator_type must be a nonempty string")
    if scheduling is not None and not isinstance(scheduling, str):
        raise ValueError(f"{consumer} supports canonical string scheduling only")
    if not isinstance(labels, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in labels.items()
    ):
        raise ValueError(f"{consumer} label_selector must map strings to strings")
    if resources:
        raise ValueError(
            f"{consumer} has unsupported actor options {sorted(resources)!r}"
        )

    return ExecutionRequirements(
        profile=ExecutionProfile(
            min_workers=minimum,
            initial_workers=initial,
            max_workers=int(maximum),
            num_cpus=float(num_cpus),
            num_gpus=1.0,
            memory=float(memory) if memory is not None else None,
            custom_resources=tuple(
                sorted((key, float(value)) for key, value in custom.items())
            ),
            accelerator_type=accelerator,
            scheduling_strategy=scheduling,
            label_selector=tuple(sorted(labels.items())),
            runtime_env_serialized=_runtime_env_serialized(runtime_env),
            retry_policy=retry,
        )
    )


def map_eligibility(logical_op: Any, context: DataContext) -> Eligibility:
    if type(logical_op) is not MapBatches:
        return _decline("not an exact MapBatches logical operator")
    if not _is_sync_callable(logical_op.fn):
        return _decline("MapBatches UDF is not synchronous")
    if logical_op.batch_format != "cudf":
        return _decline("MapBatches batch_format is not 'cudf'")
    if (
        not isinstance(logical_op.batch_size, numbers.Integral)
        or isinstance(logical_op.batch_size, bool)
        or logical_op.batch_size <= 0
    ):
        return _decline("MapBatches batch_size is not a positive integer")
    if logical_op.per_block_limit is not None:
        return _decline("MapBatches per-block limits are unsupported")
    if logical_op.ray_remote_args_fn is not None:
        return _decline("MapBatches dynamic remote arguments are unsupported")
    try:
        execution = _profile(
            logical_op.compute,
            logical_op.ray_remote_args,
            context,
            consumer=f"MapBatches({logical_op.name})",
        )
        config = MapBatchesConfig(
            udf=logical_op.fn,
            udf_is_class=inspect.isclass(logical_op.fn),
            batch_size=int(logical_op.batch_size),
            zero_copy_batch=bool(logical_op.zero_copy_batch),
            fn_args=tuple(logical_op.fn_args or ()),
            fn_kwargs=tuple((logical_op.fn_kwargs or {}).items()),
            constructor_args=tuple(logical_op.fn_constructor_args or ()),
            constructor_kwargs=tuple((logical_op.fn_constructor_kwargs or {}).items()),
        )
    except (TypeError, ValueError) as error:
        return _decline(str(error))
    spec = OperatorSpec(
        (
            TransformSpec(
                MAP_BATCHES,
                FRAME_STREAM,
                FRAME_STREAM,
                config=config,
            ),
        ),
        execution,
    )
    return Eligibility(
        True,
        spec=spec,
        creation_options=CreationOptions(
            name="GPUMapBatches",
            min_rows_per_bundle=logical_op.min_rows_per_bundled_input,
            target_max_block_size_override=context.target_max_block_size,
        ),
    )


def _read_has_explicit_profile(logical_op: Read) -> bool:
    compute = logical_op.compute
    explicit_compute = isinstance(compute, ActorPoolStrategy) or (
        isinstance(compute, TaskPoolStrategy) and compute.size is not None
    )
    meaningful_args = {
        key
        for key, value in logical_op.ray_remote_args.items()
        if not (key == "scheduling_strategy" and value == "SPREAD")
    }
    return explicit_compute or bool(meaningful_args)


def _read_execution(
    logical_op: Read,
    context: DataContext,
    inherited: Optional[ExecutionRequirements],
) -> ExecutionRequirements:
    if inherited is not None and not _read_has_explicit_profile(logical_op):
        return inherited
    compute = logical_op.compute
    if isinstance(compute, TaskPoolStrategy) and compute.size is not None:
        actor_compute = ActorPoolStrategy(size=int(compute.size))
    elif isinstance(compute, ActorPoolStrategy):
        actor_compute = compute
    else:
        available = _cluster_gpu_count()
        if available <= 0:
            raise ValueError("Parquet Read found no GPUs in Ray cluster resources")
        actor_compute = ActorPoolStrategy(
            min_size=1, initial_size=1, max_size=available
        )
    resources = dict(logical_op.ray_remote_args)
    if resources.get("scheduling_strategy") == "SPREAD":
        resources.pop("scheduling_strategy")
    resources.setdefault("num_cpus", 1)
    resources.setdefault("num_gpus", 1)
    return _profile(actor_compute, resources, context, consumer="Parquet Read")


def read_eligibility(
    logical_op: Any,
    context: DataContext,
    inherited: Optional[ExecutionRequirements] = None,
) -> Eligibility:
    recognition = recognize_scan(logical_op)
    if not recognition.accepted:
        return _decline(recognition.reason or "Parquet scan was declined")
    if context.execution_options.preserve_order:
        # The Phase-0 row-group planner balances work by size, which does not
        # preserve source order. Keep Ray's stock reader whenever ordered
        # execution was explicitly requested rather than silently changing it.
        return _decline("Parquet Read preserve_order is unsupported")
    descriptor = recognition.descriptor
    assert descriptor is not None
    try:
        execution = _read_execution(logical_op, context, inherited)
        profile = execution.profile
        assert profile is not None
        bootstrap = dict(
            profile.actor_bootstrap,
            source_kind=descriptor.source_kind,
            region=descriptor.region or "",
        )
        execution = replace(
            execution,
            profile=replace(profile, actor_bootstrap=tuple(sorted(bootstrap.items()))),
        )
        footer = read_footer_with_retry(descriptor, tuple(context.retried_io_errors))
    except ParquetPlanningError as error:
        return _decline(f"Parquet footer is unsupported ({error})")
    except (TypeError, ValueError) as error:
        return _decline(str(error))

    detected = logical_op.get_detected_parallelism()
    target_tasks = int(detected or logical_op.num_outputs or 1)

    def source_factory(_target_max_block_size: int) -> list[Any]:
        return descriptor_bundles(build_plan(footer, target_tasks).work)

    spec = OperatorSpec(
        (
            TransformSpec(
                READ_PARQUET,
                PARQUET_WORK,
                FRAME_STREAM,
                config=ReadParquetConfig(
                    descriptor.source_kind,
                    descriptor.region,
                    descriptor.projection,
                    tuple(context.retried_io_errors),
                ),
            ),
        ),
        execution,
    )
    return Eligibility(
        True,
        spec=spec,
        creation_options=CreationOptions(
            name="GPUReadParquet",
            min_rows_per_bundle=1,
            target_max_block_size_override=context.target_max_block_size,
        ),
        source_factory=source_factory,
    )


def _reachable(root: PhysicalOperator) -> tuple[PhysicalOperator, ...]:
    visited: set[PhysicalOperator] = set()
    result = []
    stack = [root]
    while stack:
        operator = stack.pop()
        if operator in visited:
            continue
        visited.add(operator)
        result.append(operator)
        stack.extend(operator.input_dependencies)
    return tuple(result)


def _consumers(operators: Iterable[PhysicalOperator]):
    result: dict[PhysicalOperator, list[PhysicalOperator]] = defaultdict(list)
    for operator in operators:
        for dependency in operator.input_dependencies:
            result[dependency].append(operator)
    return result


def _rebuild_outputs(root: PhysicalOperator) -> None:
    operators = _reachable(root)
    for operator in operators:
        set_output_dependencies(operator, ())
    for operator in operators:
        for dependency in operator.input_dependencies:
            if operator not in dependency._output_dependencies:
                dependency._output_dependencies.append(operator)


def _configure_downstream_stock_gpu_actors(root: PhysicalOperator) -> None:
    """Prevent an unsupported downstream GPU pool from starving its ancestor."""

    memo: dict[PhysicalOperator, bool] = {}

    def has_plugin_ancestor(operator: PhysicalOperator) -> bool:
        if operator in memo:
            return memo[operator]
        value = any(
            isinstance(dependency, ExecutableGPUOperator)
            or has_plugin_ancestor(dependency)
            for dependency in operator.input_dependencies
        )
        memo[operator] = value
        return value

    for operator in _reachable(root):
        if isinstance(operator, ExecutableGPUOperator):
            continue
        if not isinstance(operator, ActorPoolMapOperator):
            continue
        remote = getattr(operator, "_ray_remote_args", {})
        if float(remote.get("num_gpus", 0) or 0) <= 0 or not has_plugin_ancestor(
            operator
        ):
            continue
        configure = getattr(operator, "configure_demand_driven_start", None)
        if not callable(configure):
            raise RuntimeError(
                "patched Ray ActorPoolMapOperator lacks "
                "configure_demand_driven_start()"
            )
        configure(
            wait_for_upstream_deferred_operators=True,
            release_idle_actors_on_completion=True,
        )


class LowerClosedGPUOperators(Rule):
    """Replace eligible stock nodes with independently executable GPU nodes."""

    @classmethod
    def dependencies(cls):
        return [SetReadParallelismRule]

    @classmethod
    def dependents(cls):
        return [FuseClosedGPUOperators]

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        if not settings(plan.context).enabled:
            return plan
        original_ops = _reachable(plan.dag)
        consumers = _consumers(original_ops)
        op_map = plan.op_map.copy()
        memo: dict[PhysicalOperator, PhysicalOperator] = {}

        def rewrite(operator: PhysicalOperator) -> PhysicalOperator:
            if operator in memo:
                return memo[operator]
            if isinstance(operator, ExecutableGPUOperator):
                memo[operator] = operator
                return operator
            rewritten_inputs = tuple(
                rewrite(value) for value in operator.input_dependencies
            )
            logical = plan.op_map.get(operator)
            replacement: Optional[PhysicalOperator] = None
            eligibility: Optional[Eligibility] = None

            if type(logical) is MapBatches and len(rewritten_inputs) == 1:
                eligibility = map_eligibility(logical, plan.context)
                if eligibility.accepted:
                    assert eligibility.spec is not None
                    try:
                        replacement = ExecutableGPUMapBatchesOperator(
                            operator,
                            rewritten_inputs[0],
                            eligibility.spec,
                            plan.context,
                            eligibility.creation_options,
                        )
                    except (TypeError, ValueError) as error:
                        # Construction-time option validation is still a
                        # planning capability check. Preserve the exact stock
                        # operator just as a backend factory refusal would.
                        eligibility = _decline(str(error))
            elif type(logical) is Read and hasattr(
                operator, "get_additional_split_factor"
            ):
                # A logical source has no logical inputs, but stock Ray lowers
                # it to ``InputDataBuffer -> MapOperator``.  Replace that whole
                # physical closure at the MapOperator. The InputDataBuffer is
                # also mapped to the logical Read in Ray's op_map, so the
                # method check is part of recognizing the physical shape.
                additional_split_factor = operator.get_additional_split_factor()
                if additional_split_factor > 1:
                    # SetReadParallelismRule uses this stock physical split to
                    # honor requested read parallelism. The direct row-group
                    # reader cannot yet reproduce a split within one row group,
                    # so retain the exact stock node instead of changing the
                    # number of output blocks.
                    eligibility = _decline(
                        "Parquet Read additional_split_factor="
                        f"{additional_split_factor} is unsupported"
                    )
                else:
                    inherited = None
                    downstream = consumers.get(operator, ())
                    if len(downstream) == 1:
                        downstream_logical = plan.op_map.get(downstream[0])
                        downstream_eligibility = map_eligibility(
                            downstream_logical, plan.context
                        )
                        if downstream_eligibility.accepted:
                            assert downstream_eligibility.spec is not None
                            inherited = downstream_eligibility.spec.execution
                    eligibility = read_eligibility(logical, plan.context, inherited)
                if eligibility.accepted:
                    source = InputDataBuffer(
                        plan.context,
                        input_data_factory=eligibility.source_factory,
                    )
                    source.set_logical_operators(logical)
                    assert eligibility.spec is not None
                    try:
                        replacement = materialize_operator(
                            eligibility.spec,
                            source,
                            plan.context,
                            eligibility.creation_options,
                        )
                    except BackendRefusal as error:
                        eligibility = _decline(str(error))
                    else:
                        op_map[source] = logical

            if replacement is None:
                set_input_dependencies(operator, rewritten_inputs)
                if eligibility is not None and not eligibility.accepted:
                    operator._ray_data_gpu_fusion_decline_reason = eligibility.reason
                memo[operator] = operator
                return operator

            replacement.set_logical_operators(logical)
            op_map.pop(operator, None)
            op_map[replacement] = logical
            memo[operator] = replacement
            return replacement

        root = rewrite(plan.dag)
        _rebuild_outputs(root)
        _configure_downstream_stock_gpu_actors(root)
        reachable = set(_reachable(root))
        op_map = {
            operator: logical
            for operator, logical in op_map.items()
            if operator in reachable
        }
        return PhysicalPlan(root, op_map, plan.context)


def _select_regions(
    operators: Sequence[PhysicalOperator],
) -> tuple[tuple[ExecutableGPUOperator, ...], ...]:
    candidates = tuple(
        operator
        for operator in operators
        if isinstance(operator, ExecutableGPUOperator)
    )
    candidate_set = set(candidates)
    consumers = _consumers(operators)
    successor: dict[ExecutableGPUOperator, ExecutableGPUOperator] = {}
    predecessor: dict[ExecutableGPUOperator, ExecutableGPUOperator] = {}
    for candidate in candidates:
        downstream = consumers.get(candidate, ())
        if len(downstream) != 1:
            continue
        consumer = downstream[0]
        if (
            consumer in candidate_set
            and len(consumer.input_dependencies) == 1
            and consumer.input_dependencies[0] is candidate
        ):
            successor[candidate] = consumer
            predecessor[consumer] = candidate

    components = []
    visited = set()
    for start in candidates:
        if start in predecessor:
            continue
        component = []
        cursor = start
        while cursor not in visited:
            visited.add(cursor)
            component.append(cursor)
            if cursor not in successor:
                break
            cursor = successor[cursor]
        components.append(tuple(component))

    selected = []
    for component in components:
        index = 0
        while index < len(component) - 1:
            last_valid = None
            for end in range(index + 1, len(component)):
                try:
                    core = compile_region(
                        tuple(
                            node.gpu_fusion_spec for node in component[index : end + 1]
                        )
                    )
                except CompositionError:
                    break
                if backend_can_materialize(core):
                    last_valid = end
            if last_valid is None:
                index += 1
            else:
                selected.append(tuple(component[index : last_valid + 1]))
                index = last_valid + 1
    return tuple(selected)


def _combined_options(
    region: Sequence[ExecutableGPUOperator], core: OperatorSpec
) -> CreationOptions:
    options = region[0].gpu_creation_options
    for node in region[1:]:
        options = options.combine(node.gpu_creation_options, core)
    return options


class FuseClosedGPUOperators(Rule):
    """Fuse compatible linear candidates; refusal preserves the runnable DAG."""

    @classmethod
    def dependencies(cls):
        return [LowerClosedGPUOperators]

    @classmethod
    def dependents(cls):
        return [FuseOperators]

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        current = settings(plan.context)
        if not current.enabled or not current.fusion:
            return plan
        operators = _reachable(plan.dag)
        regions = _select_regions(operators)
        if not regions:
            return plan
        by_end = {region[-1]: region for region in regions}
        op_map = plan.op_map.copy()
        memo: dict[PhysicalOperator, PhysicalOperator] = {}

        def rewrite(operator: PhysicalOperator) -> PhysicalOperator:
            if operator in memo:
                return memo[operator]
            region = by_end.get(operator)
            if region is not None and len(region[0].input_dependencies) == 1:
                external_input = rewrite(region[0].input_dependencies[0])
                try:
                    core = compile_region(
                        tuple(node.gpu_fusion_spec for node in region)
                    )
                    fused = materialize_operator(
                        core,
                        external_input,
                        plan.context,
                        _combined_options(region, core),
                    )
                except (BackendRefusal, CompositionError, ValueError):
                    fused = None
                if fused is not None:
                    lineage = []
                    for node in region:
                        node_lineage = logical_operators(node)
                        if not node_lineage and node in op_map:
                            node_lineage = (op_map[node],)
                        lineage.extend(node_lineage)
                    if lineage:
                        fused.set_logical_operators(*lineage)
                    downstream_logical = op_map.get(region[-1])
                    for node in region:
                        op_map.pop(node, None)
                        memo[node] = fused
                    if downstream_logical is not None:
                        op_map[fused] = downstream_logical
                    return fused

            set_input_dependencies(
                operator, tuple(rewrite(value) for value in operator.input_dependencies)
            )
            memo[operator] = operator
            return operator

        root = rewrite(plan.dag)
        _rebuild_outputs(root)
        _configure_downstream_stock_gpu_actors(root)
        reachable = set(_reachable(root))
        op_map = {
            operator: logical
            for operator, logical in op_map.items()
            if operator in reachable
        }
        return PhysicalPlan(root, op_map, plan.context)


__all__ = [
    "Eligibility",
    "FuseClosedGPUOperators",
    "LowerClosedGPUOperators",
    "map_eligibility",
    "read_eligibility",
]
