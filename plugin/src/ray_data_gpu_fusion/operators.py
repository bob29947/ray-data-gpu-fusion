"""Executable Ray physical operators and the actor-pool backend factory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol

from ray_data_gpu_fusion._compat import (
    ActorPoolMapOperator,
    ActorPoolStrategy,
    BlockMapTransformFn,
    DataContext,
    MapTransformer,
    PhysicalOperator,
    RuntimeEnv,
    demand_driven_actor_kwargs,
    stock_map_transformer,
)
from ray_data_gpu_fusion.runtime import (
    EXPORT_FRAME_STREAM,
    IMPORT_PARQUET_WORK,
    IMPORT_RAY_BLOCKS,
    RegionBlockTransform,
    ensure_builtin_runtimes,
    resolve_runtime,
    supports_runtime,
)
from ray_data_gpu_fusion.specs import (
    FRAME_STREAM,
    PARQUET_WORK,
    RAY_BLOCK_STREAM,
    ExecutionRequirements,
    OperatorSpec,
    TransformSpec,
)


class BackendRefusal(ValueError):
    def __init__(self, backend: str, detail: str):
        self.backend = backend
        self.detail = detail
        super().__init__(f"backend={backend!r}; {detail}")


@dataclass(frozen=True)
class CreationOptions:
    name: Optional[str] = None
    min_rows_per_bundle: Optional[int] = None
    target_max_block_size_override: Optional[int] = None

    def combine(
        self, downstream: "CreationOptions", spec: OperatorSpec
    ) -> "CreationOptions":
        target = (
            downstream.target_max_block_size_override
            if downstream.target_max_block_size_override is not None
            else self.target_max_block_size_override
        )
        return replace(
            self,
            name=_gpu_name("GPU", spec),
            target_max_block_size_override=target,
        )


def _gpu_name(prefix: str, spec: OperatorSpec) -> str:
    return f"{prefix}[{','.join(transform.kind for transform in spec.transforms)}]"


class GPUCandidate:
    """A physical node carrying a boundary-free, independently runnable spec."""

    @property
    def gpu_fusion_spec(self) -> OperatorSpec:
        return self._gpu_fusion_spec

    @property
    def gpu_creation_options(self) -> CreationOptions:
        return self._gpu_creation_options

    @property
    def execution_backend_name(self) -> str:
        return "|".join(sorted(self.gpu_fusion_spec.execution.backend_keys))


class ExecutableGPUOperator(GPUCandidate):
    """Marker: correctness does not depend on this candidate being fused."""


def _compute(spec: OperatorSpec) -> ActorPoolStrategy:
    profile = spec.profile
    return ActorPoolStrategy(
        min_size=profile.min_workers,
        initial_size=profile.initial_workers,
        max_size=profile.max_workers,
        max_tasks_in_flight_per_actor=profile.max_tasks_in_flight_per_worker,
        enable_true_multi_threading=False,
    )


def ray_remote_args(spec: OperatorSpec) -> dict[str, Any]:
    profile = spec.profile
    retry = profile.retry_policy
    args: dict[str, Any] = {
        "num_cpus": profile.num_cpus,
        "num_gpus": profile.num_gpus,
        "max_concurrency": 1,
        "max_restarts": retry.actor_max_restarts,
        "max_task_retries": retry.actor_max_task_retries,
    }
    if profile.memory is not None:
        args["memory"] = profile.memory
    if profile.custom_resources:
        args["resources"] = dict(profile.custom_resources)
    if profile.accelerator_type is not None:
        args["accelerator_type"] = profile.accelerator_type
    if profile.scheduling_strategy is not None:
        args["scheduling_strategy"] = profile.scheduling_strategy
    if profile.label_selector:
        args["label_selector"] = dict(profile.label_selector)
    if profile.runtime_env_serialized is not None:
        args["runtime_env"] = RuntimeEnv.deserialize(profile.runtime_env_serialized)
    return args


class ActorPoolGPUOperator(ActorPoolMapOperator, ExecutableGPUOperator):
    """One Ray actor-pool shell around a closed native GPU program."""

    def __init__(
        self,
        native_spec: OperatorSpec,
        execution_spec: OperatorSpec,
        input_op: PhysicalOperator,
        data_context: DataContext,
        creation_options: Optional[CreationOptions] = None,
    ) -> None:
        creation_options = creation_options or CreationOptions()
        if (
            execution_spec.input_payload != RAY_BLOCK_STREAM
            or execution_spec.output_payload != RAY_BLOCK_STREAM
        ):
            raise ValueError("actor-pool regions require Ray-block endpoints")
        if not execution_spec.execution.resolved:
            raise ValueError("actor-pool execution requirements are unresolved")
        ensure_builtin_runtimes()
        registrations = tuple(resolve_runtime(t) for t in execution_spec.transforms)
        transform = RegionBlockTransform(execution_spec, registrations)
        transformer = MapTransformer(
            [BlockMapTransformFn(transform)], init_fn=transform.initialize
        )
        self._gpu_fusion_spec = native_spec
        self._gpu_execution_spec = execution_spec
        self._gpu_creation_options = creation_options
        super().__init__(
            map_transformer=transformer,
            input_op=input_op,
            data_context=data_context,
            compute_strategy=_compute(execution_spec),
            name=creation_options.name or _gpu_name("GPU", native_spec),
            min_rows_per_bundle=creation_options.min_rows_per_bundle,
            supports_fusion=False,
            ray_remote_args=ray_remote_args(execution_spec),
            target_max_block_size_override=(
                creation_options.target_max_block_size_override
            ),
            **demand_driven_actor_kwargs(),
        )

    @property
    def gpu_execution_spec(self) -> OperatorSpec:
        return self._gpu_execution_spec


class ExecutableGPUMapBatchesOperator(ActorPoolMapOperator, ExecutableGPUOperator):
    """Standalone MapBatches using Ray's already-planned stock transformer."""

    def __init__(
        self,
        stock_operator: PhysicalOperator,
        input_op: PhysicalOperator,
        native_spec: OperatorSpec,
        data_context: DataContext,
        creation_options: Optional[CreationOptions] = None,
    ) -> None:
        creation_options = creation_options or CreationOptions()
        self._gpu_fusion_spec = native_spec
        self._gpu_creation_options = creation_options
        super().__init__(
            map_transformer=stock_map_transformer(stock_operator),
            input_op=input_op,
            data_context=data_context,
            compute_strategy=_compute(native_spec),
            name=creation_options.name or stock_operator.name,
            min_rows_per_bundle=creation_options.min_rows_per_bundle,
            supports_fusion=False,
            ray_remote_args=ray_remote_args(native_spec),
            target_max_block_size_override=(
                creation_options.target_max_block_size_override
            ),
            default_logical_memory_enabled=getattr(
                data_context, "default_map_logical_memory_enabled", False
            ),
            **demand_driven_actor_kwargs(),
        )


class BackendFactory(Protocol):
    def prepare(self, core_spec: OperatorSpec) -> OperatorSpec: ...
    def supports(self, execution_spec: OperatorSpec) -> bool: ...
    def create(
        self,
        core_spec: OperatorSpec,
        execution_spec: OperatorSpec,
        input_op: PhysicalOperator,
        data_context: DataContext,
        creation_options: Optional[CreationOptions] = None,
    ) -> PhysicalOperator: ...


_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(
    name: str, factory: BackendFactory, *, replace_existing: bool = False
) -> None:
    if not name:
        raise ValueError("backend name must be nonempty")
    if name in _FACTORIES and not replace_existing:
        raise ValueError(f"backend already registered: {name!r}")
    _FACTORIES[name] = factory


def unregister_backend(
    name: str, *, expected: Optional[BackendFactory] = None
) -> Optional[BackendFactory]:
    factory = _FACTORIES.get(name)
    if factory is None:
        return None
    if expected is not None and factory is not expected:
        raise ValueError("registered backend is not the expected implementation")
    return _FACTORIES.pop(name)


def get_backend(name: str) -> Optional[BackendFactory]:
    return _FACTORIES.get(name)


def _boundary(
    kind: str,
    input_payload: Any,
    output_payload: Any,
    execution: ExecutionRequirements,
) -> TransformSpec:
    return TransformSpec(kind, input_payload, output_payload)


class ActorPoolBackendFactory:
    name = "actor_pool"

    def prepare(self, core_spec: OperatorSpec) -> OperatorSpec:
        if not core_spec.execution.resolved:
            raise BackendRefusal(self.name, "execution profile is unresolved")
        if core_spec.input_payload == PARQUET_WORK:
            ingress = _boundary(
                IMPORT_PARQUET_WORK,
                RAY_BLOCK_STREAM,
                PARQUET_WORK,
                core_spec.execution,
            )
        elif core_spec.input_payload == FRAME_STREAM:
            ingress = _boundary(
                IMPORT_RAY_BLOCKS,
                RAY_BLOCK_STREAM,
                FRAME_STREAM,
                core_spec.execution,
            )
        else:
            raise BackendRefusal(
                self.name,
                f"no materialized ingress for {core_spec.input_payload.key!r}",
            )
        if core_spec.output_payload != FRAME_STREAM:
            raise BackendRefusal(
                self.name,
                f"no materialized egress for {core_spec.output_payload.key!r}",
            )
        egress = _boundary(
            EXPORT_FRAME_STREAM,
            FRAME_STREAM,
            RAY_BLOCK_STREAM,
            core_spec.execution,
        )
        return OperatorSpec(
            (ingress, *core_spec.transforms, egress), core_spec.execution
        )

    def supports(self, execution_spec: OperatorSpec) -> bool:
        return (
            execution_spec.execution.resolved
            and execution_spec.execution.backend_name == self.name
            and execution_spec.input_payload == RAY_BLOCK_STREAM
            and execution_spec.output_payload == RAY_BLOCK_STREAM
            and all(supports_runtime(t) for t in execution_spec.transforms)
        )

    def create(
        self,
        core_spec: OperatorSpec,
        execution_spec: OperatorSpec,
        input_op: PhysicalOperator,
        data_context: DataContext,
        creation_options: Optional[CreationOptions] = None,
    ) -> PhysicalOperator:
        return ActorPoolGPUOperator(
            core_spec,
            execution_spec,
            input_op,
            data_context,
            creation_options,
        )


ACTOR_POOL_BACKEND = ActorPoolBackendFactory()
_FACTORIES.setdefault(ACTOR_POOL_BACKEND.name, ACTOR_POOL_BACKEND)


def backend_can_materialize(core_spec: OperatorSpec) -> bool:
    for name in sorted(core_spec.execution.backend_keys):
        factory = get_backend(name)
        if factory is None:
            continue
        try:
            selected = replace(
                core_spec,
                execution=replace(core_spec.execution, backend_keys=frozenset((name,))),
            )
            execution_spec = factory.prepare(selected)
            if factory.supports(execution_spec):
                return True
        except (BackendRefusal, ValueError):
            continue
    return False


def materialize_operator(
    core_spec: OperatorSpec,
    input_op: PhysicalOperator,
    data_context: DataContext,
    creation_options: Optional[CreationOptions] = None,
) -> PhysicalOperator:
    refusals = []
    for name in sorted(core_spec.execution.backend_keys):
        factory = get_backend(name)
        if factory is None:
            refusals.append(f"{name}: not registered")
            continue
        candidate: Optional[PhysicalOperator] = None
        try:
            selected = replace(
                core_spec,
                execution=replace(core_spec.execution, backend_keys=frozenset((name,))),
            )
            execution_spec = factory.prepare(selected)
            if not factory.supports(execution_spec):
                raise BackendRefusal(name, "prepared region is unsupported")
            candidate = factory.create(
                selected,
                execution_spec,
                input_op,
                data_context,
                creation_options,
            )
            if not isinstance(candidate, PhysicalOperator) or not isinstance(
                candidate, ExecutableGPUOperator
            ):
                raise BackendRefusal(name, "factory returned a non-executable node")
            return candidate
        except (BackendRefusal, TypeError, ValueError) as error:
            if candidate is not None:
                for dependency in candidate.input_dependencies:
                    dependency._output_dependencies = [
                        value
                        for value in dependency._output_dependencies
                        if value is not candidate
                    ]
            # These exception classes are the factory's documented planning
            # validation surface.  Normalize them to a backend refusal so the
            # lowering rule can retain the exact stock node.  RuntimeError and
            # other unexpected failures still propagate as programming errors.
            refusals.append(str(error))
    raise BackendRefusal(
        "|".join(sorted(core_spec.execution.backend_keys)), "; ".join(refusals)
    )


__all__ = [
    "ACTOR_POOL_BACKEND",
    "ActorPoolGPUOperator",
    "BackendRefusal",
    "CreationOptions",
    "ExecutableGPUMapBatchesOperator",
    "ExecutableGPUOperator",
    "GPUCandidate",
    "backend_can_materialize",
    "get_backend",
    "materialize_operator",
    "ray_remote_args",
    "register_backend",
    "unregister_backend",
]
