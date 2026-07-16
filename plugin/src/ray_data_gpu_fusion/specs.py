"""Immutable payload, execution, and composition contracts for GPU regions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, FrozenSet, Optional, Tuple, Union


@dataclass(frozen=True)
class PayloadKind:
    key: str
    materializable_boundary: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("payload key must be a nonempty string")


RAY_BLOCK_STREAM = PayloadKind("ray_block_stream", True)
PARQUET_WORK = PayloadKind("parquet_work")
FRAME_STREAM = PayloadKind("frame_stream")
DEVICE_HANDLE = PayloadKind("device_handle", True)  # Reserved for a later phase.

ActorTaskRetryOnErrors = Union[bool, Tuple[type[BaseException], ...]]
RetriedMapErrors = Union[bool, Tuple[str, ...]]


@dataclass(frozen=True)
class RetryPolicy:
    actor_max_restarts: int = -1
    actor_max_task_retries: int = -1
    actor_task_retry_on_errors: ActorTaskRetryOnErrors = False
    actor_init_retry_on_errors: bool = False
    actor_init_max_retries: int = 0
    retried_map_errors: RetriedMapErrors = False
    max_map_retries: int = 0
    retried_io_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.actor_max_restarts < -1 or self.actor_max_task_retries < -1:
            raise ValueError("actor retry counts must be -1 or nonnegative")
        if self.actor_init_max_retries < 0 or self.max_map_retries < 0:
            raise ValueError("map and initialization retries must be nonnegative")
        object.__setattr__(self, "retried_io_errors", tuple(self.retried_io_errors))


@dataclass(frozen=True)
class ExecutionProfile:
    runtime_id: str = "ray-data-gpu-fusion-v1"
    worker_lifecycle: str = "actor_pool"
    min_workers: int = 1
    initial_workers: int = 1
    max_workers: int = 1
    max_tasks_in_flight_per_worker: int = 1
    num_cpus: float = 1.0
    num_gpus: float = 1.0
    memory: Optional[float] = None
    custom_resources: tuple[tuple[str, float], ...] = ()
    accelerator_type: Optional[str] = None
    scheduling_strategy: Optional[str] = None
    label_selector: tuple[tuple[str, str], ...] = ()
    runtime_env_serialized: Optional[str] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    actor_bootstrap: tuple[tuple[str, str], ...] = (("allocator", "rmm-pool-v1"),)

    def __post_init__(self) -> None:
        for name in ("custom_resources", "label_selector", "actor_bootstrap"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.runtime_id or self.worker_lifecycle != "actor_pool":
            raise ValueError("Phase 0 requires a named actor-pool runtime")
        if self.min_workers < 1:
            raise ValueError("min_workers must be positive")
        if not self.min_workers <= self.initial_workers <= self.max_workers:
            raise ValueError("initial_workers must be within the pool bounds")
        if self.max_tasks_in_flight_per_worker != 1:
            raise ValueError("Phase 0 requires one task in flight per GPU actor")
        if self.num_gpus != 1.0:
            raise ValueError("Phase 0 requires exactly one GPU per actor")
        if len(dict(self.actor_bootstrap)) != len(self.actor_bootstrap):
            raise ValueError("actor_bootstrap contains duplicate keys")


@dataclass(frozen=True)
class ExecutionRequirements:
    backend_keys: FrozenSet[str] = frozenset(("actor_pool",))
    profile: Optional[ExecutionProfile] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_keys", frozenset(self.backend_keys))
        if not self.backend_keys:
            raise ValueError("at least one execution backend is required")

    @property
    def resolved(self) -> bool:
        return self.profile is not None and len(self.backend_keys) == 1

    @property
    def backend_name(self) -> str:
        if len(self.backend_keys) != 1:
            raise ValueError("execution requirements do not select one backend")
        return next(iter(self.backend_keys))


class CompositionReason(str, Enum):
    EXECUTION_INCOMPATIBLE = "EXECUTION_INCOMPATIBLE"
    REQUIREMENTS_UNSATISFIED = "REQUIREMENTS_UNSATISFIED"
    PAYLOAD_INCOMPATIBLE = "PAYLOAD_INCOMPATIBLE"
    MATERIALIZABLE_BOUNDARY = "MATERIALIZABLE_BOUNDARY"


class CompositionError(ValueError):
    def __init__(self, reason: CompositionReason, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


def merge_profiles(left: ExecutionProfile, right: ExecutionProfile) -> ExecutionProfile:
    fixed = (
        "runtime_id",
        "worker_lifecycle",
        "min_workers",
        "initial_workers",
        "max_workers",
        "max_tasks_in_flight_per_worker",
        "num_cpus",
        "num_gpus",
        "retry_policy",
    )
    for name in fixed:
        if getattr(left, name) != getattr(right, name):
            raise CompositionError(
                CompositionReason.EXECUTION_INCOMPATIBLE,
                f"execution profile field {name!r} differs",
            )

    def optional(name: str, empty_is_none: bool = False) -> Any:
        a, b = getattr(left, name), getattr(right, name)
        a_empty = a is None or (empty_is_none and not a)
        b_empty = b is None or (empty_is_none and not b)
        if a_empty:
            return b
        if b_empty or a == b:
            return a
        raise CompositionError(
            CompositionReason.EXECUTION_INCOMPATIBLE,
            f"execution profile field {name!r} differs",
        )

    bootstrap = dict(left.actor_bootstrap)
    for key, value in right.actor_bootstrap:
        if key in bootstrap and bootstrap[key] != value:
            raise CompositionError(
                CompositionReason.EXECUTION_INCOMPATIBLE,
                f"actor bootstrap field {key!r} differs",
            )
        bootstrap[key] = value
    return replace(
        left,
        memory=optional("memory"),
        custom_resources=optional("custom_resources", True),
        accelerator_type=optional("accelerator_type"),
        scheduling_strategy=optional("scheduling_strategy"),
        label_selector=optional("label_selector", True),
        runtime_env_serialized=optional("runtime_env_serialized"),
        actor_bootstrap=tuple(sorted(bootstrap.items())),
    )


def merge_execution(
    left: ExecutionRequirements, right: ExecutionRequirements
) -> ExecutionRequirements:
    backends = left.backend_keys & right.backend_keys
    if not backends:
        raise CompositionError(
            CompositionReason.EXECUTION_INCOMPATIBLE,
            "operators have no common execution backend",
        )
    if left.profile is not None and right.profile is not None:
        profile = merge_profiles(left.profile, right.profile)
    else:
        profile = left.profile if left.profile is not None else right.profile
    return ExecutionRequirements(backends, profile)


@dataclass(frozen=True)
class RuntimeKey:
    runtime_id: str
    kind: str
    input_payload: PayloadKind
    output_payload: PayloadKind


@dataclass(frozen=True, eq=False)
class TransformSpec:
    kind: str
    input_payload: PayloadKind
    output_payload: PayloadKind
    config: Any = field(default=None, compare=False, repr=False)
    runtime_id: str = "ray-data-gpu-fusion-v1"

    def __post_init__(self) -> None:
        if not self.kind or not self.runtime_id:
            raise ValueError("transform kind and runtime_id must be nonempty")

    @property
    def runtime_key(self) -> RuntimeKey:
        return RuntimeKey(
            self.runtime_id, self.kind, self.input_payload, self.output_payload
        )


@dataclass(frozen=True, eq=False)
class OperatorSpec:
    transforms: tuple[TransformSpec, ...]
    execution: ExecutionRequirements
    requires: tuple[Any, ...] = ()
    provides: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "provides", tuple(self.provides))
        if not self.transforms:
            raise ValueError("OperatorSpec requires at least one transform")
        for upstream, downstream in zip(self.transforms, self.transforms[1:]):
            if upstream.output_payload != downstream.input_payload:
                raise ValueError("adjacent transform payloads do not match")
            if upstream.output_payload.materializable_boundary:
                raise ValueError(
                    "a materializable boundary cannot occur inside a region"
                )

    @property
    def input_payload(self) -> PayloadKind:
        return self.transforms[0].input_payload

    @property
    def output_payload(self) -> PayloadKind:
        return self.transforms[-1].output_payload

    @property
    def profile(self) -> ExecutionProfile:
        if self.execution.profile is None:
            raise ValueError("execution requirements are unresolved")
        return self.execution.profile


def fuse_specs(upstream: OperatorSpec, downstream: OperatorSpec) -> OperatorSpec:
    if upstream.output_payload != downstream.input_payload:
        raise CompositionError(
            CompositionReason.PAYLOAD_INCOMPATIBLE,
            f"{upstream.output_payload.key} cannot feed {downstream.input_payload.key}",
        )
    if upstream.output_payload.materializable_boundary:
        raise CompositionError(
            CompositionReason.MATERIALIZABLE_BOUNDARY,
            f"cannot fuse across {upstream.output_payload.key}",
        )
    if not all(requirement in upstream.provides for requirement in downstream.requires):
        raise CompositionError(
            CompositionReason.REQUIREMENTS_UNSATISFIED,
            "upstream properties do not satisfy the downstream requirements",
        )
    return OperatorSpec(
        upstream.transforms + downstream.transforms,
        merge_execution(upstream.execution, downstream.execution),
        upstream.requires,
        downstream.provides,
    )


def compile_region(specs: tuple[OperatorSpec, ...]) -> OperatorSpec:
    if not specs:
        raise ValueError("a GPU region cannot be empty")
    result = specs[0]
    for spec in specs[1:]:
        result = fuse_specs(result, spec)
    return result


__all__ = [
    "CompositionError",
    "CompositionReason",
    "DEVICE_HANDLE",
    "ExecutionProfile",
    "ExecutionRequirements",
    "FRAME_STREAM",
    "OperatorSpec",
    "PARQUET_WORK",
    "PayloadKind",
    "RAY_BLOCK_STREAM",
    "RetryPolicy",
    "RuntimeKey",
    "TransformSpec",
    "compile_region",
    "fuse_specs",
]
