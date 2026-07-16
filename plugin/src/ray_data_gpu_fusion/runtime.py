"""Actor-local transform registry and closed cuDF region execution."""

from __future__ import annotations

import collections
import os
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from types import GeneratorType
from typing import Any, Callable, Optional, Protocol

from ray_data_gpu_fusion._compat import (
    BlockAccessor,
    DataContext,
    TaskContext,
    UserCodeException,
    _is_ray_debugger_post_mortem_enabled,
)
from ray_data_gpu_fusion.specs import (
    FRAME_STREAM,
    PARQUET_WORK,
    RAY_BLOCK_STREAM,
    OperatorSpec,
    RuntimeKey,
    TransformSpec,
)


IMPORT_RAY_BLOCKS = "import_ray_blocks"
IMPORT_PARQUET_WORK = "import_parquet_work"
MAP_BATCHES = "map_batches"
EXPORT_FRAME_STREAM = "export_frame_stream"


@dataclass(frozen=True, eq=False)
class FrameStream:
    """One actor task's device-resident frame stream.

    The envelope deliberately proves no distribution or ordering property.
    Unfused regions export it to Arrow-backed Ray blocks; only fusion permits it
    to flow directly between transforms in one actor invocation.
    """

    partition_id: int
    frames: Any

    def __post_init__(self) -> None:
        if not isinstance(self.partition_id, int) or self.partition_id < 0:
            raise ValueError("partition_id must be a nonnegative integer")


@dataclass(frozen=True, eq=False)
class MapBatchesConfig:
    udf: Any
    udf_is_class: bool
    batch_size: int
    zero_copy_batch: bool
    fn_args: tuple[Any, ...] = ()
    fn_kwargs: tuple[tuple[str, Any], ...] = ()
    constructor_args: tuple[Any, ...] = ()
    constructor_kwargs: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "fn_args",
            "fn_kwargs",
            "constructor_args",
            "constructor_kwargs",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        for name in ("fn_kwargs", "constructor_kwargs"):
            values = getattr(self, name)
            if len(dict(values)) != len(values):
                raise ValueError(f"{name} contains duplicate keys")


@dataclass
class RuntimeContext:
    operator_spec: OperatorSpec
    state: dict[str, Any] = field(default_factory=dict)

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        if key not in self.state:
            self.state[key] = factory()
        return self.state[key]


class TransformRuntime(Protocol):
    def initialize(self) -> None: ...
    def apply(self, value: Any, task_context: TaskContext) -> Any: ...


RuntimeFactory = Callable[[TransformSpec, RuntimeContext], TransformRuntime]
ConfigValidator = Callable[[Any], bool]


@dataclass(frozen=True)
class RuntimeRegistration:
    validate_config: ConfigValidator = field(compare=False, repr=False)
    factory: RuntimeFactory = field(compare=False, repr=False)


_RUNTIMES: dict[RuntimeKey, RuntimeRegistration] = {}
_BUILTINS_REGISTERED = False


def register_runtime(
    key: RuntimeKey,
    registration: RuntimeRegistration,
    *,
    replace_existing: bool = False,
) -> None:
    if key in _RUNTIMES and not replace_existing:
        raise ValueError(f"runtime already registered: {key!r}")
    _RUNTIMES[key] = registration


def unregister_runtime(
    key: RuntimeKey, *, expected: Optional[RuntimeRegistration] = None
) -> RuntimeRegistration:
    registration = _RUNTIMES[key]
    if expected is not None and registration is not expected:
        raise ValueError("registered runtime is not the expected implementation")
    del _RUNTIMES[key]
    return registration


def resolve_runtime(spec: TransformSpec) -> RuntimeRegistration:
    ensure_builtin_runtimes()
    registration = _RUNTIMES.get(spec.runtime_key)
    if registration is None:
        raise ValueError(f"no runtime registered for {spec.runtime_key!r}")
    try:
        accepted = registration.validate_config(spec.config)
    except Exception as error:
        raise ValueError(f"invalid config for {spec.kind!r}: {error}") from error
    if accepted is not True:
        raise ValueError(f"invalid config for {spec.kind!r}")
    return registration


def supports_runtime(spec: TransformSpec) -> bool:
    try:
        resolve_runtime(spec)
    except ValueError:
        return False
    return True


class RegionWorker:
    def __init__(
        self,
        operator_spec: OperatorSpec,
        registrations: Optional[tuple[RuntimeRegistration, ...]] = None,
    ) -> None:
        self._operator_spec = operator_spec
        self._registrations = registrations or tuple(
            resolve_runtime(transform) for transform in operator_spec.transforms
        )
        if len(self._registrations) != len(operator_spec.transforms):
            raise ValueError("runtime registrations do not align with transforms")
        self._context = RuntimeContext(operator_spec)
        self._runtimes: list[TransformRuntime] = []
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._runtimes = [
            registration.factory(transform, self._context)
            for transform, registration in zip(
                self._operator_spec.transforms, self._registrations
            )
        ]
        for runtime in self._runtimes:
            runtime.initialize()
        self._initialized = True

    def apply(self, value: Any, task_context: TaskContext) -> Any:
        self.initialize()
        for runtime in self._runtimes:
            value = runtime.apply(value, task_context)
        return value


class RegionBlockTransform:
    """Ray ``BlockMapTransformFn`` callable for one closed actor region."""

    def __init__(
        self,
        operator_spec: OperatorSpec,
        registrations: Optional[tuple[RuntimeRegistration, ...]] = None,
    ) -> None:
        if (
            operator_spec.input_payload != RAY_BLOCK_STREAM
            or operator_spec.output_payload != RAY_BLOCK_STREAM
        ):
            raise ValueError("a materialized region must have Ray-block endpoints")
        self._worker = RegionWorker(operator_spec, registrations)

    def initialize(self) -> None:
        self._worker.initialize()

    def __call__(
        self, blocks: Iterable[Any], task_context: TaskContext
    ) -> Iterable[Any]:
        return self._worker.apply(blocks, task_context)


def _raise_user_code(error: Exception) -> None:
    context = DataContext.get_current()
    if _is_ray_debugger_post_mortem_enabled() or context.raise_original_map_exception:
        raise error
    raise UserCodeException("UDF failed to process a data block.") from error


def _invoke(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as error:
        _raise_user_code(error)


def _iterate_user_output(value: Any) -> Iterator[Any]:
    # Match the pinned Ray MapBatches contract exactly. Ray flattens Python
    # generator results, but treats every other Iterator (for example a
    # ``list_iterator``) as one batch and rejects it during batch validation.
    # Accepting arbitrary iterators here would make a UDF's validity depend on
    # whether this plugin happened to fuse the operator.
    values = value if isinstance(value, GeneratorType) else iter((value,))
    try:
        iterator = iter(values)
    except Exception as error:
        _raise_user_code(error)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception as error:
            _raise_user_code(error)


_GIB = 1024**3


def live_rmm_pool_maximum(free_memory: int) -> int:
    maximum = min(int(free_memory * 0.70), int(free_memory) - 2 * _GIB)
    return max(0, maximum // 256 * 256)


def refresh_aws_credentials(
    region: str, session: Any = None, credentials: Any = None
) -> tuple[Any, Any]:
    if os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3"):
        raise RuntimeError("native S3 execution does not support endpoint overrides")
    if session is None:
        import botocore.session

        session = botocore.session.get_session()
    if credentials is None:
        credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("no ambient AWS credentials are available")
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        raise RuntimeError("the ambient AWS credential provider is incomplete")
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)
    os.environ["AWS_DEFAULT_REGION"] = region
    os.environ["AWS_REGION"] = region
    return session, credentials


class CudfRuntimeContext:
    """Mutable CUDA state owned by one Ray actor process."""

    def __init__(self) -> None:
        self.cudf: Any = None
        self.cupy: Any = None
        self._rmm_upstream: Any = None
        self._rmm_pool: Any = None
        self._source_kind: Optional[str] = None
        self._region: Optional[str] = None
        self._aws_session: Any = None
        self._aws_credentials: Any = None
        self._s3_client: Any = None

    def ensure_cudf(
        self,
        *,
        source_kind: Optional[str] = None,
        region: Optional[str] = None,
    ) -> "CudfRuntimeContext":
        if self.cudf is None:
            import rmm

            free_memory, _ = rmm.mr.available_device_memory()
            maximum = live_rmm_pool_maximum(free_memory)
            if maximum <= 0:
                raise RuntimeError("GPU execution requires more than 2 GiB free memory")
            self._rmm_upstream = rmm.mr.get_current_device_resource()
            self._rmm_pool = rmm.mr.PoolMemoryResource(
                self._rmm_upstream,
                initial_pool_size=min(_GIB, maximum),
                maximum_pool_size=maximum,
            )
            rmm.mr.set_current_device_resource(self._rmm_pool)

            import cupy
            from rmm.allocators.cupy import rmm_cupy_allocator

            cupy.cuda.set_allocator(rmm_cupy_allocator)
            import cudf

            self.cudf, self.cupy = cudf, cupy

        if source_kind == "s3":
            self._ensure_s3(region)
        elif source_kind not in (None, "local"):
            raise ValueError(f"unsupported direct-read source kind: {source_kind!r}")
        return self

    def _ensure_s3(self, region: Optional[str]) -> None:
        if not isinstance(region, str) or not region:
            raise RuntimeError("native S3 execution requires a region")
        if self._source_kind not in (None, "s3") or self._region not in (None, region):
            raise RuntimeError("a GPU actor cannot change its direct-read source")
        if self._aws_session is None:
            client = None
            try:
                session, credentials = refresh_aws_credentials(region)
                client = session.create_client("s3", region_name=region)
                os.environ["KVIKIO_NTHREADS"] = "32"
                os.environ["KVIKIO_TASK_SIZE"] = str(16 * 1024**2)
                import kvikio.defaults as defaults
                from kvikio.remote_file import is_remote_file_available

                if not is_remote_file_available():
                    raise RuntimeError("KvikIO remote I/O is unavailable")
                defaults.set({"num_threads": 32, "task_size": 16 * 1024**2})
                self.cudf.set_option("kvikio_remote_io", True)
                if not self.cudf.get_option("kvikio_remote_io"):
                    raise RuntimeError("cuDF rejected KvikIO remote I/O")
            except Exception:
                if client is not None:
                    with suppress(Exception):
                        client.close()
                raise
            self._aws_session = session
            self._aws_credentials = credentials
            self._s3_client = client
        self._source_kind, self._region = "s3", region

    @property
    def s3_client(self) -> Any:
        if self._s3_client is None:
            raise RuntimeError("S3 runtime is not initialized")
        return self._s3_client

    def refresh_s3_for_task(self) -> None:
        if self._source_kind != "s3":
            return
        session, credentials = refresh_aws_credentials(
            self._region, self._aws_session, self._aws_credentials
        )
        client = session.create_client("s3", region_name=self._region)
        previous = self._s3_client
        self._aws_session, self._aws_credentials, self._s3_client = (
            session,
            credentials,
            client,
        )
        if previous is not None:
            with suppress(Exception):
                previous.close()


class ImportRayBlocksRuntime:
    def initialize(self) -> None:
        pass

    def apply(self, value: Any, task_context: TaskContext) -> FrameStream:
        partition_id = int(getattr(task_context, "task_idx", 0))

        def frames() -> Iterator[Any]:
            for block in value:
                frame = BlockAccessor.for_block(block).to_batch_format("cudf")
                if len(frame):
                    yield frame

        return FrameStream(partition_id, frames())


def rebatched_frames(
    frames: Iterable[Any], batch_size: int, cudf: Any
) -> Iterator[Any]:
    pending: collections.deque[list[Any]] = collections.deque()
    available = 0

    def take(rows: int) -> Any:
        nonlocal available
        pieces = []
        remaining = rows
        while remaining:
            frame, offset = pending[0]
            count = min(remaining, len(frame) - offset)
            pieces.append(frame.iloc[offset : offset + count])
            offset += count
            remaining -= count
            available -= count
            if offset == len(frame):
                pending.popleft()
            else:
                pending[0][1] = offset
        if len(pieces) == 1:
            return pieces[0].reset_index(drop=True)
        return cudf.concat(pieces, ignore_index=True)

    for frame in frames:
        if len(frame) == 0:
            continue
        pending.append([frame, 0])
        available += len(frame)
        while available >= batch_size:
            yield take(batch_size)
    if available:
        yield take(available)


def _normalize_frame(batch: Any, cudf: Any) -> Any:
    if isinstance(batch, cudf.DataFrame):
        return batch
    block = BlockAccessor.batch_to_block(batch)
    return cudf.DataFrame.from_arrow(BlockAccessor.for_block(block).to_arrow())


class MapBatchesRuntime:
    def __init__(self, config: MapBatchesConfig, context: CudfRuntimeContext) -> None:
        self._config, self._context = config, context
        self._udf: Any = None

    def initialize(self) -> None:
        self._context.ensure_cudf()
        if self._config.udf_is_class:
            self._udf = _invoke(
                self._config.udf,
                *self._config.constructor_args,
                **dict(self._config.constructor_kwargs),
            )
        else:
            self._udf = self._config.udf

    def _apply_frames(self, frames: Iterable[Any]) -> Iterator[Any]:
        cudf = self._context.cudf
        empty_input = None

        def nonempty_inputs() -> Iterator[Any]:
            nonlocal empty_input
            for value in frames:
                frame = _normalize_frame(value, cudf)
                if len(frame):
                    yield frame
                else:
                    # Ray's batcher does not invoke a MapBatches UDF for an
                    # empty input block. Retain the last typed empty frame so a
                    # fused downstream map can pass it through without losing
                    # the Arrow schema at the closed-region boundary.
                    empty_input = frame

        invoked_udf = False
        emitted_nonempty = False
        empty_output = None
        for batch in rebatched_frames(nonempty_inputs(), self._config.batch_size, cudf):
            invoked_udf = True
            udf_input = batch if self._config.zero_copy_batch else batch.copy(deep=True)
            result = _invoke(
                self._udf,
                udf_input,
                *self._config.fn_args,
                **dict(self._config.fn_kwargs),
            )
            for output in _iterate_user_output(result):
                try:
                    frame = _normalize_frame(output, cudf).reset_index(drop=True)
                    if len(frame):
                        emitted_nonempty = True
                        yield frame
                    else:
                        # BlockOutputBuffer emits one typed empty block when all
                        # UDF outputs are empty. Keep only the last empty frame,
                        # matching DelegatingBlockBuilder's empty-block behavior.
                        empty_output = frame
                except Exception as error:
                    _raise_user_code(error)

        if not emitted_nonempty:
            if empty_output is not None:
                yield empty_output
            elif not invoked_udf and empty_input is not None:
                # A fused downstream map sees the same empty input boundary as
                # an unfused Ray map: its UDF is not called, but one typed empty
                # result remains materializable.
                yield empty_input
            else:
                # A task with no input batches, or a generator UDF with no
                # yields, still produces Ray's single schema-less empty block.
                yield cudf.DataFrame()

    def apply(self, value: Any, task_context: TaskContext) -> FrameStream:
        del task_context
        if not isinstance(value, FrameStream):
            raise TypeError("map_batches expected an actor-local FrameStream")
        return FrameStream(value.partition_id, self._apply_frames(value.frames))


class ImportParquetWorkRuntime:
    def initialize(self) -> None:
        pass

    def apply(self, value: Any, task_context: TaskContext) -> Any:
        del task_context
        from ray_data_gpu_fusion.parquet import ParquetWork, deserialize_work

        if isinstance(value, ParquetWork):
            return value
        descriptors: list[Any] = []
        for block in value:
            descriptors.extend(BlockAccessor.for_block(block).to_numpy("work").tolist())
        if len(descriptors) != 1:
            raise ValueError(
                "a GPU Parquet task requires exactly one work descriptor, got "
                f"{len(descriptors)}"
            )
        return deserialize_work(descriptors[0])


class ExportFrameStreamRuntime:
    def initialize(self) -> None:
        pass

    def apply(self, value: Any, task_context: TaskContext) -> Iterator[Any]:
        del task_context
        if not isinstance(value, FrameStream):
            raise TypeError("GPU region egress expected an actor-local FrameStream")
        blocks = []
        for frame in value.frames:
            try:
                block = frame.to_arrow(preserve_index=False)
            except (AttributeError, TypeError):
                block = BlockAccessor.for_block(frame).to_arrow()
            blocks.append(block)
        # Eager materialization is intentional: GPU/I/O/UDF failures surface in
        # the actor task before the streaming generator yields its first block.
        return iter(tuple(blocks))


def _none(config: Any) -> bool:
    return config is None


def _instance(expected: type[Any]) -> ConfigValidator:
    return lambda config: isinstance(config, expected)


def _boundary_factory(spec: TransformSpec, context: RuntimeContext) -> TransformRuntime:
    del context
    if spec.kind == IMPORT_RAY_BLOCKS:
        return ImportRayBlocksRuntime()
    if spec.kind == IMPORT_PARQUET_WORK:
        return ImportParquetWorkRuntime()
    if spec.kind == EXPORT_FRAME_STREAM:
        return ExportFrameStreamRuntime()
    raise ValueError(f"unknown boundary runtime {spec.kind!r}")


def _map_factory(spec: TransformSpec, context: RuntimeContext) -> TransformRuntime:
    cudf_context = context.get_or_create("cudf", CudfRuntimeContext)
    return MapBatchesRuntime(spec.config, cudf_context)


def _read_factory(spec: TransformSpec, context: RuntimeContext) -> TransformRuntime:
    from ray_data_gpu_fusion.parquet import ReadParquetRuntime

    cudf_context = context.get_or_create("cudf", CudfRuntimeContext)
    return ReadParquetRuntime(spec.config, cudf_context)


def ensure_builtin_runtimes() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from ray_data_gpu_fusion.parquet import READ_PARQUET, ReadParquetConfig

    entries = (
        (
            RuntimeKey(
                "ray-data-gpu-fusion-v1",
                IMPORT_RAY_BLOCKS,
                RAY_BLOCK_STREAM,
                FRAME_STREAM,
            ),
            RuntimeRegistration(_none, _boundary_factory),
        ),
        (
            RuntimeKey(
                "ray-data-gpu-fusion-v1",
                IMPORT_PARQUET_WORK,
                RAY_BLOCK_STREAM,
                PARQUET_WORK,
            ),
            RuntimeRegistration(_none, _boundary_factory),
        ),
        (
            RuntimeKey(
                "ray-data-gpu-fusion-v1", MAP_BATCHES, FRAME_STREAM, FRAME_STREAM
            ),
            RuntimeRegistration(_instance(MapBatchesConfig), _map_factory),
        ),
        (
            RuntimeKey(
                "ray-data-gpu-fusion-v1",
                READ_PARQUET,
                PARQUET_WORK,
                FRAME_STREAM,
            ),
            RuntimeRegistration(_instance(ReadParquetConfig), _read_factory),
        ),
        (
            RuntimeKey(
                "ray-data-gpu-fusion-v1",
                EXPORT_FRAME_STREAM,
                FRAME_STREAM,
                RAY_BLOCK_STREAM,
            ),
            RuntimeRegistration(_none, _boundary_factory),
        ),
    )
    for key, registration in entries:
        _RUNTIMES.setdefault(key, registration)
    _BUILTINS_REGISTERED = True


__all__ = [
    "CudfRuntimeContext",
    "EXPORT_FRAME_STREAM",
    "FrameStream",
    "IMPORT_PARQUET_WORK",
    "IMPORT_RAY_BLOCKS",
    "MAP_BATCHES",
    "MapBatchesConfig",
    "RegionBlockTransform",
    "RuntimeContext",
    "RuntimeRegistration",
    "ensure_builtin_runtimes",
    "live_rmm_pool_maximum",
    "register_runtime",
    "resolve_runtime",
    "supports_runtime",
    "unregister_runtime",
]
