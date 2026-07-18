from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from ray_data_gpu_fusion._compat import ActorPoolStrategy, DataContext, TaskPoolStrategy
from ray_data_gpu_fusion.parquet import (
    ScanDescriptor,
    ScanRecognition,
)
from ray_data_gpu_fusion.rules import map_eligibility, read_eligibility
from ray_data_gpu_fusion.runtime import (
    ExportFrameStreamRuntime,
    FrameStream,
    MapBatchesConfig,
    MapBatchesRuntime,
    _iterate_user_output,
)
from ray_data_gpu_fusion.specs import ExecutionProfile, ExecutionRequirements
from ray_data_gpu_fusion.specs import FRAME_STREAM, OperatorSpec, TransformSpec


def _map_logical(fn):
    return SimpleNamespace(
        fn=fn,
        name="test-map",
        batch_format="cudf",
        batch_size=8,
        zero_copy_batch=True,
        per_block_limit=None,
        ray_remote_args_fn=None,
        ray_remote_args={"num_gpus": 1, "num_cpus": 0},
        compute=ActorPoolStrategy(size=1),
        fn_args=(),
        fn_kwargs={},
        fn_constructor_args=(),
        fn_constructor_kwargs={},
        min_rows_per_bundled_input=8,
    )


def test_async_callable_class_is_declined(monkeypatch):
    class AsyncUdf:
        async def __call__(self, batch):
            return batch

    # ``map_eligibility`` intentionally requires the exact Ray logical type.
    # Patch that nominal check only; the callable classification is the behavior
    # under test.
    import ray_data_gpu_fusion.rules as rules

    monkeypatch.setattr(rules, "MapBatches", SimpleNamespace)
    result = map_eligibility(_map_logical(AsyncUdf), DataContext.get_current().copy())
    assert not result.accepted
    assert "synchronous" in result.reason


def test_closed_region_egress_is_arrow():
    class FakeCudfFrame:
        def to_arrow(self, preserve_index=False):
            assert preserve_index is False
            return pa.table({"value": [1, 2]})

    runtime = ExportFrameStreamRuntime()
    blocks = tuple(runtime.apply(FrameStream(0, iter([FakeCudfFrame()])), None))
    assert len(blocks) == 1
    assert isinstance(blocks[0], pa.Table)


def test_native_map_flattens_only_python_generators():
    def output_generator():
        yield "first"
        yield "second"

    assert tuple(_iterate_user_output(output_generator())) == ("first", "second")

    list_iterator = iter(["first", "second"])
    result = tuple(_iterate_user_output(list_iterator))
    assert len(result) == 1
    assert result[0] is list_iterator


class _FakeFrame:
    def __init__(self, values=(), *, column="value"):
        self.values = tuple(values)
        self.column = column

    def __len__(self):
        return len(self.values)

    @property
    def iloc(self):
        return self

    def __getitem__(self, value):
        assert isinstance(value, slice)
        return _FakeFrame(self.values[value], column=self.column)

    def reset_index(self, *, drop):
        assert drop is True
        return self

    def copy(self, *, deep):
        assert deep is True
        return _FakeFrame(self.values, column=self.column)

    def to_arrow(self, *, preserve_index=False):
        assert preserve_index is False
        return pa.table({self.column: pa.array(self.values, type=pa.int64())})


class _FakeCudf:
    DataFrame = _FakeFrame

    @staticmethod
    def concat(frames, *, ignore_index):
        assert ignore_index is True
        frames = tuple(frames)
        assert frames
        column = frames[0].column
        assert all(frame.column == column for frame in frames)
        return _FakeFrame(
            tuple(value for frame in frames for value in frame.values),
            column=column,
        )


class _FakeCudfContext:
    cudf = _FakeCudf

    def ensure_cudf(self, **_):
        return self


def _fake_map_runtime(udf):
    runtime = MapBatchesRuntime(
        MapBatchesConfig(
            udf=udf,
            udf_is_class=False,
            batch_size=8,
            zero_copy_batch=True,
        ),
        _FakeCudfContext(),
    )
    runtime.initialize()
    return runtime


def test_empty_typed_frame_survives_a_fused_map_chain_without_extra_udf_call():
    calls = []

    def produce_empty(frame):
        calls.append(("producer", len(frame)))
        return _FakeFrame(column="derived")

    def must_not_receive_empty(_frame):
        pytest.fail("Ray does not invoke a MapBatches UDF for an empty input block")

    first = _fake_map_runtime(produce_empty)
    second = _fake_map_runtime(must_not_receive_empty)
    value = FrameStream(0, iter((_FakeFrame((1, 2)),)))

    value = first.apply(value, None)
    value = second.apply(value, None)
    blocks = tuple(ExportFrameStreamRuntime().apply(value, None))

    assert calls == [("producer", 2)]
    assert len(blocks) == 1
    assert blocks[0].num_rows == 0
    assert blocks[0].schema == pa.schema([("derived", pa.int64())])


def test_source_mutation_is_a_planning_decline(tmp_path, monkeypatch):
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    import ray_data_gpu_fusion.rules as rules

    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), path)
    listed_size = path.stat().st_size
    descriptor = ScanDescriptor(
        filesystem=pafs.LocalFileSystem(),
        source_kind="local",
        paths=(str(path),),
        listed_file_sizes=(listed_size,),
        projection=("value",),
        file_schema=pa.schema([("value", pa.int64())]),
        region=None,
    )
    with path.open("ab") as stream:
        stream.write(b"source-mutated-after-listing")

    monkeypatch.setattr(
        rules, "recognize_scan", lambda _logical: ScanRecognition(descriptor=descriptor)
    )
    logical = SimpleNamespace(
        compute=TaskPoolStrategy(),
        ray_remote_args={"scheduling_strategy": "SPREAD"},
    )
    inherited = ExecutionRequirements(profile=ExecutionProfile())
    result = read_eligibility(
        logical, DataContext.get_current().copy(), inherited=inherited
    )
    assert not result.accepted
    assert "listed_size_changed" in result.reason


def test_preserve_order_direct_read_is_declined_before_footer_io(monkeypatch):
    import pyarrow.fs as pafs
    import ray_data_gpu_fusion.rules as rules

    descriptor = ScanDescriptor(
        filesystem=pafs.LocalFileSystem(),
        source_kind="local",
        paths=("/unused/input.parquet",),
        listed_file_sizes=(1,),
        projection=("value",),
        file_schema=pa.schema([("value", pa.int64())]),
        region=None,
    )
    monkeypatch.setattr(
        rules, "recognize_scan", lambda _: ScanRecognition(descriptor=descriptor)
    )
    monkeypatch.setattr(
        rules,
        "read_footer_with_retry",
        lambda *_: pytest.fail("ordered reads must decline before footer I/O"),
    )
    context = DataContext.get_current().copy()
    context.execution_options.preserve_order = True

    result = read_eligibility(SimpleNamespace(), context)

    assert not result.accepted
    assert "preserve_order" in result.reason


def test_read_with_stock_additional_split_factor_stays_stock(monkeypatch):
    from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
    from ray.data._internal.execution.operators.map_transformer import (
        BlockMapTransformFn,
        MapTransformer,
    )
    from ray.data._internal.execution.operators.task_pool_map_operator import (
        TaskPoolMapOperator,
    )
    from ray.data._internal.logical.interfaces import PhysicalPlan
    from ray.data._internal.logical.operators import Read
    from ray_data_gpu_fusion.config import CONFIG_KEY, FusionSettings
    from ray_data_gpu_fusion.rules import LowerClosedGPUOperators
    import ray_data_gpu_fusion.rules as rules

    class FakeDatasource:
        def get_name(self):
            return "Parquet"

    def identity(blocks, _):
        yield from blocks

    context = DataContext.get_current().copy()
    context.set_config(CONFIG_KEY, FusionSettings(True, True))
    source = InputDataBuffer(context, input_data=[])
    physical_read = TaskPoolMapOperator(
        MapTransformer([BlockMapTransformFn(identity, disable_block_shaping=True)]),
        source,
        context,
    )
    physical_read.set_additional_split_factor(4)
    datasource = FakeDatasource()
    logical_read = Read(
        datasource=datasource,
        datasource_or_legacy_reader=datasource,
        parallelism=4,
    )
    # Public read plans map both the InputDataBuffer and its read MapOperator
    # to the logical Read. Lowering must recognize only the MapOperator as the
    # replaceable closure and leave the source node alone during recursion.
    plan = PhysicalPlan(
        physical_read,
        {source: logical_read, physical_read: logical_read},
        context,
    )
    monkeypatch.setattr(
        rules,
        "read_eligibility",
        lambda *_: pytest.fail("split reads must remain on the stock node"),
    )

    lowered = LowerClosedGPUOperators().apply(plan)

    assert lowered.dag is physical_read
    assert "additional_split_factor=4" in (
        physical_read._ray_data_gpu_fusion_decline_reason
    )


def test_diagnostics_lists_regions_and_stock_refusals(monkeypatch):
    import ray_data_gpu_fusion.diagnostics as diagnostics
    from ray_data_gpu_fusion.config import CONFIG_KEY, FusionSettings
    from ray_data_gpu_fusion.operators import ExecutableGPUOperator

    execution = ExecutionRequirements(profile=ExecutionProfile())
    spec = OperatorSpec(
        (
            TransformSpec("read_parquet", FRAME_STREAM, FRAME_STREAM),
            TransformSpec("map_batches", FRAME_STREAM, FRAME_STREAM),
        ),
        execution,
    )

    class Candidate(ExecutableGPUOperator):
        name = "GPU[read_parquet,map_batches]"
        input_dependencies = []
        _gpu_fusion_spec = spec

    class Stock:
        name = "MapBatches(async)"
        input_dependencies = []
        _ray_data_gpu_fusion_decline_reason = "MapBatches UDF is not synchronous"

    class Root:
        name = "root"
        input_dependencies = [Candidate(), Stock()]

    root = Root()
    context = DataContext.get_current().copy()
    context.set_config(CONFIG_KEY, FusionSettings(True, True))
    logical_plan = SimpleNamespace(context=context)
    dataset = SimpleNamespace(_logical_plan=logical_plan)
    monkeypatch.setattr(diagnostics, "explain_plan", lambda _: "RAY PLAN\n")
    monkeypatch.setattr(
        diagnostics,
        "optimized_physical_plan",
        lambda _: SimpleNamespace(dag=root),
    )

    result = diagnostics.explain(dataset)
    assert "GPU Fusion Decisions" in result
    assert "GPU fused" in result
    assert "external_boundary=ray_arrow_blocks" in result
    assert "stock: MapBatches(async)" in result
    assert "not synchronous" in result


def test_lowering_is_executable_and_leaves_stock_gpu_pool_ordinary():
    from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
    from ray.data._internal.logical.interfaces import PhysicalPlan
    from ray.data._internal.logical.operators import InputData, MapBatches
    from ray.data._internal.planner.plan_udf_map_op import plan_udf_map_op
    from ray_data_gpu_fusion.config import CONFIG_KEY, FusionSettings
    from ray_data_gpu_fusion.operators import ExecutableGPUMapBatchesOperator
    from ray_data_gpu_fusion.rules import LowerClosedGPUOperators

    class Identity:
        def __call__(self, batch):
            return batch

    context = DataContext.get_current().copy()
    context.set_config(CONFIG_KEY, FusionSettings(True, False))
    logical_input = InputData([])
    eligible = MapBatches(
        fn=Identity,
        input_dependencies=[logical_input],
        batch_size=8,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_gpus": 1, "num_cpus": 0},
    )
    # This is a stock GPU actor because its batch format is outside Phase 0.
    unsupported = MapBatches(
        fn=Identity,
        input_dependencies=[eligible],
        batch_size=8,
        batch_format="pyarrow",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_gpus": 1, "num_cpus": 0},
    )
    source = InputDataBuffer(context, input_data=[])
    stock_eligible = plan_udf_map_op(eligible, [source], context)
    stock_downstream = plan_udf_map_op(unsupported, [stock_eligible], context)
    stock_remote_args = dict(stock_downstream._ray_remote_args)
    plan = PhysicalPlan(
        stock_downstream,
        {stock_eligible: eligible, stock_downstream: unsupported},
        context,
    )

    lowered = LowerClosedGPUOperators().apply(plan)
    assert lowered.dag is stock_downstream
    assert isinstance(
        lowered.dag.input_dependencies[0], ExecutableGPUMapBatchesOperator
    )
    assert lowered.dag._ray_remote_args == stock_remote_args
    assert lowered.dag.input_dependencies[0]._ray_remote_args["num_gpus"] == 1
    assert lowered.dag.input_dependencies[0]._ray_remote_args["num_cpus"] == 0


def test_compatible_nodes_fuse_and_incompatible_nodes_remain_executable():
    from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
    from ray.data._internal.logical.interfaces import PhysicalPlan
    from ray.data._internal.logical.operators import InputData, MapBatches
    from ray.data._internal.planner.plan_udf_map_op import plan_udf_map_op
    from ray_data_gpu_fusion.config import CONFIG_KEY, FusionSettings
    from ray_data_gpu_fusion.operators import (
        ActorPoolGPUOperator,
        ExecutableGPUMapBatchesOperator,
    )
    from ray_data_gpu_fusion.rules import (
        FuseClosedGPUOperators,
        LowerClosedGPUOperators,
    )

    class Identity:
        def __call__(self, batch):
            return batch

    def make_plan(first_size, second_size):
        context = DataContext.get_current().copy()
        context.set_config(CONFIG_KEY, FusionSettings(True, True))
        logical_input = InputData([])
        first = MapBatches(
            fn=Identity,
            input_dependencies=[logical_input],
            batch_size=8,
            batch_format="cudf",
            compute=ActorPoolStrategy(size=first_size),
            ray_remote_args={"num_gpus": 1, "num_cpus": 0},
        )
        second = MapBatches(
            fn=Identity,
            input_dependencies=[first],
            batch_size=8,
            batch_format="cudf",
            compute=ActorPoolStrategy(size=second_size),
            ray_remote_args={"num_gpus": 1, "num_cpus": 0},
        )
        source = InputDataBuffer(context, input_data=[])
        first_physical = plan_udf_map_op(first, [source], context)
        second_physical = plan_udf_map_op(second, [first_physical], context)
        plan = PhysicalPlan(
            second_physical,
            {first_physical: first, second_physical: second},
            context,
        )
        return LowerClosedGPUOperators().apply(plan), source

    lowered, source = make_plan(1, 1)
    assert isinstance(lowered.dag, ExecutableGPUMapBatchesOperator)
    assert isinstance(
        lowered.dag.input_dependencies[0], ExecutableGPUMapBatchesOperator
    )
    fused = FuseClosedGPUOperators().apply(lowered)
    assert isinstance(fused.dag, ActorPoolGPUOperator)
    assert fused.dag.input_dependencies == [source]
    assert tuple(t.kind for t in fused.dag.gpu_fusion_spec.transforms) == (
        "map_batches",
        "map_batches",
    )

    incompatible, _ = make_plan(1, 2)
    result = FuseClosedGPUOperators().apply(incompatible)
    assert isinstance(result.dag, ExecutableGPUMapBatchesOperator)
    assert isinstance(result.dag.input_dependencies[0], ExecutableGPUMapBatchesOperator)
    assert result.dag._ray_remote_args["num_gpus"] == 1
    assert result.dag.input_dependencies[0]._ray_remote_args["num_gpus"] == 1
