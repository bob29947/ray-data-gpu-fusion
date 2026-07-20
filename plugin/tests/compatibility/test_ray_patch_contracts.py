from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import ray
import ray.data
import ray_data_gpu_fusion as rgf
from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.execution import resource_manager
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.optimizers import PhysicalOptimizer
from ray.data._internal.logical.operators import Read
from ray.data.context import DataContext
from ray_data_gpu_fusion._compat import (
    AdmissionKind,
    ParquetDatasource,
    RESOURCE_ADMISSION_CONTEXT_ATTR,
    RESOURCE_ADMISSION_CONTROL_VERSION,
    RayCompatibilityError,
    compatibility,
    optimized_physical_plan,
)
from ray_data_gpu_fusion.config import CONFIG_KEY
from ray_data_gpu_fusion.operators import (
    CreationOptions,
    ExecutableGPUMapBatchesOperator,
)
from ray_data_gpu_fusion.rules import Eligibility
from ray_data_gpu_fusion.specs import (
    FRAME_STREAM,
    ExecutionProfile,
    ExecutionRequirements,
    OperatorSpec,
    TransformSpec,
)


class _RecordRule(Rule):
    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        plan.context.set_config(
            "phase0.test.rule_runs",
            plan.context.get_config("phase0.test.rule_runs", 0) + 1,
        )
        return plan


class _ContextWithoutAdmissionFlag:
    def __init__(self, delegate):
        self._delegate = delegate

    def __getattr__(self, name):
        if name == RESOURCE_ADMISSION_CONTEXT_ATTR:
            raise AttributeError(name)
        return getattr(self._delegate, name)


def _plugin_eligibility(logical_op, context: DataContext) -> Eligibility:
    spec = OperatorSpec(
        (TransformSpec("test_plugin_map", FRAME_STREAM, FRAME_STREAM),),
        ExecutionRequirements(profile=ExecutionProfile(num_cpus=0)),
    )
    return Eligibility(
        True,
        spec=spec,
        creation_options=CreationOptions(
            name=f"GPU[{logical_op.name}]",
            min_rows_per_bundle=logical_op.min_rows_per_bundled_input,
            target_max_block_size_override=context.target_max_block_size,
        ),
    )


def _reachable(root) -> tuple[object, ...]:
    visited = set()
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


@pytest.fixture
def one_gpu_context():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, num_gpus=1, include_dashboard=False)
    if float(ray.cluster_resources().get("GPU", 0)) < 1:
        pytest.skip("requires a Ray cluster advertising one logical GPU")

    context = DataContext.get_current()
    previous_rules = list(context.custom_physical_optimizer_rule_classes)
    previous_admission = getattr(context, RESOURCE_ADMISSION_CONTEXT_ATTR)
    setattr(context, RESOURCE_ADMISSION_CONTEXT_ATTR, True)
    try:
        yield context
    finally:
        context.custom_physical_optimizer_rule_classes = previous_rules
        setattr(context, RESOURCE_ADMISSION_CONTEXT_ATTR, previous_admission)
        context.remove_config(CONFIG_KEY)
        if started_here:
            ray.shutdown()


def test_required_hooks_and_generic_gpu_admission_capability_are_available():
    context = DataContext.get_current().copy()

    info = compatibility(context)

    assert info.supported
    assert info.missing_seams == ()
    assert (
        resource_manager.RESOURCE_ADMISSION_CONTROL_VERSION
        == RESOURCE_ADMISSION_CONTROL_VERSION
    )
    assert getattr(context, RESOURCE_ADMISSION_CONTEXT_ATTR) is True


def test_plugin_source_does_not_reference_the_legacy_actor_lifecycle_seam():
    source_root = Path(__file__).parents[2] / "src"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(source_root.rglob("*.py"))
    )

    for symbol in (
        "demand_driven_actor_kwargs",
        "defer_actor_start",
        "wait_for_upstream_deferred_operators",
        "release_idle_actors_on_completion",
        "configure_demand_driven_start",
        "GPU_ACTOR_ADMISSION_CONTROL_VERSION",
        "uses_gpu_actor_admission_control",
        "_enable_gpu_actor_admission_control",
    ):
        assert symbol not in source


def test_physical_optimizer_rules_are_plan_local_and_deduplicated():
    enabled_context = DataContext.get_current().copy()
    enabled_context.custom_physical_optimizer_rule_classes = [
        _RecordRule,
        _RecordRule,
    ]
    enabled_input = InputDataBuffer(enabled_context, input_data=[])

    PhysicalOptimizer().optimize(PhysicalPlan(enabled_input, {}, enabled_context))

    assert enabled_context.get_config("phase0.test.rule_runs") == 1

    disabled_context = DataContext.get_current().copy()
    disabled_input = InputDataBuffer(disabled_context, input_data=[])
    PhysicalOptimizer().optimize(PhysicalPlan(disabled_input, {}, disabled_context))
    assert disabled_context.get_config("phase0.test.rule_runs") is None


def test_physical_optimizer_rejects_non_rule_classes():
    context = DataContext.get_current().copy()
    context.custom_physical_optimizer_rule_classes = [object]
    plan = PhysicalPlan(InputDataBuffer(context, input_data=[]), {}, context)

    with pytest.raises(TypeError, match="Rule subclasses"):
        PhysicalOptimizer().optimize(plan)


def test_default_public_parquet_read_has_external_scan_descriptor(tmp_path):
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=1, include_dashboard=False)
    context = DataContext.get_current()
    previous_v2 = context.use_datasource_v2
    context.use_datasource_v2 = False
    try:
        path = tmp_path / "ordinary.parquet"
        pq.write_table(pa.table({"a": [1, 2], "b": [3, 4]}), path)

        dataset = ray.data.read_parquet(str(path))
        logical_op = dataset._logical_plan.dag
        assert isinstance(logical_op, Read)

        result = logical_op.datasource_or_legacy_reader.get_external_scan_descriptor()

        assert result.reason is None
        assert result.descriptor is not None
        assert result.descriptor.source_kind == "local"
        assert result.descriptor.projection == ("a", "b")
        assert result.descriptor.paths == (str(path),)
        assert result.descriptor.listed_file_sizes[0] == path.stat().st_size
        assert isinstance(result.descriptor.filesystem, pafs.LocalFileSystem)
    finally:
        context.use_datasource_v2 = previous_v2
        if started_here:
            ray.shutdown()


@pytest.mark.parametrize(
    ("missing", "expected_seam"),
    (
        ("candidate", "resource_admission_control_v1"),
        ("candidate_hook", "actor_pool_resource_admission_spec"),
        ("context", "resource_admission_context"),
        ("h1", "plan_local_physical_rules"),
        ("h2", "parquet_external_scan_descriptor"),
    ),
)
def test_enable_fails_closed_when_a_required_capability_is_missing(
    monkeypatch, missing, expected_seam
):
    context = DataContext.get_current().copy()
    original_rules = list(context.custom_physical_optimizer_rule_classes)

    if missing == "candidate":
        monkeypatch.delattr(resource_manager, "RESOURCE_ADMISSION_CONTROL_VERSION")
    elif missing == "candidate_hook":
        monkeypatch.delattr(ActorPoolMapOperator, "resource_admission_spec")
    elif missing == "context":
        context = _ContextWithoutAdmissionFlag(context)
    elif missing == "h1":
        delattr(context, "custom_physical_optimizer_rule_classes")
    else:
        monkeypatch.delattr(ParquetDatasource, "get_external_scan_descriptor")

    info = compatibility(context)
    assert not info.supported
    assert expected_seam in info.missing_seams

    with pytest.raises(RayCompatibilityError):
        rgf.enable(context=context)

    assert context.get_config(CONFIG_KEY) is None
    if missing != "h1":
        assert context.custom_physical_optimizer_rule_classes == original_rules


def test_enable_fails_closed_when_resource_admission_is_disabled():
    context = DataContext.get_current().copy()
    original_rules = list(context.custom_physical_optimizer_rule_classes)
    setattr(context, RESOURCE_ADMISSION_CONTEXT_ATTR, False)

    info = compatibility(context)
    assert not info.supported
    assert "resource_admission_disabled" in info.missing_seams

    with pytest.raises(RayCompatibilityError):
        rgf.enable(context=context)

    assert context.custom_physical_optimizer_rule_classes == original_rules
    assert context.get_config(CONFIG_KEY) is None


def test_planning_fails_closed_if_resource_admission_is_disabled_after_enable():
    from ray_data_gpu_fusion.rules import LowerClosedGPUOperators

    context = DataContext.get_current().copy()
    rgf.enable(context=context)
    setattr(context, RESOURCE_ADMISSION_CONTEXT_ATTR, False)
    plan = PhysicalPlan(InputDataBuffer(context, input_data=[]), {}, context)

    with pytest.raises(RayCompatibilityError):
        LowerClosedGPUOperators().apply(plan)


@pytest.mark.parametrize(
    ("attribute", "value"),
    (
        ("op_resource_reservation_enabled", False),
        ("wait_for_min_actors_s", 1),
    ),
)
def test_generic_admission_supports_previous_actor_mode_restrictions(attribute, value):
    context = DataContext.get_current().copy()
    setattr(context, attribute, value)

    assert compatibility(context).supported


def test_stock_callable_class_gpu_map_batches_share_one_gpu(one_gpu_context):
    class IdentityActor:
        def __call__(self, batch):
            return batch

    class SecondIdentityActor:
        def __call__(self, batch):
            return batch

    def identity_task(batch):
        return batch

    dataset = (
        ray.data.from_items([{"value": 1}, {"value": 2}])
        .map_batches(
            IdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
        .map_batches(
            identity_task,
            batch_format="pyarrow",
            compute=TaskPoolStrategy(size=1),
            num_cpus=1,
        )
        .map_batches(
            SecondIdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
    )

    actor_regions = [
        operator
        for operator in _reachable(optimized_physical_plan(dataset._logical_plan).dag)
        if isinstance(operator, ActorPoolMapOperator)
    ]
    assert len(actor_regions) == 2
    assert all(
        region.resource_admission_spec().kind is AdmissionKind.ELASTIC_POOL
        for region in actor_regions
    )

    assert sorted(dataset.take_all(), key=lambda row: row["value"]) == [
        {"value": 1},
        {"value": 2},
    ]


def test_plugin_created_gpu_regions_share_one_gpu(monkeypatch, one_gpu_context):
    import ray_data_gpu_fusion.rules as rules

    class IdentityActor:
        def __call__(self, batch):
            return batch

    class SecondIdentityActor:
        def __call__(self, batch):
            return batch

    monkeypatch.setattr(rules, "map_eligibility", _plugin_eligibility)
    rgf.enable(context=one_gpu_context, fusion=False)
    dataset = (
        ray.data.from_items([{"value": 1}, {"value": 2}])
        .map_batches(
            IdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
        .map_batches(
            SecondIdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
    )

    physical = optimized_physical_plan(dataset._logical_plan)
    plugin_regions = [
        operator
        for operator in _reachable(physical.dag)
        if isinstance(operator, ExecutableGPUMapBatchesOperator)
    ]
    assert len(plugin_regions) == 2
    assert all(region._ray_remote_args["num_gpus"] == 1 for region in plugin_regions)
    assert all(
        region.resource_admission_spec().kind is AdmissionKind.ELASTIC_POOL
        for region in plugin_regions
    )

    assert sorted(dataset.take_all(), key=lambda row: row["value"]) == [
        {"value": 1},
        {"value": 2},
    ]


def test_incompatible_plugin_gpu_regions_hand_off_one_gpu(monkeypatch, one_gpu_context):
    import ray_data_gpu_fusion.rules as rules

    class IdentityActor:
        def __call__(self, batch):
            return batch

    class SecondIdentityActor:
        def __call__(self, batch):
            return batch

    def incompatible_eligibility(logical_op, context: DataContext) -> Eligibility:
        # A CPU reservation mismatch is an explicit fusion incompatibility, but
        # both standalone regions still request the same one logical GPU.
        num_cpus = 0 if logical_op.fn is IdentityActor else 0.25
        spec = OperatorSpec(
            (TransformSpec("test_plugin_map", FRAME_STREAM, FRAME_STREAM),),
            ExecutionRequirements(profile=ExecutionProfile(num_cpus=num_cpus)),
        )
        return Eligibility(
            True,
            spec=spec,
            creation_options=CreationOptions(
                name=f"GPU[{logical_op.name}]",
                min_rows_per_bundle=logical_op.min_rows_per_bundled_input,
                target_max_block_size_override=context.target_max_block_size,
            ),
        )

    monkeypatch.setattr(rules, "map_eligibility", incompatible_eligibility)
    rgf.enable(context=one_gpu_context)
    dataset = (
        ray.data.from_items([{"value": 1}, {"value": 2}])
        .map_batches(
            IdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
        .map_batches(
            SecondIdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0.25,
        )
    )

    physical = optimized_physical_plan(dataset._logical_plan)
    plugin_regions = [
        operator
        for operator in _reachable(physical.dag)
        if isinstance(operator, ExecutableGPUMapBatchesOperator)
    ]
    assert len(plugin_regions) == 2
    assert all(
        region.resource_admission_spec().kind is AdmissionKind.ELASTIC_POOL
        for region in plugin_regions
    )

    assert sorted(dataset.take_all(), key=lambda row: row["value"]) == [
        {"value": 1},
        {"value": 2},
    ]


def test_plugin_decline_to_stock_gpu_actor_is_admission_safe(
    monkeypatch, one_gpu_context
):
    import ray_data_gpu_fusion.rules as rules

    class PluginIdentityActor:
        def __call__(self, batch):
            return batch

    class FallbackIdentityActor:
        def __call__(self, batch):
            return batch

    stock_eligibility = rules.map_eligibility

    def selective_eligibility(logical_op, context):
        if logical_op.fn is PluginIdentityActor:
            return _plugin_eligibility(logical_op, context)
        return stock_eligibility(logical_op, context)

    monkeypatch.setattr(rules, "map_eligibility", selective_eligibility)
    rgf.enable(context=one_gpu_context, fusion=False)
    dataset = (
        ray.data.from_items([{"value": 1}, {"value": 2}])
        .map_batches(
            PluginIdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
        .map_batches(
            FallbackIdentityActor,
            batch_size=2,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
    )

    physical = optimized_physical_plan(dataset._logical_plan)
    operators = _reachable(physical.dag)
    plugin_regions = [
        operator
        for operator in operators
        if isinstance(operator, ExecutableGPUMapBatchesOperator)
    ]
    stock_regions = [
        operator
        for operator in operators
        if isinstance(operator, ActorPoolMapOperator)
        and not isinstance(operator, ExecutableGPUMapBatchesOperator)
    ]
    assert plugin_regions
    assert stock_regions
    assert all(
        region.resource_admission_spec().kind is AdmissionKind.ELASTIC_POOL
        for region in (*plugin_regions, *stock_regions)
    )

    assert sorted(dataset.take_all(), key=lambda row: row["value"]) == [
        {"value": 1},
        {"value": 2},
    ]
