from __future__ import annotations

from typing import Iterable
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import ray
from ray.data._internal.actor_autoscaler.default_actor_autoscaler import (
    DefaultActorAutoscaler,
)
from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.optimizers import PhysicalOptimizer
from ray.data._internal.logical.operators import Read
from ray.data.block import Block
from ray.data.context import DataContext


class _RecordRule(Rule):
    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        plan.context.set_config(
            "phase0.test.rule_runs",
            plan.context.get_config("phase0.test.rule_runs", 0) + 1,
        )
        return plan


class _ConfigureDemandDrivenActors(Rule):
    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        for operator in plan.dag.post_order_iter():
            if isinstance(operator, ActorPoolMapOperator):
                operator.configure_demand_driven_start(
                    wait_for_upstream_deferred_operators=False,
                    release_idle_actors_on_completion=True,
                )
        plan.context.custom_physical_optimizer_rule_classes = [
            rule
            for rule in plan.context.custom_physical_optimizer_rule_classes
            if rule is not type(self)
        ]
        return plan


class _ConfigureSequentialDemandDrivenActors(Rule):
    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        actor_operators = [
            operator
            for operator in plan.dag.post_order_iter()
            if isinstance(operator, ActorPoolMapOperator)
        ]
        for operator in actor_operators:
            operator.configure_demand_driven_start(
                wait_for_upstream_deferred_operators=True,
                release_idle_actors_on_completion=True,
            )
        plan.context.set_config(
            "phase0.test.sequential_actor_regions", len(actor_operators)
        )
        plan.context.custom_physical_optimizer_rule_classes = [
            rule
            for rule in plan.context.custom_physical_optimizer_rule_classes
            if rule is not type(self)
        ]
        return plan


def _identity_transformer() -> MapTransformer:
    def identity(blocks: Iterable[Block], _):
        yield from blocks

    return MapTransformer([BlockMapTransformFn(identity, disable_block_shaping=True)])


def _actor_op(
    context: DataContext,
    input_op,
    *,
    defer_actor_start: bool = False,
    wait_for_upstream_deferred_operators: bool = False,
    release_idle_actors_on_completion: bool = False,
) -> ActorPoolMapOperator:
    return ActorPoolMapOperator(
        _identity_transformer(),
        input_op,
        context,
        ActorPoolStrategy(size=2),
        ray_remote_args={"num_cpus": 0, "num_gpus": 1},
        defer_actor_start=defer_actor_start,
        wait_for_upstream_deferred_operators=(wait_for_upstream_deferred_operators),
        release_idle_actors_on_completion=release_idle_actors_on_completion,
    )


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


def test_demand_driven_actor_lifecycle_is_generic_and_zero_before_activation():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    upstream = _actor_op(
        context,
        source,
        defer_actor_start=True,
        release_idle_actors_on_completion=True,
    )
    downstream = _actor_op(
        context,
        upstream,
        defer_actor_start=True,
        wait_for_upstream_deferred_operators=True,
        release_idle_actors_on_completion=True,
    )

    assert downstream.defer_actor_start
    assert downstream.actor_pool_start_deferred
    assert downstream.can_add_input()
    assert not downstream._upstream_deferred_operators_finished()

    minimum, maximum = downstream.min_max_resource_requirements()
    assert minimum == ExecutionResources.zero()
    assert maximum.gpu == 2
    assert downstream.min_scheduling_resources() == ExecutionResources.zero()

    upstream._is_execution_marked_finished = True
    assert downstream._upstream_deferred_operators_finished()


def test_stock_actor_map_can_be_configured_by_a_physical_rule():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    operator = _actor_op(context, source)

    assert not operator.defer_actor_start
    operator.configure_demand_driven_start()

    assert operator.defer_actor_start
    assert operator.actor_pool_start_deferred


def test_autoscaler_does_not_restore_deferred_pool_minimum():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    operator = _actor_op(context, source, defer_actor_start=True)
    operator.has_completed = MagicMock(return_value=False)
    topology = {operator: MagicMock()}
    autoscaler = DefaultActorAutoscaler(
        topology,
        MagicMock(),
        config=context.autoscaling_config,
    )

    request = autoscaler._derive_target_scaling_config(
        operator._actor_pool,
        operator,
        topology[operator],
    )

    assert request.delta == 0
    assert request.reason == "waiting for input demand"


def test_waiting_for_upstream_requires_deferred_start():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])

    with pytest.raises(ValueError, match="requires deferred actor startup"):
        _actor_op(
            context,
            source,
            wait_for_upstream_deferred_operators=True,
        )


def test_deferred_pool_processes_with_first_ready_actor():
    class IdentityBatch:
        def __call__(self, batch):
            return batch

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=1, include_dashboard=False)
    context = DataContext.get_current()
    previous_rules = list(context.custom_physical_optimizer_rule_classes)
    context.custom_physical_optimizer_rule_classes = [
        *previous_rules,
        _ConfigureDemandDrivenActors,
    ]
    try:
        dataset = ray.data.from_items([{"value": 1}, {"value": 2}]).map_batches(
            IdentityBatch,
            batch_format="pyarrow",
            compute=ActorPoolStrategy(size=2),
            num_cpus=1,
        )

        assert dataset.take_all() == [{"value": 1}, {"value": 2}]
    finally:
        context.custom_physical_optimizer_rule_classes = previous_rules
        if started_here:
            ray.shutdown()


def test_two_unfused_actor_regions_share_one_scarce_resource():
    class IdentityActor:
        def __call__(self, batch):
            return batch

    def identity_task(batch):
        return batch

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=1, include_dashboard=False)
    context = DataContext.get_current()
    previous_rules = list(context.custom_physical_optimizer_rule_classes)
    context.custom_physical_optimizer_rule_classes = [
        *previous_rules,
        _ConfigureSequentialDemandDrivenActors,
    ]
    try:
        dataset = (
            ray.data.from_items([{"value": 1}, {"value": 2}])
            .map_batches(
                IdentityActor,
                batch_format="pyarrow",
                compute=ActorPoolStrategy(size=1),
                num_cpus=1,
            )
            .map_batches(
                identity_task,
                batch_format="pyarrow",
                compute=TaskPoolStrategy(size=1),
                num_cpus=1,
            )
            .map_batches(
                IdentityActor,
                batch_format="pyarrow",
                compute=ActorPoolStrategy(size=1),
                num_cpus=1,
            )
        )

        assert dataset.take_all() == [{"value": 1}, {"value": 2}]
        assert dataset.context.get_config("phase0.test.sequential_actor_regions") == 2
    finally:
        context.custom_physical_optimizer_rule_classes = previous_rules
        if started_here:
            ray.shutdown()
