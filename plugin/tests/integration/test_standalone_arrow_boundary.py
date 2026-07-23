from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
import ray.data
from ray.data import ActorPoolStrategy

import ray_data_gpu_fusion as rgf
from ray_data_gpu_fusion._compat import optimized_physical_plan
from ray_data_gpu_fusion.operators import ActorPoolGPUOperator


def _reachable(root):
    visited = set()
    stack = [root]
    while stack:
        operator = stack.pop()
        if operator in visited:
            continue
        visited.add(operator)
        yield operator
        stack.extend(operator.input_dependencies)


def _gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="requires one CUDA GPU")
@pytest.mark.gpu
def test_fusion_disabled_map_batches_materializes_arrow_blocks():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, num_gpus=1, include_dashboard=False)
    context = ray.data.DataContext.get_current()
    rgf.enable(context=context, fusion=False)

    class Identity:
        def __call__(self, batch):
            return batch

    try:
        dataset = ray.data.from_items([{"value": 1}, {"value": 2}]).map_batches(
            Identity,
            batch_format="cudf",
            batch_size=2,
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )
        bundles = list(dataset.iter_internal_ref_bundles())
        blocks = [ray.get(entry.ref) for bundle in bundles for entry in bundle.blocks]
        assert blocks
        assert all(isinstance(block, pa.Table) for block in blocks)
        explanation = rgf.explain(dataset)
        assert "GPU standalone" in explanation
        assert "external_boundary=ray_arrow_blocks" in explanation
    finally:
        rgf.disable(context=context)
        if started_here:
            ray.shutdown()


@pytest.mark.skipif(not _gpu_available(), reason="requires one CUDA GPU")
@pytest.mark.gpu
def test_parquet_map_fusion_runs_in_an_admission_managed_ray_actor(tmp_path):
    class AddOne:
        def __call__(self, batch):
            result = batch.copy(deep=True)
            result["plus_one"] = result["value"] + 1
            return result

    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3, 4]}), path, row_group_size=2)

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, num_gpus=1, include_dashboard=False)
    context = ray.data.DataContext.get_current()
    previous_v2 = context.use_datasource_v2
    context.use_datasource_v2 = False
    rgf.enable(context=context)

    try:
        dataset = ray.data.read_parquet(str(path), override_num_blocks=1).map_batches(
            AddOne,
            batch_format="cudf",
            batch_size=2,
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=0,
        )

        physical = optimized_physical_plan(dataset._logical_plan)
        regions = [
            operator
            for operator in _reachable(physical.dag)
            if isinstance(operator, ActorPoolGPUOperator)
        ]
        assert len(regions) == 1
        assert tuple(
            transform.kind for transform in regions[0].gpu_fusion_spec.transforms
        ) == ("read_parquet", "map_batches")
        assert regions[0].resource_admission_spec().unit_resources is not None

        explanation = rgf.explain(dataset)
        assert "GPU fused" in explanation
        assert "'read_parquet', 'map_batches'" in explanation
        assert sorted(dataset.take_all(), key=lambda row: row["value"]) == [
            {"value": 1, "plus_one": 2},
            {"value": 2, "plus_one": 3},
            {"value": 3, "plus_one": 4},
            {"value": 4, "plus_one": 5},
        ]
    finally:
        rgf.disable(context=context)
        context.use_datasource_v2 = previous_v2
        if started_here:
            ray.shutdown()
