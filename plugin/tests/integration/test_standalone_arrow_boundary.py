from __future__ import annotations

import pyarrow as pa
import pytest

import ray
import ray.data
from ray.data import ActorPoolStrategy

import ray_data_gpu_fusion as rgf


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
