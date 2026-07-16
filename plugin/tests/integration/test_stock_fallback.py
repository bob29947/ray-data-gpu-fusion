from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

import ray
import ray.data

import ray_data_gpu_fusion as rgf


def test_unsupported_map_batches_executes_on_stock_ray():
    """A planning decline must leave an ordinary executable Ray plan."""

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, include_dashboard=False)
    context = ray.data.DataContext.get_current()
    rgf.enable(context=context)

    def add_one(batch: pa.Table) -> pa.Table:
        return batch.append_column(
            "plus_one",
            pc.add(batch.column("value"), 1),
        )

    try:
        dataset = ray.data.from_items([{"value": 1}, {"value": 2}]).map_batches(
            add_one,
            batch_format="pyarrow",
            batch_size=2,
        )
        explanation = rgf.explain(dataset)
        assert "stock:" in explanation
        assert "batch_format is not 'cudf'" in explanation
        assert sorted(dataset.take_all(), key=lambda row: row["value"]) == [
            {"value": 1, "plus_one": 2},
            {"value": 2, "plus_one": 3},
        ]
    finally:
        rgf.disable(context=context)
        if started_here:
            ray.shutdown()
