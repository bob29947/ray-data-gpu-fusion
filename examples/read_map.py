#!/usr/bin/env python3
"""Run an ordinary Ray Data Parquet -> cuDF MapBatches pipeline."""

from __future__ import annotations

import argparse
import json


class AddFeatures:
    def __init__(self, *, multiplier: int) -> None:
        self._multiplier = multiplier

    def __call__(self, frame):
        frame = frame.copy(deep=True)
        frame["feature"] = frame["value"] * self._multiplier
        return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--no-fusion", action="store_true")
    args = parser.parse_args()

    import ray
    import ray.data
    import ray_data_gpu_fusion as rgf

    ray.init()
    rgf.enable(fusion=not args.no_fusion)
    dataset = ray.data.read_parquet(args.input, override_num_blocks=1).map_batches(
        AddFeatures,
        fn_constructor_kwargs={"multiplier": 3},
        batch_format="cudf",
        batch_size=args.batch_size,
        compute=ray.data.ActorPoolStrategy(size=args.actors),
        num_gpus=1,
    )
    print(rgf.explain(dataset))
    materialized = dataset.materialize()
    print(json.dumps({"rows": materialized.count()}, sort_keys=True))
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
