#!/usr/bin/env python3
"""Correctness-first local stock/plugin smoke benchmark.

The two arms run in fresh subprocesses so the plan-local plugin state and Ray
runtime cannot leak between them. There is intentionally no speedup gate in
Phase 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class AddFeatures:
    def __call__(self, frame):
        frame = frame.copy(deep=True)
        frame["feature"] = frame["value"] * 7
        return frame


def _child(mode: str, parquet: str) -> int:
    import pyarrow as pa
    import ray
    import ray.data

    ray.init(num_cpus=4, num_gpus=1, include_dashboard=False)
    if mode == "plugin":
        import ray_data_gpu_fusion as rgf

        rgf.enable()

    dataset = ray.data.read_parquet(parquet).map_batches(
        AddFeatures,
        batch_format="cudf",
        batch_size=4096,
        compute=ray.data.ActorPoolStrategy(size=1),
        num_gpus=1,
    )
    started = time.perf_counter()
    table = dataset.materialize().to_arrow_refs()
    blocks = ray.get(table)
    combined = pa.concat_tables(blocks)
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256(
        combined.combine_chunks().to_pandas().sort_values("id").to_json().encode()
    ).hexdigest()
    print(
        "RESULT="
        + json.dumps(
            {
                "mode": mode,
                "rows": combined.num_rows,
                "digest": digest,
                "elapsed_s": elapsed,
                "stats": dataset.stats(),
            },
            sort_keys=True,
        )
    )
    ray.shutdown()
    return 0


def _parse_result(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT="):
            return json.loads(line.removeprefix("RESULT="))
    raise RuntimeError(f"benchmark child produced no result:\n{output}")


def _orchestrate() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory(prefix="ray-gpu-fusion-smoke-") as tmp:
        path = Path(tmp) / "input.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": pa.array(range(50_000)),
                    "value": pa.array((index % 997 for index in range(50_000))),
                }
            ),
            path,
            row_group_size=5_000,
        )
        results = []
        for mode in ("stock", "plugin"):
            completed = subprocess.run(
                [sys.executable, __file__, "--child", mode, str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            results.append(_parse_result(completed.stdout))
        if results[0]["digest"] != results[1]["digest"]:
            raise RuntimeError(f"stock/plugin result mismatch: {results}")
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("stock", "plugin"))
    parser.add_argument("parquet", nargs="?")
    args = parser.parse_args()
    if args.child:
        if not args.parquet:
            parser.error("a Parquet path is required in child mode")
        return _child(args.child, args.parquet)
    return _orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())
