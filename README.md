# Ray Data GPU Fusion Plugin

Phase 0 of an external GPU execution backend for Ray Data. The backend is a
separate Python distribution, but it executes as part of Ray's physical plan:

```text
Ray Dataset API
  -> stock logical optimizer and planner
  -> plan-local GPU lowering rule
  -> plan-local GPU fusion rule
  -> Ray StreamingExecutor
  -> Ray-managed GPU actor pools
```

Ray remains responsible for scheduling, ObjectRefs, block ownership,
backpressure, retries, metrics, cancellation, and shutdown. The plugin owns
operator recognition, native GPU composition, and cuDF execution inside those
Ray-managed actors. It does not start a second scheduler or wrap Dataset
actions.

## Repository layout

- `ray-stock/` is an official Ray submodule pinned to an untouched commit.
- `ray-patches/` contains the small, backend-neutral Ray extension seams.
- `plugin/` is the independently packaged `ray-data-gpu-fusion` distribution.
- `environment/` and `pins/` make the runtime and source provenance repeatable.
- `scripts/` builds the derived Ray wheel and validates the installation.

See [`docs/architecture.md`](docs/architecture.md) for the execution and GPU
handoff design and [`docs/phase-0.md`](docs/phase-0.md) for the exact support
matrix.

## Phase 0 scope

The initial backend recognizes eligible `ray.data.read_parquet()` and
`Dataset.map_batches()` nodes. Every recognized node is lowered to a closed,
independently executable Ray physical operator before optional fusion.

Within a fused region, intermediate frames remain in cuDF. Between unfused
regions, the closed-operator contract materializes Arrow-backed Ray blocks, and
demand-driven actor pools hand GPUs from the upstream region to the downstream
region without eager reservation.

MapGroupPartitions, range partitioning, preprocessors, expressions, shuffles,
aggregates, and joins are intentionally outside Phase 0.

## Bootstrap

```bash
git submodule update --init --recursive
./scripts/bootstrap.sh
./scripts/test.sh
```

`.venv` is a Conda prefix despite its conventional name. The bootstrap installs
the derived Ray wheel and installs `plugin/` editable. It never installs Ray
from `ray-stock/` in editable mode and never puts the Ray source checkout on
`PYTHONPATH`. Set `CONDA_EXE=/path/to/conda` if Conda is not discoverable on
`PATH` or at `/opt/miniconda3/bin/conda`.

## Usage

```python
import ray
import ray.data
import ray_data_gpu_fusion as rgf

rgf.enable()

result = (
    # One-file Phase-0 example: avoid a stock post-read SplitBlocks node.
    ray.data.read_parquet(
        "/shared/data/input.parquet",
        columns=["a", "b"],
        override_num_blocks=1,
    )
    .map_batches(
        MyGpuTransform,
        batch_format="cudf",
        batch_size=131_072,
        compute=ray.data.ActorPoolStrategy(size=2),
        num_gpus=1,
    )
)
```

`rgf.explain(result)` reports Phase-0 Read/Map eligibility, fused regions, and
stable reasons for relevant nodes that remain on stock Ray. Planning refusals
fall back to the original executable Ray node. Runtime failures are never
replayed on a different implementation.

Use `rgf.enable(fusion=False)` to prove and inspect the independently
executable closed nodes. Unfused GPU regions exchange Arrow-backed Ray blocks
in Phase 0.

The default bootstrap covers local/shared-filesystem Parquet and cuDF
MapBatches. The S3 adapter additionally requires ambient AWS credentials,
`botocore`, and the RAPIDS 25.12 Python `kvikio` package on every GPU worker;
credentialed S3 execution is not part of the hardware-free validation in this
repository.

## Pinned baseline

- Ray commit: `2741c6461d2bd3e5ff114af67be7a1190453dadd`
- Ray version: `3.0.0.dev0`
- Python ABI: CPython 3.11, Linux x86-64
- RAPIDS: 25.12, CUDA 12

See `pins/stock-ray.json` and `pins/source-snapshot.json` for hashes and source
provenance.
