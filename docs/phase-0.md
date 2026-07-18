# Phase 0 contract

## Supported fast path

| Ray Data shape | Phase-0 requirement | Result |
| --- | --- | --- |
| `read_parquet()` | Legacy V1 Parquet datasource; ordinary local/shared path or ambient-credential S3; unordered execution with no predicate, partition columns, custom filesystem, schema override, shuffle, read callback, or extra stock split factor (`additional_split_factor == 1`) | Exact row-group cuDF read candidate |
| `map_batches()` | Exact `MapBatches`; synchronous callable; explicit positive integer `batch_size`; `batch_format="cudf"`; finite actor pool; one GPU and one in-flight task per actor | Standalone GPU actor candidate and native fusion transform |
| Linear compatible chain | Matching execution and payload contracts | One fused Ray actor-pool operator |
| Linear incompatible chain | Any contract differs | Separate admission-controlled Ray actor pools with Arrow blocks between them |

Everything else remains on its stock Ray implementation.

Eligible GPU actor pools with statically declared per-actor resources use Ray's
capability-version-1 admission controller when operator reservation is enabled,
`wait_for_min_actors_s <= 0`, and no user-supplied dynamic `ray_remote_args_fn`
is configured.
Fixed and autoscaling pool sizes are both eligible. Unsupported or internally
disabled cases retain stock actor behavior.

## Intentionally deferred

- device-resident data exchange between separate regions;
- device-resident streaming across separate GPU regions (ordinary Arrow
  boundaries may overlap only when Ray admits both actor pools);
- DataSource V2 Parquet recognition;
- scalar-expression and preprocessor adapters;
- partitioning, grouped partitions, shuffles, sort, aggregate, join, and zip;
- multi-GPU work inside one actor;
- asynchronous or concurrently invoked UDFs;
- custom object-store or scheduler behavior.

## User code

The Dataset program remains ordinary Ray code. Activation must happen before
the Dataset is constructed:

```python
import ray
import ray.data
import ray_data_gpu_fusion as rgf

ray.init()
rgf.enable()

ds = (
    # For this one-file example, requesting one block avoids Ray inserting a
    # stock post-read SplitBlocks node, which Phase 0 deliberately declines.
    ray.data.read_parquet("/shared/input.parquet", override_num_blocks=1)
    .map_batches(
        MyTransform,
        batch_format="cudf",
        batch_size=131_072,
        compute=ray.data.ActorPoolStrategy(size=2),
        num_gpus=1,
    )
)

print(rgf.explain(ds))
result = ds.materialize()
```

Use `rgf.enable(fusion=False)` to inspect or test the independently executable
closed nodes and `rgf.disable()` to remove only this plugin's rules from the
current context.
