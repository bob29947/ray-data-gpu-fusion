# Phase 0 contract

## Supported fast path

| Ray Data shape | Phase-0 requirement | Result |
| --- | --- | --- |
| `read_parquet()` | Legacy V1 Parquet datasource; ordinary local/shared path or ambient-credential S3; unordered execution with no predicate, partition columns, custom filesystem, schema override, shuffle, read callback, or extra stock split factor | Exact row-group cuDF read candidate |
| `map_batches()` | Exact `MapBatches`; synchronous callable; explicit positive integer `batch_size`; `batch_format="cudf"`; finite actor pool; one GPU and one in-flight task per actor | Standalone GPU actor candidate and native fusion transform |
| Linear compatible chain | Matching execution and payload contracts | One fused Ray actor-pool operator |
| Linear incompatible chain | Any contract differs | Separate demand-driven Ray actor pools with Arrow blocks between them |

Everything else remains on its stock Ray implementation.

## Intentionally deferred

- device-resident data exchange between separate regions;
- streaming overlap across separate GPU regions (Phase 0 drains the upstream
  region and may buffer its full Arrow output before the next pool starts);
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
    ray.data.read_parquet("/shared/input")
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
