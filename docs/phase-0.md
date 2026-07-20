# Phase 0 contract

## Supported fast path

| Ray Data shape | Phase-0 requirement | Result |
| --- | --- | --- |
| `read_parquet()` | Legacy V1 Parquet datasource; ordinary local/shared path or ambient-credential S3; unordered execution with no predicate, partition columns, custom filesystem, schema override, shuffle, read callback, or extra stock split factor (`additional_split_factor == 1`) | Exact row-group cuDF read candidate |
| `map_batches()` | Exact `MapBatches`; synchronous callable; explicit positive integer `batch_size`; `batch_format="cudf"`; finite actor pool; one GPU and one in-flight task per actor | Standalone GPU actor candidate and native fusion transform |
| Linear compatible chain | Matching execution and payload contracts | One fused Ray actor-pool operator |
| Linear incompatible chain | Any contract differs | Separate admission-controlled Ray actor pools with Arrow blocks between them |

Everything else remains on its stock Ray implementation. The generic resource
admission contract below changes when supported stock GPU resource owners may
acquire resources; it does not make those APIs plugin fusion candidates.

## Ray resource-admission coverage

Candidate C exposes internal resource-admission capability version 1. The
first production adapters cover these physical resource owners:

| Ray Data operation | Physical owner | Admission behavior |
| --- | --- | --- |
| Actor-based GPU `map_batches()` | `ActorPoolMapOperator` | Elastic pool with a one-complete-actor progress floor and scaling capped by its grant |
| Actor-based `map_groups(batch_format="cudf")` | The same `ActorPoolMapOperator` adapter | Elastic pool; cuDF conversion remains inside each per-group actor call |
| GPU shuffle | `GPUShuffleOperator` rank pool | One atomic fixed gang containing every configured rank |
| GPU hash aggregate | GPU shuffle base implementation | The same fixed-gang lifecycle, inherited without a controller special case |

The complete shuffle gang is one admission floor. Actor pools defer worker
startup until admitted, while a shuffle placement group becomes usable only
after every rank is ready. Fixed and autoscaling actor pools are both eligible.
`wait_for_min_actors_s` readiness is tracked asynchronously after admission, so
it does not block topology construction.

Deadlock-safety floors remain active when
`op_resource_reservation_enabled=False`; that setting controls proportional
sharing, not admission safety. A GPU actor pool with a user-supplied dynamic
`ray_remote_args_fn` and no static resource envelope, a GPU task, or another
GPU operator without an admission specification keeps legacy scheduling and
emits a once-per-execution warning that deadlock protection does not apply.
The internal `DataContext` rollback field can disable candidate C as a whole.

## Intentionally deferred

- device-resident data exchange between separate regions;
- device-resident streaming across separate GPU regions (ordinary Arrow
  boundaries may overlap only when Ray admits both actor pools);
- DataSource V2 Parquet recognition;
- scalar-expression and preprocessor adapters;
- plugin-native lowering or fusion for grouped partitions, shuffles, sort,
  aggregate, join, and zip (stock actor-based cuDF `map_groups`, GPU shuffle,
  and GPU hash aggregate still receive the admission behavior above);
- admission gating for GPU task pools and undeclared future GPU operators;
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
