# External GPU Execution Plugin for Ray Data

Status: Phase 0 implemented against Ray commit
`2741c6461d2bd3e5ff114af67be7a1190453dadd`.

This document describes the `ray-data-gpu-fusion` plugin. The Ray extension
hooks for plan-local physical optimizer rules and backend-neutral Parquet scan
metadata are treated as existing prerequisites. Generic GPU resource admission
is also a required Ray capability, but remains independent of this plugin.

## Summary

`ray-data-gpu-fusion` is an opt-in physical-planning and actor-execution
extension for Ray Data. Applications continue to construct and execute ordinary
Ray Datasets. The plugin participates in Ray's physical optimizer, replaces
supported physical nodes with independently executable GPU nodes, and fuses
compatible adjacent GPU nodes into one actor-local execution region.

Ray remains the owner of the Dataset plan and the distributed execution
lifecycle. The plugin does not wrap Dataset actions, interpret a Dataset graph,
or run a scheduler.

The Phase-0 external contract is:

```text
Arrow-backed Ray blocks
    -> one Ray-managed GPU actor region
    -> Arrow-backed Ray blocks
```

Within one fused region, intermediate values are cuDF frames and remain on the
device. Across separate regions, values are materialized as ordinary
Arrow-backed Ray blocks.

## Goals

- Let an external package provide native GPU implementations for Ray Data
  operations without owning Dataset execution.
- Lower every accepted operation into a closed, independently executable
  physical operator before considering fusion.
- Fuse compatible linear GPU operations without changing their observable
  execution, retry, placement, or data-format contracts.
- Preserve ordinary Ray scheduling, ObjectRef ownership, backpressure,
  spilling, metrics, cancellation, retries, and shutdown.
- Fall back to the exact stock Ray physical node when planning cannot prove
  that the plugin implementation is compatible.
- Keep the plugin architecture extensible to future Ray Data APIs and future
  execution backends.

## Non-goals

- Replacing Ray's logical planner, physical planner, StreamingExecutor, or Ray
  Core scheduler.
- Introducing a plugin-owned Dataset action or graph runner.
- Replaying a failed GPU region on a stock implementation.
- Providing a device-resident boundary between separate Ray physical
  operators in Phase 0.
- Implementing native plugin versions of shuffle, sort, groupby, aggregate,
  join, or multi-input operators in Phase 0.
- Supporting asynchronous UDFs, concurrent calls within one GPU actor, or
  multiple GPUs inside one actor.

## Ownership boundary

Ray owns:

- Dataset construction and logical optimization;
- physical plan construction and final rule ordering;
- actor creation and placement;
- GPU resource allocation and admission;
- ObjectRefs and block ownership;
- streaming, backpressure, and spilling;
- task and actor retries;
- metrics, cancellation, and shutdown.

The plugin owns:

- recognizing supported logical and physical shapes;
- validating whether a stock operation can be reproduced safely;
- describing native GPU transforms and their composition contracts;
- deciding whether adjacent GPU transforms can fuse;
- materializing closed GPU physical operators;
- actor-local cuDF initialization and transform execution;
- GPU-specific Parquet footer planning and exact row-group reads;
- diagnostics for accepted, fused, and declined nodes.

## User API

The public API is intentionally small:

```python
import ray
import ray.data
import ray_data_gpu_fusion as rgf

ray.init()
rgf.enable()

ds = (
    ray.data.read_parquet(
        "/shared/input.parquet",
        columns=["a", "b"],
        override_num_blocks=1,
    )
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

`enable()` must run before Dataset construction because a Dataset captures its
`DataContext`. `enable(fusion=False)` installs lowering but disables the fusion
rule, which exposes the independently executable closed nodes. `disable()`
removes only this plugin's rules from the selected context.

## Ray prerequisites

The plugin checks all required Ray capabilities before changing a context:

1. the exact supported Ray commit;
2. plan-local physical optimizer rule classes on `DataContext`;
3. the backend-neutral Parquet scan descriptor;
4. generic resource-admission capability version 1;
5. elastic resource admission for `ActorPoolMapOperator`; and
6. the resource-admission rollback field being enabled.

Missing or disabled capabilities cause `enable()` to fail closed. The checks
are repeated during physical planning because a context can be mutated after
plugin activation.

All imports from `ray.data._internal` are isolated in the plugin compatibility
module. Supporting another Ray revision should require a new compatibility
adapter rather than spreading version checks through the implementation.

## Planning lifecycle

The physical planning sequence is:

```text
Ray Dataset logical plan
    -> Ray logical optimization
    -> Ray physical planner
    -> Ray read-parallelism rule
    -> plugin closed-node lowering
    -> plugin GPU-region fusion
    -> Ray's normal physical fusion
    -> Ray StreamingExecutor
```

The plugin contributes two plan-local rules:

- `LowerClosedGPUOperators` recognizes and replaces supported nodes.
- `FuseClosedGPUOperators` combines compatible linear candidates.

Both rules return ordinary Ray `PhysicalPlan` objects. The final DAG contains
ordinary Ray physical operators and is executed by the normal
`StreamingExecutor`.

## Closed GPU operator model

The central correctness rule is:

> Every lowered node must be executable by Ray without being fused.

Fusion is therefore an optimization over already valid nodes. A fusion refusal
leaves a runnable DAG, and `enable(fusion=False)` is a valid execution mode.

Each candidate carries two related descriptions:

- a native fusion specification, containing only the meaningful GPU
  transforms; and
- a materialized execution specification, which adds Ray-block ingress and
  egress transforms required by a selected backend.

For example, the native specification for one map is:

```text
FrameStream -> MapBatches -> FrameStream
```

The actor-pool backend closes it as:

```text
RayBlockStream
    -> ImportRayBlocks
    -> MapBatches
    -> ExportFrameStream
    -> RayBlockStream
```

For a Parquet read, the closed region is:

```text
RayBlockStream
    -> ImportParquetWork
    -> ReadParquet
    -> ExportFrameStream
    -> RayBlockStream
```

## Specification model

### Payloads

Transforms declare explicit input and output payload kinds:

| Payload | Meaning | Materializable boundary |
| --- | --- | --- |
| `RayBlockStream` | Ordinary Ray blocks crossing a physical edge | Yes |
| `ParquetWork` | Actor-local exact row-group work descriptor | No |
| `FrameStream` | Actor-local stream of cuDF frames | No |
| `DeviceHandle` | Reserved for a future device-resident boundary | Yes |

A materializable payload may be used between physical operators, but it cannot
appear in the middle of one fused region.

### Transform specification

A transform specification contains:

- a stable runtime ID;
- a transform kind;
- input and output payload kinds; and
- immutable transform configuration.

Examples are `read_parquet`, `map_batches`, `import_ray_blocks`, and
`export_frame_stream`.

### Operator specification

An operator specification contains:

- an ordered tuple of transforms;
- execution requirements;
- required data properties; and
- provided data properties.

Adjacent transform payloads must match. Required/provided properties give
future adapters a place to express ordering, partitioning, schema, or grouping
contracts without adding special cases to the fusion algorithm.

### Execution requirements

Phase 0 uses an actor-pool execution profile containing:

- minimum, initial, and maximum actor counts;
- exactly one task in flight per actor;
- one GPU per actor;
- CPU, memory, and custom resources;
- accelerator type and label selector;
- scheduling strategy and runtime environment;
- actor and map retry policy; and
- actor bootstrap requirements such as the allocator configuration and direct
  read source.

Fusion must not silently change any of these semantics.

## Lowering MapBatches

The Phase-0 adapter accepts only an exact `MapBatches` logical operator with:

- a synchronous callable or callable class;
- `batch_format="cudf"`;
- an explicit positive integer `batch_size`;
- a finite `ActorPoolStrategy`;
- one GPU per actor;
- one task in flight per actor;
- no per-block limit; and
- no dynamic `ray_remote_args_fn`.

The adapter extracts the UDF, constructor arguments, invocation arguments,
batch size, zero-copy setting, actor-pool shape, resources, runtime
environment, and retry behavior into immutable specifications.

The standalone node uses Ray's already-planned stock `MapTransformer`. This is
the narrowest way to preserve stock MapBatches behavior when the node remains
unfused. The candidate additionally carries a native `map_batches` transform
that the plugin runtime uses when fusion succeeds.

## Lowering Parquet reads

The Parquet adapter accepts an exact legacy Ray `Read` backed by the exact
`ParquetDatasource` type. It requests the neutral external scan descriptor and
declines the read when reproducing its semantics is not proven safe.

Phase-0 restrictions include:

- local/shared filesystem or ambient-credential S3;
- a nonempty ordinary column projection;
- no predicate or partition columns;
- no custom filesystem or schema override;
- no shuffle, callback, path column, or row hash;
- unordered execution; and
- no additional stock split factor.

### Planning-time footer work

After recognition, the plugin:

1. obtains a stable identity for every source object;
2. verifies that its current size matches the size recorded by Ray;
3. reads Parquet footer metadata;
4. verifies that the source did not change during footer inspection;
5. validates projected columns and row-group size statistics;
6. balances exact row groups across the target number of tasks; and
7. serializes one `ParquetWork` descriptor per task into an input block.

The balancing order prioritizes uncompressed bytes, compressed bytes, and row
count. Assignments are deterministic and receive a content-derived scheme ID.

### Actor-time read

The actor receives exactly one `ParquetWork` descriptor. It revalidates source
identity before and after each read and asks cuDF to read only the assigned row
groups and projected columns. Row groups are read in bounded chunks to avoid an
unbounded call for files with many row groups.

Local reads use cuDF directly. S3 reads require ambient credentials and the
pinned KvikIO stack; credentials are refreshed for actor tasks. Endpoint
overrides are declined because Phase 0 cannot prove equivalent native access.

## Fusion algorithm

Fusion considers only linear physical edges. A candidate must have one
consumer, and the consumer must have exactly that candidate as its one input.
Branches, fan-in, exchanges, and multi-input operators form region boundaries.

Two adjacent specifications compose only when:

1. the upstream output payload equals the downstream input payload;
2. the connecting payload is not a materializable boundary;
3. upstream properties satisfy downstream requirements;
4. the execution backend sets intersect; and
5. execution profiles merge without changing semantics.

Profile compatibility includes actor-pool bounds, task concurrency, CPU/GPU
resources, retry policy, memory, custom resources, placement constraints,
runtime environment, and actor bootstrap configuration.

The rule selects maximal materializable subregions. A backend or composition
refusal leaves the original executable GPU candidates in place.

For a compatible chain:

```text
ReadParquet -> MapBatches -> MapBatches
```

Ray receives one physical actor-pool operator whose actor-local program is:

```text
ImportParquetWork
    -> ReadParquet
    -> MapBatches
    -> MapBatches
    -> ExportFrameStream
```

## Backend materialization

Execution backends are selected through a registry rather than hard-coded in
the fusion rule. A backend factory must:

1. prepare a closed execution specification by adding materialized
   boundaries;
2. report whether every transform has a registered runtime; and
3. create an executable Ray physical operator.

Phase 0 registers only the `actor_pool` backend. It produces an
`ActorPoolMapOperator` with:

- a `MapTransformer` containing one closed region transform;
- the resolved actor-pool strategy;
- the exact Ray remote arguments from the execution profile;
- stock Ray block bundling and target block-size settings; and
- elastic resource-admission participation.

The created operator explicitly disables Ray's stock map-operator fusion. The
plugin has already compiled its native region, and a second fusion mechanism
must not rewrite its closed boundary contract.

## Actor-local runtime

The runtime is a registry keyed by:

```text
(runtime ID, transform kind, input payload, output payload)
```

Each registration contains a configuration validator and a runtime factory.
This keeps planning specifications serializable and lets future transform
adapters add runtimes without changing the region executor.

One `RegionWorker` is created per Ray actor process. During initialization it:

1. resolves transform registrations;
2. creates runtime instances in transform order; and
3. initializes shared actor-local state once.

During a task it passes the current value through each runtime in order.

### CUDA and allocator initialization

The shared cuDF context initializes lazily inside the actor process. It:

- inspects free device memory;
- creates an RMM pool with a bounded maximum;
- configures CuPy to use the RMM allocator;
- imports cuDF only after allocator setup; and
- initializes the optional S3/KvikIO state when required.

The pool leaves at least 2 GiB outside the plugin pool and caps itself at 70%
of observed free memory.

### MapBatches semantics

The native runtime rebatches cuDF frames to the requested batch size and
preserves stock Ray behavior for:

- callable classes and constructor arguments;
- zero-copy versus copied inputs;
- Python generator outputs;
- invalid user outputs and `UserCodeException` wrapping;
- typed empty batches; and
- schema-less empty output when no batch is produced.

Only Python generator results are flattened, matching the pinned Ray
MapBatches contract.

### Region egress

The egress runtime converts cuDF frames into Arrow tables. It materializes all
output blocks before yielding the first result. Consequently, an I/O, CUDA, or
UDF failure is reported atomically through the current Ray actor task instead
of after partial output has escaped the region.

## Boundaries between regions

Two separate GPU regions exchange ordinary Arrow-backed ObjectRefs:

```text
GPU region A
    -> cuDF-to-Arrow conversion
    -> Ray Object Store
    -> Arrow-to-cuDF conversion
    -> GPU region B
```

This boundary costs device/host conversion and Object Store capacity, but it
provides clear ownership, retry, backpressure, spill, and failure semantics.
It also makes one-GPU handoff possible: upstream output can queue or spill
while Ray's resource-admission controller transfers the GPU to the downstream
actor pool.

A device-resident inter-operator payload is reserved for a later phase. It
requires an explicit ownership, locality, reconstruction, and spill contract
before it can replace the Arrow boundary.

## Resource admission

Every plugin-created actor region must expose Ray's generic `ELASTIC_POOL`
admission specification. The plugin verifies this immediately after creating
the physical operator.

The plugin does not implement admission policy. Ray's resource manager decides
which physical operators may create and retain GPU actors, protects a
one-complete-actor progress floor, prevents later stages from leapfrogging a
non-fitting frontier, and caps actor-pool scaling through grants.

This separation is intentional:

- the plugin describes its ordinary actor resources;
- `ActorPoolMapOperator` adapts them to the generic admission contract; and
- Ray coordinates those resources across the complete physical DAG.

Stock actor-based GPU operations and plugin-created regions therefore follow
the same lifecycle.

## Fallback and failure semantics

Planning and execution have different policies.

### Planning refusal

Expected capability failures are conservative fallbacks:

- an unsupported operation remains the exact stock Ray node;
- an unsupported Parquet scan remains the stock reader;
- an incompatible edge remains two independently executable GPU nodes; and
- a backend materialization refusal preserves the original runnable DAG.

Stable refusal reasons are attached to relevant stock nodes for diagnostics.
Unexpected programming errors are not converted into fallback decisions.

### Runtime failure

Once execution starts, the selected physical plan is authoritative. I/O,
source mutation, CUDA, actor, and user-code failures follow Ray's normal retry
and failure path. The plugin never replays completed or partially completed
work on a stock implementation.

This avoids duplicate side effects and prevents a UDF's implementation from
changing after a runtime error.

## Diagnostics

`rgf.explain(dataset)` runs Ray's normal logical optimizer, physical planner,
and physical optimizer without executing the Dataset. It reports:

- whether the plugin and fusion are enabled;
- Ray's ordinary plan explanation;
- standalone plugin GPU nodes;
- fused transform sequences; and
- stable reasons for relevant nodes that stayed on stock Ray.

The diagnostic path does not maintain a second planner or reinterpret the
Dataset graph.

## Package structure

```text
ray_data_gpu_fusion/
    __init__.py      public API
    config.py        plan-local enablement
    _compat.py       pinned Ray adapter and capability checks
    specs.py         immutable payload and execution contracts
    rules.py         recognition, lowering, and fusion
    operators.py     backend registry and Ray physical operators
    runtime.py       actor-local transform registry and execution
    parquet.py       Parquet planning and cuDF runtime
    diagnostics.py   explain output
```

Ray-private imports remain isolated in `_compat.py`. Planning contracts remain
in `specs.py`; Ray operator construction remains in `operators.py`; actor-local
execution remains in `runtime.py`. These boundaries should be preserved when
new APIs are added.

## Adding another Ray Data operation

A new plugin adapter should provide four pieces:

1. **Recognition:** match one exact logical/physical shape and decline every
   unsupported option with a stable reason.
2. **Native specification:** emit one or more transforms with explicit payload,
   property, execution, and retry contracts.
3. **Runtime registration:** register actor-local implementations for each new
   transform key.
4. **Standalone materialization:** provide a closed execution path with normal
   Ray-block endpoints before allowing the transform to fuse.

Simple linear operations such as filter or scalar projection can use
`FrameStream -> FrameStream`. A native read can use a work-descriptor payload
similar to Parquet. Operations involving grouping, ordering, exchanges, or
multiple inputs must introduce explicit property and boundary contracts rather
than special-casing the current fusion algorithm.

A native GPU sort, for example, must first answer:

- whether it is a local per-block sort or a global range exchange;
- what ordering and partitioning properties it requires and provides;
- whether it needs an atomic GPU gang;
- where materialization occurs;
- how output ownership and retry work; and
- which neighboring operations may legally move inside its region.

## Phase-0 support matrix

| Shape | Plugin behavior |
| --- | --- |
| Eligible Parquet read | Exact row-group GPU read candidate |
| Eligible cuDF MapBatches | Standalone actor candidate and native fusion transform |
| Compatible linear chain | One fused actor-pool physical operator |
| Incompatible GPU chain | Separate admitted actor pools with Arrow between them |
| Unsupported option or shape | Exact stock Ray physical implementation |
| Stock GPU shuffle or hash aggregate | Not plugin-fused; still covered by Ray resource admission |

## Validation strategy

The plugin is validated at four levels:

1. **Specification and runtime unit tests** cover payload composition,
   execution-profile compatibility, batching, empty outputs, source mutation,
   and fusion selection.
2. **Ray contract tests** verify H1/H2, capability versioning, plan locality,
   failure-closed behavior, and resource-admission participation.
3. **Integration tests** execute standalone Arrow boundaries, fused Parquet-map
   regions, and stock fallback.
4. **Stock-versus-plugin smoke tests** run in separate processes and compare
   output digests before reporting timing.

Correctness and ownership invariants are gates. Phase 0 intentionally has no
required speedup threshold.

## Deferred work

- Device-resident boundaries between separate physical operators.
- DataSource V2 Parquet recognition.
- Native filter, expression, preprocessor, and projection adapters.
- Native plugin sort, shuffle, groupby, aggregate, and join regions.
- GPU task-pool admission.
- Asynchronous or concurrently invoked GPU UDFs.
- Multi-GPU execution within one actor.
- Credentialed S3 hardware validation in the default test suite.

These additions should extend the payload, property, backend, and runtime
registries rather than weakening the closed-node or exact-compatibility rules.
