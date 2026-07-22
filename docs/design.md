# External GPU Execution Backends for Ray Data

**Status:** Draft design; Phase-0 prototype implemented and validated  
**Scope:** Ray Data physical planning and execution  
**Initial backend:** cuDF Parquet read and `map_batches`

## 1. Problem

Ray Data already lets users run GPU functions with APIs such as
`Dataset.map_batches(..., batch_format="cudf", num_gpus=1)`. Ray also already
provides the distributed systems machinery that a GPU backend needs: logical
planning, streaming execution, actor scheduling, ObjectRefs, backpressure,
retries, cancellation, metrics, and resource accounting.

What Ray Data does not provide is a clean way for an external execution backend
to participate in physical planning. A backend that wants to recognize an
existing Ray Data operation, replace its physical implementation, and fuse it
with adjacent operations currently has unattractive choices:

1. put backend-specific implementation directly in Ray;
2. maintain a large Ray fork;
3. expose a second Dataset API or action wrapper;
4. monkey-patch Ray's global planner state; or
5. run a second graph executor beside Ray.

Each choice creates the wrong ownership boundary. Putting cuDF-specific
planning and execution in Ray expands Ray's maintenance surface and couples GPU
library releases to Ray releases. A second API or executor stops being ordinary
Ray Data: it must reproduce scheduling, failure, block ownership, and lifecycle
semantics that Ray already implements.

There are four additional execution problems that a useful backend must solve.

### 1.1 GPU operations need to be modular before they are fused

Fusion cannot be required for correctness. An eligible GPU operation must have
a complete standalone implementation with an input boundary, actor-local
execution, and an output boundary. Otherwise a failed fusion decision can leave
an operator that cannot execute.

The desired invariant is:

> Every accepted GPU node is independently executable. Fusion only removes
> boundaries between compatible executable nodes.

### 1.2 Ray blocks and GPU batches solve different problems

A Ray block is an ObjectRef-backed storage and scheduling unit. A map batch is
one UDF invocation inside a Ray map task. One large Ray block can be divided
into many `batch_size`-sized cuDF batches, but those batches normally still run
serially within one actor task. Additional Ray blocks are what create work that
can be scheduled across multiple actors and GPUs.

Ray therefore sometimes appends `SplitBlocks(k)` to a read. For example, one
Parquet read task may produce one Arrow block while Ray wants 192 downstream
blocks. The read task reads once and then slices its Arrow output into smaller
row ranges. A replacement reader must either preserve that block-count contract
or explicitly decline the replacement.

### 1.3 Separate GPU regions can compete for the same finite GPUs

Consider two GPU regions that cannot fuse. If both eagerly create actor pools
whose configured size equals the cluster GPU count, the upstream pool can hold
all GPUs while the downstream pool waits, or both pools can partially reserve
resources without either making useful progress. Resource admission must be
coordinated by Ray's allocator and scheduler.

### 1.4 CPU and GPU operations require explicit data boundaries

A mixed pipeline can legitimately contain:

```text
GPU read -> CPU operator -> GPU map
```

An ordinary Ray CPU operator consumes host-side Ray blocks, not an actor-local
cuDF frame. Phase 0 therefore needs a correctness-first boundary:

```text
GPU region -> Arrow ObjectRef -> CPU region -> Arrow ObjectRef -> GPU region
```

This may perform extra device-to-host and host-to-device transfers. Avoiding
those transfers is not a Phase-0 requirement; preserving Ray semantics and
operator modularity is.

## 2. Proposed solution

Add a small, backend-neutral extension contract to Ray Data and keep all GPU
recognition, composition, and execution in an external plugin.

The user continues to write an ordinary Ray Dataset program:

```python
import ray
import ray.data
import ray_data_gpu_fusion as rgf

ray.init()
rgf.enable()

ds = (
    ray.data.read_parquet("/shared/input.parquet")
    .map_batches(
        TransformA,
        batch_format="cudf",
        batch_size=131_072,
        compute=ray.data.ActorPoolStrategy(size=2),
        num_gpus=1,
    )
    .map_batches(
        TransformB,
        batch_format="cudf",
        batch_size=131_072,
        compute=ray.data.ActorPoolStrategy(size=2),
        num_gpus=1,
    )
)

print(rgf.explain(ds))
result = ds.materialize()
```

The program is still planned and executed by Ray:

```text
Ray Dataset API
        |
        v
Ray logical optimizer and physical planner
        |
        v
plugin rules invoked by Ray's PhysicalOptimizer
  1. recognize and lower independently executable GPU candidates
  2. fuse compatible linear candidate regions
        |
        v
Ray StreamingExecutor
        |
        v
Ray Core actors, ObjectRefs, resources, retries, metrics, and shutdown
```

The plugin does not provide an action wrapper, scheduler, graph runner, object
store, or resource manager.

The proposal has three parts:

1. **Ray extension contract.** Ray exposes generic resource-aware GPU actor
   admission, plan-local physical rules, and a neutral Parquet scan descriptor.
2. **External backend plugin.** The plugin recognizes Ray operations, creates
   closed GPU candidates, composes compatible candidates, and supplies
   actor-local cuDF runtimes.
3. **Arrow external boundary.** Every standalone Phase-0 GPU region accepts or
   produces ordinary Ray blocks. cuDF remains actor-local and may persist only
   across transforms fused into the same region.

### 2.1 Relationship to the existing closed implementation

The initial plugin is not a from-scratch reimplementation. It is a scoped
extraction and adaptation of the closed GPU operator implementation originally
developed inside the modified Ray tree. The operator specification algebra,
runtime registry, cuDF MapBatches execution, exact-row-group Parquet planner,
actor backend, and fusion algorithm moved into the plugin.

The integration boundary was rewritten: Ray-specific GPU planner changes were
replaced with the generic candidate and two local hooks described below. Grouped
partitions, range planning, and shuffle support were intentionally not extracted
for Phase 0. The dirty-tree source used for the extraction is recorded in
[`source-snapshot.json`](../pins/source-snapshot.json).

## 3. Goals and non-goals

### 3.1 Goals

- Keep the Ray Dataset API and normal Dataset actions unchanged.
- Keep GPU library imports and backend-specific logic outside Ray.
- Make every accepted GPU operation independently executable.
- Fuse compatible operations without making fusion necessary for correctness.
- Preserve the exact stock physical node when planning declines a replacement.
- Use Ray actors, resources, ObjectRefs, retries, and lifecycle behavior.
- Allow separate GPU regions to share a constrained GPU pool without deadlock.
- Provide stable explanations for GPU selection and refusal.
- Make future scalar, encoder, and preprocessor adapters use the same execution
  and composition substrate.

### 3.2 Phase-0 non-goals

- Device-resident exchange between separate physical regions.
- A GPU-aware Ray Object Store or custom scheduler.
- Cost-based CPU/GPU placement.
- Grouped partitions, shuffles, sort, aggregate, join, or zip.
- Multi-GPU work inside one actor.
- Asynchronous or concurrently invoked UDFs.
- Support for every Parquet option or filesystem.
- Automatic runtime fallback after GPU execution has started.

## 4. Ownership boundary

| Concern | Owner |
| --- | --- |
| Dataset API and logical plan | Ray |
| Built-in logical and physical planning | Ray |
| Invocation of external physical rules | Ray |
| GPU recognition and refusal reasons | Plugin |
| Closed operator and payload specifications | Plugin |
| Fusion compatibility | Plugin |
| cuDF, RMM, CuPy, and KvikIO execution | Plugin |
| Actor creation, placement, and resources | Ray |
| ObjectRefs, block ownership, and backpressure | Ray |
| Retries, cancellation, metrics, and shutdown | Ray |
| Phase-0 inter-region data format | Arrow Ray blocks |

This ownership boundary is the central design decision. Ray exposes generic
mechanisms; the plugin supplies policy and GPU implementation.

## 5. Ray extension contract

The prototype keeps generic admission separate from the two plugin-extension
hooks. None contains cuDF, CUDA, fusion logic, or plugin imports.

| Layer | Capability | Provenance |
| --- | --- | --- |
| C | Generic admission for static GPU actor pools and fixed GPU shuffle/hash-aggregate gangs | `pins/pr-candidate.json` records the exact local commit, changed files, LOC, tree, patch, and wheel hashes |
| H1 | Plan-local physical optimizer rules | First and only first hook in `ray-hooks/` |
| H2 | Backend-neutral Parquet scan descriptor | Second and only second hook in `ray-hooks/` |

The snippets below omit unrelated unchanged code and some validation branches;
the complete C/H1/H2 diffs are linked in each subsection.

### 5.1 Plan-local physical optimizer rules

#### Why this change is needed

Ray's physical optimizer normally constructs a fixed built-in ruleset. An
external package can construct a `PhysicalOperator`, but it has no supported
place to inspect and replace nodes after Ray's planner has produced the
physical DAG. Registering rules in process-global state would also be incorrect:
two Datasets in the same process may enable different plugins, and planning may
happen in a different process from Dataset construction.

The extension therefore belongs on `DataContext`, which Ray already copies into
the logical and physical plan. The rule list becomes a property of one Dataset
plan rather than a global plugin registry.

#### Where it lives

- `python/ray/data/context.py`
- `python/ray/data/_internal/logical/optimizers.py`
- Full diff: [`0001-ray-data-support-plan-local-physical-optimizer-rules.patch`](../ray-hooks/0001-ray-data-support-plan-local-physical-optimizer-rules.patch)

#### Ray code

`DataContext` carries importable rule classes:

```python
# python/ray/data/context.py

# Physical optimizer extensions attached to plans created with this context.
# Keeping these classes on DataContext avoids process-global rule registration
# and propagates the extension to remote Dataset planners.
custom_physical_optimizer_rule_classes: List[Type["Rule"]] = field(
    default_factory=list
)
```

`PhysicalOptimizer` combines the built-in and plan-local rules while preserving
Ray's existing dependency ordering:

```python
# python/ray/data/_internal/logical/optimizers.py

class PhysicalOptimizer(Optimizer):
    """The optimizer for physical operators."""

    def __init__(self):
        self._custom_rule_classes = ()

    @property
    def rules(self) -> List[Rule]:
        ruleset = Ruleset()
        seen = set()
        for rule_cls in (*get_physical_ruleset(), *self._custom_rule_classes):
            if not isinstance(rule_cls, type) or not issubclass(rule_cls, Rule):
                raise TypeError(
                    "custom_physical_optimizer_rule_classes must contain "
                    f"Rule subclasses, got {rule_cls!r}"
                )
            if rule_cls in seen:
                continue
            ruleset.add(rule_cls)
            seen.add(rule_cls)
        return [rule_cls() for rule_cls in ruleset]

    def optimize(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Optimize using the stock rules plus this plan's context-local rules."""

        self._custom_rule_classes = tuple(
            plan.context.custom_physical_optimizer_rule_classes
        )
        return super().optimize(plan)
```

The type check prevents arbitrary callables from running inside Ray's optimizer.
Deduplication makes repeated plugin enablement idempotent. `Ruleset` continues to
honor each rule's `dependencies()` and `dependents()` declarations.

#### How the plugin uses it

`rgf.enable()` adds two ordinary Ray `Rule` subclasses to the current context:

```python
# ray_data_gpu_fusion/config.py

classes = get_rule_classes(selected)
for rule in (LowerClosedGPUOperators, FuseClosedGPUOperators):
    if rule not in classes:
        classes.append(rule)
set_rule_classes(selected, classes)
selected.set_config(CONFIG_KEY, FusionSettings(True, bool(fusion)))
```

`LowerClosedGPUOperators` declares `SetReadParallelismRule` as a dependency so
it sees Ray's final read shape. `FuseClosedGPUOperators` declares the lowering
rule as a dependency and Ray's stock `FuseOperators` as a dependent. The order
is therefore:

```text
Ray read parallelism
  -> plugin closed lowering
  -> plugin GPU fusion
  -> remaining Ray physical rules
```

### 5.2 Backend-neutral Parquet scan descriptor

#### Why this change is needed

The logical operation tells the plugin that a Dataset came from
`read_parquet()`, but that is not enough to safely reproduce the read. Ray's
`ParquetDatasource` has already normalized paths, filesystem selection,
projection, schema, fragments, partition columns, predicates, callbacks, and
read options. Reconstructing those facts from the public call or reading private
fields from the plugin would be version-fragile and could silently change
semantics.

Ray should make the semantic decision: either provide a neutral description of
an ordinary scan or provide a stable reason why an external implementation must
not replace it. The descriptor performs no additional I/O and contains no
GPU-specific type.

#### Where it lives

- `python/ray/data/_internal/datasource/parquet_datasource.py`
- Full diff: [`0002-ray-data-expose-backend-neutral-parquet-scan-descriptor.patch`](../ray-hooks/0002-ray-data-expose-backend-neutral-parquet-scan-descriptor.patch)

#### Ray code

The result is explicitly either a descriptor or a refusal:

```python
# python/ray/data/_internal/datasource/parquet_datasource.py

@dataclass(frozen=True)
class ParquetExternalScanDescriptor:
    """Backend-neutral facts for reopening an ordinary Parquet scan.

    The descriptor is produced and consumed on the planning process. In
    particular, ``filesystem`` should not be embedded in actor task payloads.
    """

    filesystem: Any
    source_kind: Literal["local", "s3"]
    paths: Tuple[str, ...]
    listed_file_sizes: Tuple[int, ...]
    projection: Tuple[str, ...]
    file_schema: "pyarrow.Schema"
    region: Optional[str]


@dataclass(frozen=True)
class ParquetExternalScanResult:
    """An external scan descriptor or a stable reason it is unavailable."""

    descriptor: Optional[ParquetExternalScanDescriptor] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.descriptor is None) == (self.reason is None):
            raise ValueError("Exactly one of descriptor or reason must be set")
```

Ray records at construction time whether the original API options are safe to
reproduce externally:

```python
external_scan_options_compatible = (
    _external_scan_paths_use_ambient_access(paths, filesystem)
    and not dataset_kwargs
    and not to_batch_kwargs
    and _block_udf is None
    and schema is None
    and meta_provider is None
    and partition_filter is None
    and shuffle is None
    and not include_paths
    and not include_row_hash
)
```

The descriptor method then validates the normalized datasource state. The
excerpt below shows the capability refusals and the successful result; the full
patch contains projection, fragment, and file-size validation.

```python
def get_external_scan_descriptor(self) -> ParquetExternalScanResult:
    """Describe an ordinary Parquet scan for an external execution backend.

    This method performs no I/O and does not inspect process environment. It
    returns a stable rejection reason whenever reproducing the scan outside
    the stock PyArrow reader could change user-visible semantics.
    """

    import pyarrow.fs as pafs

    if not self._external_scan_options_compatible:
        return ParquetExternalScanResult(reason="read_options")
    if self._predicate_expr is not None:
        return ParquetExternalScanResult(reason="predicate")
    if self._partition_columns:
        return ParquetExternalScanResult(reason="partition_columns")

    # Exact projection and fragment validation occurs here.

    filesystem = self._filesystem
    if type(filesystem) is pafs.LocalFileSystem:
        source_kind: Literal["local", "s3"] = "local"
        region = None
    elif type(filesystem) is pafs.S3FileSystem:
        source_kind = "s3"
        region = filesystem.region
        if not isinstance(region, str) or not region:
            return ParquetExternalScanResult(reason="s3_region")
    else:
        return ParquetExternalScanResult(reason="filesystem")

    return ParquetExternalScanResult(
        descriptor=ParquetExternalScanDescriptor(
            filesystem=filesystem,
            source_kind=source_kind,
            paths=paths,
            listed_file_sizes=tuple(int(size) for size in listed_sizes),
            projection=projection,
            file_schema=self._file_schema,
            region=region,
        )
    )
```

The filesystem remains planning-process state. The plugin serializes immutable
row-group work descriptors for actor tasks rather than embedding the filesystem
object in every task.

#### How the plugin uses it

The plugin consumes only the neutral method and copies the result into its
versioned scan model:

```python
# ray_data_gpu_fusion/parquet.py

result = datasource.get_external_scan_descriptor()
descriptor = getattr(result, "descriptor", None)
if descriptor is None:
    return _reject(f"external_scan:{getattr(result, 'reason', 'declined')}")

normalized = ScanDescriptor(
    filesystem=descriptor.filesystem,
    source_kind=descriptor.source_kind,
    paths=descriptor.paths,
    listed_file_sizes=descriptor.listed_file_sizes,
    projection=descriptor.projection,
    file_schema=descriptor.file_schema,
    region=descriptor.region,
)
```

The plugin then reads footers, verifies source identity, plans exact row-group
work, and selects cuDF/KvikIO runtime behavior. None of that GPU implementation
is added to Ray.

### 5.3 Generic resource admission

#### Why this change is needed

Several GPU resource owners can become runnable in one physical DAG. If each
requests its configured minimum independently, later pools or fixed gangs can
hold scarce GPUs while an earlier operator cannot acquire enough capacity to
make progress. A plugin cannot safely coordinate this from inside its UDF
because worker activation, allocations, scaling, and idle release belong to
Ray.

C adds a generic physical-operator admission contract to Ray's resource
manager. Static GPU actor pools use an elastic-pool adapter, while GPU shuffle
and hash aggregate use a fixed-gang adapter. Unsupported operators retain the
stock lifecycle with a warning that deadlock protection does not apply.

#### Where it lives

- Ray resource-manager, actor-pool, and GPU-shuffle internals changed by C
- Full diff: [`0001-ray-data-generic-resource-admission.patch`](../ray-pr-candidate/0001-ray-data-generic-resource-admission.patch)

#### Ray code

The candidate exposes a narrow internal capability contract:

```python
RESOURCE_ADMISSION_CONTROL_VERSION = 1

# Internal, environment-backed rollback field on DataContext; defaults true.
_enable_resource_admission_control: bool
```

Each participating operator reports elastic-pool or fixed-gang bundle
requirements through `PhysicalOperator.resource_admission_spec()`. Claimants
are scanned in topological order:

1. every floor that fits is admitted and receives an allocator allocation;
2. an elastic pool starts on demand and may scale only within its grant;
3. the first floor that does not fit becomes the frontier; if it fits explicit
   limits but is larger than current cluster capacity, it receives only its
   minimum acquisition to drive autoscaling, while a frontier waiting on an
   earlier owner receives zero units; and
4. later claimants are blocked, preventing them from leapfrogging the frontier.

Completed, dormant, and blocked elastic pools cancel pending actors and release
idle actors, but never active work. Fixed gangs activate atomically and retain
their grant after work starts until extraction completes. This lets a frontier
request acquire capacity as soon as it is available. When enough GPUs exist,
multiple admitted stages keep streaming concurrently.

#### How the plugin uses it

The plugin creates ordinary GPU actor pools with statically declared per-actor
resources. Fixed and autoscaling pool sizes are both eligible. The plugin
checks capability version 1 and requires every region it creates to report an
`ELASTIC_POOL` specification, but does not implement a second admission policy.
Ray selects all participants across the complete physical DAG, including stock
and plugin-created pools.

### 5.4 Why these changes belong in Ray

The extension boundary is intentionally narrow:

- optimizer invocation must be in Ray because Ray owns the physical plan;
- scan description must be in Ray because Ray owns datasource semantics; and
- resource admission must be in Ray because Ray owns resource allocation,
  worker pools, gangs, and autoscaling.

Everything that answers a GPU-specific question remains in the plugin:

- whether an operation is GPU eligible;
- how to read a row group with cuDF or KvikIO;
- how to represent and compose native GPU transforms;
- whether two GPU operations can fuse; and
- what actor-local code executes.

This split also makes the Ray changes useful to external backends other than
this initial cuDF implementation.

## 6. Plugin architecture

### 6.1 Enablement and compatibility

`rgf.enable()` verifies the installed Ray adapter, appends the two rule classes
to the selected `DataContext`, and records whether fusion is enabled. It must be
called before constructing the Dataset.

All imports of unstable Ray internals are centralized in one compatibility
module. The Phase-0 package fails closed on an untested Ray commit or a missing
extension seam. A production plugin should ship one tested adapter per
supported Ray release and feature-gate independent capabilities.

### 6.2 Planning stages

The plugin follows a four-stage planning model:

```text
Recognition -> Closed lowering -> Region selection -> Fusion/materialization
```

These decisions are distinct:

- **Eligibility:** Can this exact logical and physical shape execute correctly
  on the GPU backend?
- **Selection:** Should the eligible implementation be used? Phase 0 selects
  every eligible implementation and does not use a cost model.
- **Fusion:** Can adjacent selected implementations execute in one actor-local
  region without changing resources or semantics?
- **Materialization:** Can a registered backend create a complete Ray physical
  operator for the standalone or fused specification?

Any planning refusal preserves the original stock physical node.

### 6.3 Closed operator model

Each candidate carries an immutable `OperatorSpec` containing:

- ordered `TransformSpec` values;
- input and output `PayloadKind` values;
- execution requirements and actor-pool profile;
- retry policy;
- Ray remote resources;
- runtime environment and placement constraints; and
- required and provided data properties.

Phase 0 uses three active payloads:

| Payload | Meaning |
| --- | --- |
| `ray_block_stream` | Arrow-backed blocks crossing a Ray physical edge |
| `parquet_work` | Immutable exact-row-group read work |
| `frame_stream` | Actor-local cuDF frames inside a GPU region |

The backend factory closes a native specification by adding the appropriate
Ray-block ingress and Arrow egress transforms.

### 6.4 Standalone MapBatches

An eligible `MapBatches` must currently be:

- the exact Ray `MapBatches` logical operation;
- a synchronous function or callable class;
- configured with `batch_format="cudf"`;
- configured with an explicit positive integer `batch_size`;
- backed by a finite actor pool;
- allocated exactly one GPU and one in-flight task per actor; and
- free of dynamic remote arguments or per-block limits.

The standalone physical candidate deliberately reuses the `MapTransformer`
already planned by Ray. This preserves Ray's batch generation, user-code error
behavior, callable-class initialization, and Arrow output shaping.

The candidate also carries a native `map_batches` transform. If adjacent GPU
candidates fuse, the fused actor uses the plugin runtime instead of repeatedly
crossing Ray block boundaries.

### 6.5 Standalone Parquet read

An eligible Parquet read is lowered into:

```text
InputDataBuffer of immutable work descriptors
  -> Ray ActorPoolMapOperator
       -> import one Parquet work descriptor
       -> validate source identity
       -> read exact row groups into cuDF
       -> export Arrow blocks
```

Planning reads Parquet footers, checks stable file identity before and after
footer access, and deterministically balances exact row groups by uncompressed
and compressed size. Runtime revalidates source identity before performing GPU
I/O.

The reader supports ordinary local/shared files and ambient-credential S3 in
the current model. S3 additionally requires `botocore` and the RAPIDS Python
`kvikio` package on every GPU worker.

## 7. Fusion

Fusion examines linear regions of independently executable GPU candidates. It
does not fuse through branches, multiple inputs, or a non-GPU physical node.

Two candidates compose only when all of the following are compatible:

- output and input payload kinds;
- required and provided data properties;
- actor-pool minimum, initial, and maximum sizes;
- CPU, GPU, memory, and custom resources;
- maximum concurrency and tasks in flight;
- placement and scheduling strategy;
- runtime environment;
- retry semantics; and
- backend identity.

A fusion refusal leaves the original standalone GPU candidates in the plan.

For a compatible chain:

```text
read_parquet -> map_batches A -> map_batches B
```

Ray sees one physical actor-pool operator whose actor-local program is:

```text
Ray work-descriptor block
  -> Parquet work import
  -> exact-row-group cuDF read
  -> cuDF batch UDF A
  -> cuDF batch UDF B
  -> Arrow export
```

No intermediate Ray ObjectRef is created inside this region.

## 8. Execution examples

### 8.1 Stock CPU read followed by one GPU map

If the read is unsupported by the GPU adapter but the map is eligible:

```text
stock CPU Parquet read
  -> Arrow ObjectRef
  -> plugin-marked Ray GPU ActorPoolMapOperator
       -> Ray slices input into batch_size row batches
       -> Arrow-to-cuDF inside the GPU actor
       -> user UDF
       -> cuDF-to-Arrow
  -> Arrow ObjectRef
```

This path mostly reuses behavior Ray already supports. With one input block,
many map batches still execute within one actor task; `batch_size` controls UDF
invocation size, not distributed actor parallelism.

### 8.2 CPU read followed by two compatible GPU maps

The CPU read remains a stock boundary, while the maps fuse with each other:

```text
CPU read
  -> Arrow ObjectRef
  -> one fused GPU actor region
       -> Arrow-to-cuDF once
       -> map A
       -> map B
       -> cuDF-to-Arrow once
```

### 8.3 GPU read followed by a CPU operator and a GPU map

Phase 0 permits the transfer-heavy but modular plan:

```text
GPU Parquet read
  -> cuDF-to-Arrow
  -> CPU operator
  -> Arrow ObjectRef
  -> Arrow-to-cuDF
  -> GPU MapBatches
  -> cuDF-to-Arrow
```

The CPU operator is a fusion barrier. The GPU read and GPU map are separate
actor regions governed by Ray's generic admission policy. If both one-actor
floors fit, they may stream concurrently. Otherwise the downstream region waits
at or behind the frontier without leapfrogging the earlier claimant.

### 8.4 Two incompatible GPU regions

When resources, payloads, runtime environments, or properties do not compose:

```text
GPU region A -> Arrow ObjectRefs -> GPU region B
```

Region A drains and releases pending and idle actors. Region B then requests
actors and begins when Ray makes a GPU available. While B waits, its Arrow input
may accumulate in the Ray Object Store. This whole-region handoff is a Phase-0
memory tradeoff.

## 9. Read parallelism and SplitBlocks

Ray independently chooses the number of read tasks and the desired number of
output blocks. Let:

- `P` be detected or requested output parallelism;
- `T` be the number of read tasks the datasource can construct; and
- `S` be any size-based split factor already required.

When `T * S < P`, Ray computes approximately:

```text
k = ceil(P / (T * S))
```

and appends `SplitBlocks(k)` to the read. The split runs after reading and
slices each produced block into row ranges. It does not reread the file `k`
times.

For a single-file read, Ray may produce:

```text
ReadParquet -> SplitBlocks(192)
```

One read task performs I/O and decoding, then emits many smaller Arrow blocks
that downstream tasks can process in parallel.

### 9.1 Current Phase-0 behavior

The implemented plugin declines direct GPU-read replacement when the stock read
has an additional split factor. Replacing the complete read operator without
reproducing its split would change the number of output blocks and downstream
task boundaries.

This is not a Parquet or cuDF incompatibility. It is an unimplemented Ray block
partitioning contract.

### 9.2 Proposed correctness-first support

Because Phase 0 accepts Arrow materialization costs, the initial support for
`SplitBlocks(k)` should treat it as a hard external boundary:

```text
GPU Parquet read
  -> Arrow export
  -> SplitBlocks(k)
  -> Arrow ObjectRefs
  -> optional downstream GPU region
```

The split may be implemented at GPU-read egress or as a small Ray physical
operator. A downstream GPU map imports the resulting Arrow blocks into cuDF.
Read-to-map fusion does not cross this boundary.

A later GPU-native split transform may slice actor-local cuDF frames and allow
fusion to continue, but that optimization is not required for correctness.

## 10. Resource admission and lifecycle

All plugin physical regions use Ray actor pools. The plugin translates an
execution profile into ordinary Ray remote arguments, including CPUs, one GPU
per actor, memory, custom resources, placement, runtime environment, actor
restart policy, and task retry policy.

Each plugin region reports an elastic admission specification with a one-actor
floor. Ray bounds scaling by its admission grant. Topological
admitted/frontier/blocked states prevent later pools or gangs from capturing
capacity needed by the frontier. Pools that become dormant, complete, or
blocked cancel pending requests and release idle actors; active tasks continue
normally. Independent stages whose floors fit may run and stream concurrently,
while Ray Core retains placement authority.

## 11. Boundaries and memory behavior

Phase 0 uses Arrow-backed Ray blocks for every external region edge.

Benefits:

- ordinary Ray block ownership and reconstruction semantics;
- CPU operators require no special GPU awareness;
- standalone regions are independently testable;
- retries and failures stay within Ray's existing task model; and
- the plugin does not introduce a second lifetime manager.

Costs:

- device-to-host and host-to-device conversion between regions;
- possible buffering of a complete upstream region in the Object Store while a
  downstream GPU region waits;
- eager per-task Arrow materialization before the first generator yield; and
- no device-resident sharing between separate actors.

These costs are accepted for Phase 0. A future device payload must define
ownership, reconstruction, locality, spilling, failure, and scheduler semantics
before it can replace the Arrow boundary.

## 12. Fallback and failure semantics

Planning fallback and runtime failure are deliberately different.

### 12.1 Planning

- Recognition refusal preserves the exact stock Ray physical node.
- Unsupported options produce a stable refusal reason.
- Backend materialization refusal preserves the stock node.
- Fusion refusal preserves independently executable GPU nodes.
- Unknown Ray versions or missing seams fail before mutating the context.

### 12.2 Execution

After GPU execution starts, I/O, CUDA, actor, and user-code failures follow
Ray's normal retry and error propagation paths. The plugin does not replay work
on a CPU implementation after a partial GPU execution.

`rgf.explain(dataset)` constructs the optimized physical plan Ray would execute
and reports standalone regions, fused regions, and relevant stock-node refusal
reasons.

## 13. Upstream and deployment model

The prototype repository contains:

- a clean pinned Ray submodule;
- one generic PR-candidate patch C;
- exactly two local backend-neutral hooks H1/H2;
- independently pinned stock, PR-candidate, and hooked Ray wheels;
- the external plugin distribution; and
- a pinned RAPIDS environment.

This layout proves the extension contract without making the dirty GPU fork the
runtime dependency.

If Ray accepts C and the two hooks, normal deployment becomes:

```text
official Ray wheel with extension APIs
+ RAPIDS runtime on GPU workers
+ ray-data-gpu-fusion on driver and workers
```

The prototype build preserves `stock`, `stock+C`, and `stock+C+H1+H2` as
separate artifacts. Bootstrap installs only the last layer and proves it equals
a direct stock+C+H1+H2 build. Once those capabilities ship in an official Ray
wheel, the derived wheels and patch application step leave the installation
path and may remain only as provenance.

An official Ray release will have a different version and commit. The plugin
must publish a corresponding tested compatibility adapter; the current
prototype intentionally does not accept an arbitrary later Ray commit merely
because similarly named methods are present.

The three capabilities are independently identifiable:

- generic resource admission is candidate C and has an
  internal rollback field;
- physical-rule injection is required for any optimizer plugin;
- the external scan descriptor is required only for direct Parquet GPU reads.

## 14. Extending to additional Ray Data APIs

A new operation adapter should satisfy four requirements before participating
in fusion:

1. recognize one exact logical and physical shape and decline unsupported
   options;
2. emit a native transform with explicit payload and property contracts;
3. register an actor-local runtime for the transform; and
4. provide a closed standalone materialization path.

Scalar expressions, encoders, projections, and preprocessors can reuse the
existing frame-stream payload and composition engine. If implemented on GPU,
they can remove CPU boundaries from plans such as:

```text
GPU read -> expression/preprocessor -> GPU MapBatches
```

Operations involving exchanges, ordering, multiple inputs, or partitioning
must add explicit payload and property contracts rather than special cases to
the existing MapBatches adapter.

## 15. Phase-0 scope

Implemented fast paths:

- ordinary local/shared-filesystem Parquet scans;
- ambient-credential S3 recognition and runtime substrate;
- exact synchronous cuDF `MapBatches` actor shapes;
- standalone closed GPU reads and maps;
- compatible linear fusion;
- Arrow boundaries between incompatible regions; and
- resource-aware admission across eligible GPU pools.

Intentionally deferred:

- `SplitBlocks(k)` preservation for direct GPU reads;
- DataSource V2 Parquet recognition;
- scalar-expression, encoder, and preprocessor adapters;
- grouped partitions, shuffles, sort, aggregate, join, and zip;
- device-resident inter-region exchange; and
- cost-based backend selection.

The exact implemented eligibility matrix is maintained separately in
[`phase-0.md`](phase-0.md).

## 16. Validation strategy

The implementation is validated at four levels:

1. **Ray extension contracts:** plan-local rule isolation, datasource descriptor
   behavior, generic actor admission, fairness, and scarce-resource handoff.
2. **Plugin unit contracts:** eligibility, composition, empty-batch parity,
   generator behavior, lowering, fusion, and refusal diagnostics.
3. **Ray integration:** unsupported operations execute on the unchanged stock
   path and standalone GPU maps materialize Arrow boundaries.
4. **GPU correctness smoke:** stock and genuinely fused
   `read_parquet -> map_batches` plans produce identical result digests.

Unknown versions and unsupported shapes remain stock or fail closed rather than
silently changing semantics.

## 17. Alternatives considered

### 17.1 Keep the complete GPU backend inside Ray

This gives direct access to internals but couples Ray to cuDF-specific planning,
runtime state, release cadence, and API surface. It also makes supporting a
second external backend harder. Rejected in favor of generic Ray seams.

### 17.2 Wrap Dataset construction and actions

A wrapper could inspect a Dataset and run its own graph, but it would no longer
be Ray's physical plan and would need to duplicate execution semantics. Rejected.

### 17.3 Monkey-patch the optimizer

Process-global monkey-patching is difficult to isolate across Datasets and
planning processes and is fragile across Ray versions. Rejected.

### 17.4 Fuse first and implement only fused regions

This makes fusion necessary for correctness and leaves no safe outcome when a
compatibility check fails. Rejected; standalone closure precedes fusion.

### 17.5 Introduce a device-resident Ray block immediately

This may eventually remove transfers, but it requires a complete distributed
ownership, locality, spilling, and reconstruction design. Deferred until the
external plugin and closed-region model are established.

## 18. Open design items

- Define the precise physical representation of an Arrow
  `SplitBlocks(k)` boundary after a GPU read.
- Separate automatic Ray read parallelism from an explicit
  `override_num_blocks` contract when deciding whether exact block count must be
  reproduced.
- Decide which Ray extension APIs should become public and versioned rather
  than compatibility-adapted internals.
- Define packaging and worker-image requirements for credentialed S3 execution.
- Establish memory benchmarks for whole-region Arrow buffering and eager task
  egress.
- Define the ownership contract required for a future device-resident payload.
