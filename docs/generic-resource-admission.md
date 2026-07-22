# Generic Resource Admission for GPU Actors and Shuffle Gangs

Status: implemented in PR-candidate commit
`5b9ac4f86ca7bdf3d0b5f3d4d6b657f2e9e1fd5e`, based on stock Ray commit
`2741c6461d2bd3e5ff114af67be7a1190453dadd`.

This document describes the Ray Data PR candidate only. Plugin compatibility
changes, local planning hooks, packaging, and generated wheels are outside the
PR's code footprint.

## Summary

Ray Data needs to coordinate scarce GPU resources across a complete physical
pipeline. Today, a resource-owning physical operator can begin creating actors
or submitting work as execution starts, before the Ray Data resource manager
has made a pipeline-level decision about which operator must run first. When
several streaming stages need more GPUs than the pipeline can use at once,
output backpressure and partial resource acquisition can form a deadlock.

This PR adds an internal, versioned resource-admission contract to
`PhysicalOperator`. Operators describe their minimum progress floor as complete
resource bundles. A topological controller admits the maximal prefix of
operators whose floors fit, prevents later operators from leapfrogging the
first non-fitting frontier, and grants excess resources only after every
admitted floor is protected.

The production adapters in this PR cover:

- actor-based GPU `map_batches`;
- actor-based `map_groups(batch_format="cudf")`, through the same actor-pool
  operator;
- the complete GPU shuffle rank group; and
- `GPUHashAggregateOperator`, through its GPU shuffle base class.

The contract remains extensible to a future task-specific admission kind, but
this PR does not define or wire production GPU task admission. GPU tasks and
unknown GPU operators retain legacy scheduling and emit a once-per-execution
warning that deadlock protection does not apply.

No public Dataset API or Ray Core change is required.

## Problem statement

Ray Data pipelines can contain several physical operators that acquire GPUs
through long-lived actor pools, coordinated rank actors, or tasks.
The streaming executor knows the dependency graph between these operators, but
worker creation has historically not been gated by a pipeline-level admission
decision.

As a result:

- an operator can begin asynchronous actor or placement-group creation before
  the resource manager has decided that it should own GPUs;
- an actor pool can retain more actors than it needs to make progress;
- multiple operators can each hold or request only part of the capacity needed
  by another operator; and
- output backpressure can prevent the current GPU owner from finishing and
  releasing its resources.

### Deadlock example

Consider this physical pipeline:

```text
Input -> GPU Pool A -> GPU Pool B -> Output
```

Its resource configuration is:

```text
Cluster capacity: 1 GPU
Pool A actor requirement: 1 GPU
Pool B actor requirement: 1 GPU
```

Without pipeline-level admission, the following cycle is possible:

1. Pool A owns the only GPU.
2. Pool B begins requesting its actor but cannot start.
3. Pool A's output queue reaches its backpressure limit.
4. Pool A cannot finish and release its actor until Pool B consumes more
   output.
5. Pool B cannot consume output because it is waiting for the GPU held by Pool
   A.

The same issue is more severe for a coordinated GPU shuffle. A four-rank
shuffle cannot make progress with only two ranks, even if a proportional
allocator assigns it two GPUs.

### Why Ray Core cannot solve the cycle

Ray Core correctly schedules individual actors, tasks, and placement groups.
It does not know:

- which resource requests belong to one Ray Data physical DAG;
- which operator is an unfinished ancestor of another operator;
- which streaming edge is blocked by output backpressure;
- which complete group of workers is required for operator progress; or
- when an otherwise healthy actor should be released so another physical stage
  can run.

Ray Core should continue to perform exact placement, node-label enforcement,
placement-group scheduling, and cluster autoscaling. Ray Data must decide which
physical operators are allowed to create and retain those resource owners.

### Why the existing Ray Data allocator is insufficient

The existing allocator primarily distributes resource budgets
proportionally. That is useful for throughput, but it does not provide a
liveness guarantee for indivisible resource requests:

- a one-GPU actor cannot run with a 0.5-GPU allocation if its declared bundle
  requires one GPU;
- a fixed-rank shuffle cannot run until every rank bundle is available;
- current resource usage must not let an operator justify retaining an
  over-allocation; and
- allocator budgeting after eager worker startup is too late to control initial
  acquisition.

The policy must first reserve complete minimum progress floors, then share any
remaining capacity for performance.

## Goals and non-goals

### Goals

- Make the Ray Data physical DAG the authority for starting and retaining
  scarce resource owners.
- Express progress in complete usable units: one actor, one task, or
  one complete fixed rank group.
- Protect every admitted operator's minimum progress floor before allocating
  excess resources.
- Serialize GPU stages when their floors do not fit concurrently, while still
  allowing overlap when they do.
- Let upstream Arrow output queue or spill so an operator can finish and
  release its GPUs before a downstream stage starts.
- Support fractional CPU/GPU declarations, custom resources, label selectors,
  explicit execution limits, and cluster autoscaling.
- Provide an internal extension point for future GPU tasks, sorts, joins, and
  other physical operators without adding controller type checks.

### Non-goals

- Replacing Ray Core scheduling or placement-group semantics.
- Adding a public Dataset resource-admission API.
- Wiring production `TaskPoolMapOperator` admission in this PR.
- Fusing device memory across physical operators or removing Arrow boundaries.
- Guaranteeing deadlock protection for GPU operators that cannot declare a
  static resource envelope.

## Proposed design

Every supported GPU owner declares a versioned resource-admission
specification. The executor considers complete progress floors in deterministic
topological order before it shares excess capacity.

An elastic actor pool's progress floor is one complete actor, even when its
configured minimum pool size is larger. A fixed shuffle group's floor is the
complete set of rank bundles. Ray scheduling calls such an atomically admitted
set a gang, which is why the contract names this case `FIXED_GANG`. The first
floor that does not fit is the
frontier; all later participating operators are blocked from starting new
resource acquisition. An operator that loses its grant stops submitting new
work, releases pending and idle workers, and lets finite active work drain.

This produces the intended one-GPU handoff:

```text
admit Pool A with one actor
-> keep Pool B at a zero-unit frontier grant
-> finish Pool A while queuing or spilling Arrow output
-> release Pool A's actor
-> admit Pool B
-> consume the queued Arrow output
```

For the full supported chain, the same rule becomes:

```text
finish upstream map_batches
-> queue or spill Arrow blocks
-> release its actor
-> activate the complete shuffle rank group
-> extract queueable or spillable Arrow blocks
-> remove the placement group
-> activate map_groups
-> release its actor
-> activate the final map_batches
```

The output-backpressure escape path lets the active upstream generator keep
queuing Arrow blocks when its downstream admission boundary cannot yet run.
This lets the upstream call finish and makes its actor releasable.

The policy is not a global GPU lock. When all demanded minimum floors fit,
multiple stages are admitted and may overlap. A default GPU shuffle uses one
rank per detected cluster GPU, so it normally serializes with surrounding GPU
actors unless `gpu_shuffle_num_actors` is configured below total capacity.

This belongs in Ray Data because the streaming executor owns the physical
topology, queues, backpressure state, resource accounting, and actor
autoscaling. No Ray Core change is required.

## Internal contract

The contract is internal and explicitly versioned:

```python
RESOURCE_ADMISSION_CONTROL_VERSION = 1


class AdmissionKind(Enum):
    ELASTIC_POOL = "elastic_pool"
    FIXED_GANG = "fixed_gang"


@dataclass(frozen=True)
class AdmissionBundle:
    resources: ExecutionResources
    custom_resources: tuple[tuple[str, float], ...] = ()
    label_selector: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ResourceAdmissionSpec:
    kind: AdmissionKind
    minimum_bundles: tuple[AdmissionBundle, ...]
    unit_bundle: Optional[AdmissionBundle]
    max_units: Optional[int]
    sticky_after_start: bool


@dataclass(frozen=True)
class ResourceAdmissionGrant:
    max_units: int
```

`PhysicalOperator` supplies four lifecycle hooks:

- `resource_admission_spec()`;
- `has_internal_admission_demand()`;
- `apply_resource_admission_grant()`; and
- `can_release_resource_admission()`.

It also supplies `_resource_admission_fallback_reason()` for operators whose
resources cannot be described statically, so the validator can explain the
legacy fallback without depending on an operator type.

The controller depends only on these hooks. It has no operator-type branches
for map, shuffle, aggregate, or `map_groups`.

An operator that statically reports GPU usage but does not implement the
contract receives one warning per execution. It keeps legacy scheduling, and
the warning states that resource-deadlock protection does not apply. This is
the current behavior for GPU `TaskPoolMapOperator` and unknown future GPU
operators.

`DataContext._enable_resource_admission_control` defaults to true and is backed
by `RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL`. Setting it to false makes the
current actor and shuffle adapters use legacy acquisition and therefore removes
the admission deadlock guarantee.

## Controller and executor lifecycle

### Two-phase startup

Topology startup is split into two phases:

1. construct every `OpState` and wire all queues without starting operators;
2. construct `ResourceManager`, validate every specification, and apply an
   initial zero-unit grant;
3. call each operator's `start()` for local initialization; and
4. acquire workers only after the first admission update.

This prevents topology construction from eagerly consuming GPUs before the
controller has seen the whole DAG.

### Participation

An unfinished declared operator participates when it has external queued
input, internal input, active work, pending resources, or owned resources.
Every still-relevant declared ancestor of a direct participant is added, even
across non-participating CPU operators.

Admission completion is intentionally narrower than
`has_execution_finished()`. An upstream operator can have buffered Arrow output
and remain execution-unfinished while no longer needing its GPU. It stops
participating when its inputs are complete, its input queues and active work
are empty, and its adapter says its resources are releasable. This condition is
what permits the downstream handoff.

### States and grants

The generic states are:

| State | Meaning | New submissions | Resource behavior |
| --- | --- | --- | --- |
| `DORMANT` | No current admission demand | Disabled | Release pending and idle resources |
| `ADMITTED` | Complete minimum fits | Enabled within grant | Own the floor and optionally scale |
| `FRONTIER` | First non-fitting minimum | Disabled | Drain active work; seed one minimum acquisition only for cluster autoscaling |
| `BLOCKED` | Participant after frontier | Disabled | Drain active work and release inactive resources |

The scan reserves complete minimum bundles before normal proportional
allocation. Elastic maximums are derived from a usage-independent allocation
target, so a pool cannot justify excess actors merely by already owning them.

Admission floors remain enabled when
`op_resource_reservation_enabled=False`. In that mode, proportional sharing is
absent and admitted elastic pools receive their minimum unit count.

If any floor fits explicit execution limits but is larger than current cluster
capacity, the frontier receives its minimum unit count solely to create a
pending resource request. This gives cluster autoscaling a concrete request;
for a fixed gang it creates the atomic placement group and starts the setup
timeout. If the floor fits current cluster capacity but cannot fit beside an
earlier admitted owner, it receives zero until that owner hands off the
resource. Any floor that exceeds explicit execution limits fails before
operator startup with its required bundles and permitted limits.

After a sticky gang accepts its first input, its unfinished declared producer
closure is protected with it. If reported cluster capacity shrinks, those
progress floors may temporarily exceed the refreshed limit: demoting only the
producer would strand the gang waiting for input that can no longer be made.
The proportional allocator therefore tolerates this temporary over-capacity
floor while Ray Core restores capacity, fails the owner, or the chain drains.

## Actor-pool adapter

`ActorPoolMapOperator` reports `ELASTIC_POOL` when:

- generic admission is enabled;
- the statically declared per-actor GPU requirement is positive; and
- the user did not supply a dynamic `ray_remote_args_fn` whose resource
  envelope cannot be trusted.

The one-actor `AdmissionBundle` preserves fractional CPU/GPU/memory values,
custom resources, and the merged Dataset/operator label selector. The unit
bundle is identical to the floor bundle. `max_units` is the configured pool
maximum, or unbounded when the pool maximum is infinite.

The adapter behavior is:

- do not start the initial pool until the grant is positive;
- never grow beyond `grant.max_units`;
- use one actor as the progress floor even for a larger fixed/minimum pool;
- after demotion, cancel pending actors and kill idle actors;
- allow active actor calls to finish, but do not dispatch another internally
  queued bundle to an excess or demoted actor; and
- implement `wait_for_min_actors_s` as asynchronous readiness after admission,
  capped by the granted floor instead of blocking topology construction.

This one adapter covers stock actor `map_batches`, plugin-created actor regions,
and actor cuDF `map_groups`, because `map_groups` lowers to an ordinary
`ActorPoolMapOperator` wrapper. There is no groupby-specific admission code.

## Shuffle fixed-gang adapter

`GPUShuffleOperator` reports `FIXED_GANG` with one normalized
`CPU=1, GPU=1` bundle per rank, `max_units=1`, and
`sticky_after_start=True`. `GPUHashAggregateOperator` inherits this adapter.

`GPURankPool` uses this asynchronous lifecycle:

```text
INACTIVE -> RESERVING -> STARTING -> READY -> CLOSED
                         \-> FAILED
```

Activation creates one atomic Ray placement group with `SPREAD`, repeats the
configured label selector for every bundle, and pins rank `i` to bundle `i`.
The operator polls placement-group readiness, root UCXX setup, and worker setup
without blocking topology construction. Input is rejected until every rank is
ready. Placement-group and UCXX setup share the existing shuffle setup timeout.

Before the first insert, a revoked grant removes an idle or pending placement
group. The first insert makes the gang sticky. It remains owned until all
extraction generators finish, at which point every actor is killed and the
placement group is removed immediately, even if extracted Arrow output is
still queued downstream. Timeout, cancellation, or partial failure performs
the same whole-gang cleanup.

The configured `gpu_shuffle_num_actors` is preserved. When unset, rank count
continues to default to all GPUs detected when the operator is planned.

## Arrow boundary

Admission changes resource ownership, not the Ray Data block contract:

```text
actor map_batches: cuDF inside worker -> Arrow Ray ObjectRefs
GPU shuffle: Arrow -> cuDF/rapidsmpf -> Arrow Ray ObjectRefs
actor map_groups: per-group Arrow -> cuDF -> Arrow Ray ObjectRefs
```

Those host-side Arrow boundaries are intentional. They can be queued and
spilled while the next GPU owner waits, which is what makes a serialized
one-GPU handoff possible. There is no direct device-buffer fusion across the
shuffle or groupby boundary in this design.

## Extension model

The controller is intended to remain unchanged as coverage expands:

- future APIs that lower to `ActorPoolMapOperator` inherit elastic admission;
- a future GPU sort, join, or communicator group supplies a fixed-gang spec;
- a future `TaskPoolMapOperator` adapter can add a task-specific kind, supply
  one task bundle as its floor, and gate submissions with `max_units`; and
- a new Ray API can implement the four `PhysicalOperator` hooks without
  depending on this plugin.

Production GPU tasks intentionally remain warning-and-legacy in this release;
the internal enum will grow when task admission has production semantics.

## Validation

Implemented coverage includes:

- elastic and fixed-gang contract tests;
- topology chains, fan-in, fractional GPUs, complete gang floors, frontier
  behavior, sticky gangs, and drained-ancestor handoff;
- zero-grant startup, fixed/autoscaling actor pools, asynchronous minimum-actor
  waiting, active-call drain, and pending/idle cleanup;
- atomic placement, exact bundle indices, repeated labels, readiness, revoke,
  timeout, partial failure, cancellation, and placement-group cleanup;
- inherited `GPUHashAggregateOperator` behavior;
- warning-and-fallback behavior for GPU tasks and unknown operators; and
- plugin compatibility against `RESOURCE_ADMISSION_CONTROL_VERSION`.

A hardware-marked end-to-end test plans and runs:

```text
actor cuDF map_batches
-> one-rank GPU shuffle
-> actor map_groups(batch_format="cudf")
-> actor cuDF map_batches
```

It constrains Ray Data to one logical GPU, asserts three distinct actor pools
plus one gang, validates the grouped result, and waits for complete GPU release.
CPU-only shuffle and aggregate tests run with mocked Ray actors. The end-to-end
test requires a working CUDA driver, cuDF, RAPIDS MPF, and UCXX.

## Implementation size

Measured against unmodified stock Ray commit
`2741c6461d2bd3e5ff114af67be7a1190453dadd`, counting added physical Python
source lines and excluding blank and comment-only lines:

| Production area | Added NCLOC | Net NCLOC above stock |
| --- | ---: | ---: |
| Contract, hooks, controller, and allocator integration | 457 | 434 |
| Two-phase executor startup and backpressure gates | 45 | 43 |
| Actor adapter and autoscaler integration | 198 | 167 |
| Shuffle placement-group and asynchronous gang lifecycle | 259 | 142 |
| Context capability flag | 6 | 6 |
| **Total Ray production code** | **965** | **792** |

Tests, documentation, release notes, and plugin code are excluded. The plugin
capability migration separately adds 39 and deletes 42 production NCLOC, for a
net reduction of 3 lines.

The implementation is larger than the original 620-780 added-NCLOC estimate.
That estimate understated the complete allocator integration and the
failure-safe asynchronous placement-group lifecycle. A minimality audit
removed 143 additions and reduced net growth by 188 lines by reusing the stock
actor autoscaler, expressing setup ordering through Ray object dependencies,
and computing allocation targets on demand. The remaining code includes the
generic contract and controller, floor-first allocation, two-phase startup,
actor admission adapter, and atomic shuffle-gang cleanup rather than only the
new interface surface.

## Current limitations

- Production GPU task admission is not wired yet.
- Dynamic per-actor resource callbacks have no trustworthy static envelope and
  therefore retain legacy scheduling.
- Unknown GPU operators warn rather than fail planning, so the deadlock
  guarantee covers declared participants only.
- Admission capacity uses the standard `ExecutionResources` dimensions for
  proportional sharing. If any participating bundle has custom resources or a
  label selector, the controller conservatively admits one owner at a time;
  Ray Core still performs the exact per-node placement. This protects liveness
  without pretending that aggregate Ray Data limits model node shapes, at the
  cost of reduced overlap for constrained bundles.
- The contract is internal and may evolve by increasing
  `RESOURCE_ADMISSION_CONTROL_VERSION`.
