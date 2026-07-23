# Generic Resource Admission for GPU Actors and Shuffle Gangs

Status: implemented by the pinned PR candidate in `pins/pr-candidate.json`, based
on stock Ray commit `2741c6461d2bd3e5ff114af67be7a1190453dadd`.

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

This PR adds a small internal resource-admission contract to
`PhysicalOperator`. Operators describe an aggregate minimum progress floor and,
when elastic, the resource cost of one usable unit. A topological controller
admits the maximal topological prefix of current claimants whose floors fit,
subject to the protected closure of sticky owners. It prevents later claimants
from leapfrogging the first non-fitting frontier and grants excess resources
only after every admitted floor is protected.

The production adapters in this PR cover:

- actor-based GPU `map_batches`;
- actor-based `map_groups(batch_format="cudf")`, through the same actor-pool
  operator;
- the complete GPU shuffle rank group; and
- `GPUHashAggregateOperator`, through its GPU shuffle base class.

The aggregate contract could be implemented by future physical operators, but
this PR does not wire production GPU task admission. GPU tasks and undeclared
GPU operators remain outside admission. When a topology combines a declared
owner with a recognized adapter constraint that cannot be represented, the
controller emits one warning and clears every declared specification to avoid
partially applying the ownership policy. Actor pools then acquire eagerly and
candidate shuffle gangs retain eager atomic placement-group behavior.

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

Ray Core correctly schedules individual actors, tasks, and submitted placement
groups. It cannot infer:

- which resource requests belong to one Ray Data physical DAG;
- which operator is an unfinished ancestor of another operator;
- which streaming edge is blocked by output backpressure;
- that independently submitted workers form one complete progress group unless
  Ray Data encodes them in a placement group; or
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
- Express progress as an aggregate resource floor and, for elastic operators,
  complete usable units.
- Protect every admitted operator's minimum progress floor before allocating
  excess resources.
- Serialize GPU stages when their floors do not fit concurrently, while still
  allowing overlap when they do.
- Let upstream Arrow output queue or spill so an operator can finish and
  release its GPUs before a downstream stage starts.
- Support fractional CPU/GPU declarations, explicit execution limits, and
  cluster autoscaling, while falling back to eager candidate behavior for
  recognized constraints the aggregate model cannot represent.
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

Every supported GPU owner declares a resource-admission specification. The
executor considers complete progress floors in deterministic
topological order before it shares excess capacity.

An elastic actor pool's progress floor is its configured minimum pool size; the
grant can grow toward its configured maximum when capacity is available. A
fixed shuffle group's floor is the complete set of rank bundles acquired by one
atomic placement group. The first floor that does not fit is the frontier; all
later participating operators are blocked from starting new resource
acquisition. An operator that loses its grant stops submitting new work,
releases pending and idle workers, and lets finite active work drain.

This produces the intended one-GPU handoff:

```text
admit Pool A with its configured minimum actor set
-> keep Pool B at a zero-unit frontier grant
-> finish Pool A while queuing or spilling Arrow output
-> release Pool A's actors
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
rank per GPU in Ray's planning capacity (the autoscaler maximum when configured,
otherwise the current cluster), so it normally serializes with surrounding GPU
actors unless `gpu_shuffle_num_actors` is configured below that capacity.

This belongs in Ray Data because the streaming executor owns the physical
topology, queues, backpressure state, resource accounting, and actor
autoscaling. No Ray Core change is required.

## Internal contract

The contract is internal and deliberately small. It adds no public API and no
general bundle/type/version abstraction:

```python
@dataclass(frozen=True)
class ResourceAdmissionSpec:
    minimum_resources: ExecutionResources
    unit_resources: Optional[ExecutionResources]
    min_units: int
    max_units: Optional[int]


@dataclass(frozen=True)
class ResourceAdmissionGrant:
    max_units: int
    may_submit: bool
```

`PhysicalOperator` supplies three lifecycle hooks:

- `resource_admission_spec()`;
- `apply_resource_admission_grant()`; and
- `can_release_resource_admission()`.

The admission lifecycle depends on these hooks. The topology-wide safety check
also reads an adapter-private incompatibility marker; neither path has
operator-type branches for map, shuffle, aggregate, or `map_groups`.

If any otherwise-managed topology contains an adapter whose resource envelope
cannot be represented safely, the controller emits one topology warning,
clears all admission specifications, and retains eager placement-group-only
behavior for the whole topology. Examples detected by the current actor-pool
and shuffle adapters include dynamic actor options, custom resources, label
selectors, and unsupported placement strategies. Undeclared operators are
otherwise unchanged and outside the admission guarantee; the controller does
not guess their semantics.

`DataContext._enable_resource_admission_control` defaults to true and is backed
by `RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL`. Setting it to false makes the
controller ignore the current specifications and therefore removes the
topological admission guarantee. Actor pools return to eager acquisition; the
candidate's shuffle still uses its atomic placement group but activates it
eagerly. This is the placement-group-only ablation used by the evidence
harness, not behavior identical to the unmodified stock wheel.

## Controller and executor lifecycle

### Two-phase startup

The split startup orders work as follows:

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

A declared single-input direct successor of a fixed gang may expose its floor
after the gang's inputs complete while the gang is still draining or
extracting. This prewarms a Core request, not guaranteed resource ownership.
It is deliberately one hop only; later stages do not recursively request GPUs
before they have input.

Admission completion is intentionally narrower than
`has_execution_finished()`. An upstream operator can have buffered Arrow output
and remain execution-unfinished while no longer needing its GPU. It stops
participating when its inputs are complete, its input queues and active work
are empty, and its adapter says its resources are releasable. This condition is
what permits the downstream handoff.

### Grants and frontier scan

There is no parallel admission state machine. Each operator receives only one
grant: `max_units` caps ownership and `may_submit` gates new work. Existing
operator state remains the source of truth for pending, active, and releasable
resources.

The scan reserves complete minimum floors before normal proportional
allocation. Elastic maximums are derived from a usage-independent allocation
target, so a pool cannot justify excess actors merely by already owning them.

Admission floors remain enabled when
`op_resource_reservation_enabled=False`. In that mode, proportional sharing is
absent; the first admitted elastic pool can receive topological spare capacity
above its floor, while later admitted pools retain their protected floors.

If any floor fits explicit execution limits but is larger than current cluster
capacity, the frontier receives its minimum unit count solely to create a
pending resource request. This gives cluster autoscaling a concrete request;
for a fixed gang it creates the atomic placement group. The UCXX setup timeout
starts only after that placement group is ready, so node provisioning time is
not mistaken for communicator failure. If the floor fits current cluster
capacity but cannot fit beside an earlier admitted owner, an unseeded floor
receives zero until that owner hands off the resource. A fixed-gang request
already exposed during cold autoscaling remains pending through this handoff so
newly provisioned nodes do not lose the whole-gang demand. Any floor that
exceeds explicit execution limits fails before operator startup with its
required floor and permitted limits.

After a gang accepts its first input, `can_release_resource_admission()` keeps
it and its unfinished declared producer closure protected. If reported cluster
capacity shrinks, those
progress floors may temporarily exceed the refreshed limit: demoting only the
producer would strand the gang waiting for input that can no longer be made.
The proportional allocator therefore tolerates this temporary over-capacity
floor while Ray Core restores capacity, fails the owner, or the chain drains.

## Actor-pool adapter

`ActorPoolMapOperator` reports an aggregate elastic specification when:

- generic admission is enabled;
- the statically declared per-actor GPU requirement is positive; and
- the actor resources use the supported CPU/GPU/memory dimensions and a
  default or spread scheduling strategy.

`unit_resources` is one actor's CPU/GPU/memory cost. The floor is that cost
times the configured minimum pool size, and `max_units` is the configured pool
maximum (or unbounded for an infinite pool). Dynamic or constrained resources
trigger the whole-topology admission fallback with a warning.

The adapter behavior is:

- do not start the initial pool until the grant is positive;
- never grow beyond `grant.max_units`;
- preserve `ActorPoolStrategy(size=N/min_size=N)` by requiring all `N` actors
  in the progress floor;
- after demotion, cancel pending actors and kill idle actors;
- allow active actor calls to finish, but do not dispatch another internally
  queued bundle to an excess or demoted actor; and
- implement `wait_for_min_actors_s` as asynchronous readiness after admission,
  capped by the granted floor instead of blocking topology construction.

This one adapter covers stock actor `map_batches`, plugin-created actor regions,
and actor cuDF `map_groups`, because `map_groups` lowers to an ordinary
`ActorPoolMapOperator` wrapper. There is no groupby-specific admission code.

## Shuffle fixed-gang adapter

`GPUShuffleOperator` reports one non-elastic aggregate floor containing
`CPU=rank_count, GPU=rank_count`, with `min_units=max_units=1`.
`GPUHashAggregateOperator` inherits this adapter.

`GPURankPool` uses this asynchronous lifecycle:

```text
INACTIVE -> STARTING -> READY -> CLOSED
               \-> FAILED
```

Activation creates one atomic Ray placement group with `SPREAD` and pins rank
`i` to bundle `i`.
The operator polls placement-group readiness, then root UCXX setup and worker
setup, without blocking topology construction. Input is rejected until every
rank is ready. Placement-group provisioning is not charged against the UCXX
startup timeout, so a legitimate autoscaler node launch is not misreported as
a communicator failure.

Before the first insert, a revoked grant removes an idle or pending placement
group. The first insert makes the gang sticky. It remains owned until all
extraction generators finish, at which point every actor is killed and the
placement group is removed immediately, even if extracted Arrow output is
still queued downstream. Timeout, cancellation, or partial failure performs
the same whole-gang cleanup.

The configured `gpu_shuffle_num_actors` is preserved. When unset, rank count
continues to default to Ray's maximum configured GPU capacity when available,
or the current cluster's GPUs otherwise.

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
- a future GPU sort, join, or communicator group can report an aggregate floor;
- a future `TaskPoolMapOperator` adapter can report one task as its unit and
  gate submissions with `max_units` and `may_submit`; and
- a new physical operator can implement the three `PhysicalOperator` hooks
  without depending on this plugin.

Production GPU tasks retain legacy scheduling in this release. A future
adapter should extend behavior, not grow an enum or introduce a second state
machine.

## Validation

Implemented coverage includes:

- elastic and aggregate-gang contract tests;
- topology chains, fan-in, fractional GPUs, complete gang floors, frontier
  behavior, sticky gangs, and drained-ancestor handoff;
- zero-grant startup, fixed/autoscaling actor pools, asynchronous minimum-actor
  waiting, active-call drain, and pending/idle cleanup;
- atomic placement, exact bundle indices, repeated labels, readiness, revoke,
  timeout, partial failure, cancellation, and placement-group cleanup;
- inherited `GPUHashAggregateOperator` behavior;
- controller-wide warning and whole-topology fallback for unsupported
  constrained resources; and
- plugin compatibility against the exact two-field grant shape.

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
| Contract, hooks, controller, and allocator integration | 353 | 341 |
| Two-phase executor startup and backpressure gates | 50 | 44 |
| Actor adapter and autoscaler integration | 136 | 111 |
| Shuffle placement-group and asynchronous gang lifecycle | 262 | 141 |
| Context capability flag | 9 | 7 |
| **Total Ray production code** | **810** | **644** |

Tests, documentation, release notes, harnesses, and plugin code are excluded.
The audit is reproducible with `scripts/audit_candidate_loc.py` and enforces a
650-net-NCLOC ceiling. The simplification removed separate kind/bundle/version
and state abstractions, reused Ray's allocator/autoscaler/placement groups, and
kept only the lifecycle needed for liveness and cleanup.

## Current limitations

- Production GPU task admission is not wired yet.
- Dynamic per-actor resource callbacks have no trustworthy static envelope and
  therefore trigger the whole-topology admission fallback.
- Undeclared GPU operators retain stock behavior, so the deadlock guarantee
  covers declared participants only.
- Placement-group provisioning follows Ray's scheduler and autoscaler and is
  not bounded by `gpu_shuffle_setup_timeout_s`. Explicit execution limits fail
  an oversized gang during planning; without such a limit, an explicitly
  impossible rank can remain pending like other infeasible Ray requests.
- Admission capacity uses aggregate `ExecutionResources`; constrained custom
  resources, label selectors, and unsupported placement strategies therefore
  trigger eager placement-group-only behavior rather than claiming an unproven
  guarantee.
- The contract is private and deliberately has no version/public compatibility
  promise; the candidate wheel is checked by exact grant fields and provenance.
