# Generic Resource Admission for GPU Actors and Shuffle Gangs

Status: implemented in PR-candidate commit `8ecf8f98d65157fbbf10129285547e554cc74e05`
and preserved on `codex/generic-resource-admission`, replacing the actor-specific
contract in Ray commit `ede9354`.

## 1. Problem

Ray Data can place several long-lived GPU resource owners in one physical DAG:

- actor-based `map_batches`;
- the rank actors for GPU shuffle;
- the actor that runs `map_groups(batch_format="cudf")`;
- `GPUHashAggregateOperator`, which uses the same rank-gang machinery; and
- future GPU tasks, sorts, joins, or operators introduced by other Ray APIs.

Ray Core schedules each actor or task correctly, but it does not know which
owners form one Ray Data pipeline, which owner is an unfinished ancestor, or
when an idle owner should release a GPU so another stage can make progress.
Starting every pool and shuffle rank independently can therefore strand a
pipeline when their combined GPU demand exceeds the execution capacity.

For example, the following physical chain has four long-lived resource owners:

```text
ActorPoolMap(map_batches)
  -> Arrow ObjectRefs
GPUShuffle rank gang
  -> Arrow ObjectRefs
ActorPoolMap(map_groups wrapper)
  -> Arrow ObjectRefs
ActorPoolMap(map_batches)
```

On one GPU, none of the actor pools can coexist with a one-rank shuffle gang.
If later owners acquire or queue resources without a pipeline-level policy,
the upstream actor can be prevented from draining its streaming output while
the downstream owner waits for the GPU held upstream. Proportional resource
reservations alone do not fix this because a complete actor or gang is an
indivisible placement request, and current over-allocation must not inflate an
operator's future allocation target.

This must be solved in Ray Data rather than in a cuDF UDF or plugin. The
streaming executor is the component that owns the complete physical topology,
input and output queues, backpressure, operator resource accounting, and actor
autoscaling. No Ray Core change is required.

## 2. Why the design fixes it

Every supported GPU owner declares a versioned resource-admission
specification. The executor considers complete progress floors in deterministic
topological order before it shares excess capacity.

An actor pool's progress floor is one complete actor, even when its configured
minimum is larger. A shuffle's floor is its entire rank gang. The first floor
that does not fit is the frontier, and later participants are blocked from
leapfrogging it. Operators that lose a grant stop submitting new work; pending
and idle actors are released, while active calls finish normally.

This produces the intended one-GPU handoff:

```text
finish upstream map_batches
-> queue or spill Arrow blocks
-> release its actor
-> activate the complete shuffle gang
-> extract queueable/spillable Arrow blocks
-> remove the placement group
-> activate map_groups
-> release its actor
-> activate the final map_batches
```

The output-backpressure escape path lets the active upstream generator keep
queuing Arrow blocks when its downstream admission boundary cannot yet run.
That allows the upstream call to finish and its actor to become releasable.

The policy is not a global GPU mutex. If every demanded minimum fits, actor
stages and the gang are admitted together and may overlap. A default GPU
shuffle gang contains one rank per currently detected cluster GPU, so it
normally serializes with surrounding actors unless
`gpu_shuffle_num_actors` is configured below total capacity.

## 3. Internal contract

The contract is internal and explicitly versioned:

```python
RESOURCE_ADMISSION_CONTROL_VERSION = 1


class AdmissionKind(Enum):
    TRANSIENT = "transient"
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

## 4. Controller and executor lifecycle

### 4.1 Two-phase startup

Topology startup is split into two phases:

1. construct every `OpState` and wire all queues without starting operators;
2. construct `ResourceManager`, validate every specification, and apply an
   initial zero-unit grant;
3. call each operator's `start()` for local initialization; and
4. acquire workers only after the first admission update.

This prevents topology construction from eagerly consuming GPUs before the
controller has seen the whole DAG.

### 4.2 Participation

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

### 4.3 States and grants

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

## 5. Actor-pool adapter

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

## 6. Shuffle-gang adapter

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

## 7. Arrow boundary

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

## 8. Extension model

The controller is intended to remain unchanged as coverage expands:

- future APIs that lower to `ActorPoolMapOperator` inherit elastic admission;
- a future GPU sort, join, or communicator group supplies a fixed-gang spec;
- a future `TaskPoolMapOperator` adapter supplies one transient task bundle as
  its floor and gates submissions with `max_units`; and
- a new Ray API can implement the four `PhysicalOperator` hooks without
  depending on this plugin.

`TRANSIENT` behavior is covered with fake operators now, but production GPU
tasks intentionally remain warning-and-legacy in this release.

## 9. Validation

Implemented coverage includes:

- elastic, fixed-gang, and transient contract tests;
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

## 10. Implementation size

Measured against unmodified stock Ray commit
`2741c6461d2bd3e5ff114af67be7a1190453dadd`, counting added physical Python
source lines and excluding blank and comment-only lines:

| Production area | Added NCLOC | Net NCLOC above stock |
| --- | ---: | ---: |
| Contract, hooks, controller, and allocator integration | 458 | 435 |
| Two-phase executor startup and backpressure gates | 45 | 43 |
| Actor adapter and autoscaler integration | 198 | 167 |
| Shuffle placement-group and asynchronous gang lifecycle | 259 | 142 |
| Context capability flag | 6 | 6 |
| **Total Ray production code** | **966** | **793** |

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

## 11. Current limitations

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
