# Why Ray Data Needs GPU Resource Admission

## Problem: valid GPU requests can form a closed wait

Ray Data currently starts resource-owning physical operators before it has made
a pipeline-wide decision about which operator must make progress first. Every
individual actor or shuffle-rank request can be valid while their acquisition
order is globally impossible:

```text
upstream GPU owner holds capacity
-> downstream owner or fixed gang requests the same capacity
-> upstream output reaches backpressure
-> upstream cannot finish and release
-> downstream cannot start and consume
```

This is a temporal ownership problem, not just a request for more aggregate
GPUs. It has two user-visible forms:

1. At a capacity ceiling, the pipeline enters a closed wait and never
   completes.
2. With enough capacity to avoid the closed wait, downstream actors can reserve
   GPUs long before usable input arrives, increasing reserved-capacity time and
   slowing the pipeline.

The anonymized incident-derived shape is:

```text
GPU actor map
-> GPU shuffle/groupby
-> GPU map_groups
-> GPU actor map
```

The future-facing proxy removes the separate `map_groups` pool by using Ray's
native GPU hash aggregate, then deliberately places an ordinary CPU operator
before another GPU actor:

```text
GPU actor map
-> GPUHashAggregateOperator
-> CPU operator
-> GPU actor map
```

An ordinary CPU operator is not a Dataset execution boundary. It does not stop
the downstream GPU actor from being created early, and backpressure can still
propagate across it.

### Liveness evidence

A timeout is classified as a structural closed wait only when all four
predicates remain true for at least 30 seconds: every logical GPU has an
observed owner, no logical GPU is available, another GPU actor or placement
group is pending, and the configured GPU ceiling has been reached.

| Workload and capacity | Existing workload elapsed outcome | Closed-wait evidence | Admission workload elapsed outcome |
| --- | --- | --- | --- |
| Incident-derived, 4 GPUs, rank 4 | Stock times out at 120.042 s | 112.925 s / 56 samples | Completes in 18.812 s |
| Incident-derived, 4 GPUs, rank 4 | Placement-groups-only times out at 180.152 s | 177.465 s / 298 samples | Completes in 18.554 s |
| Native aggregate + CPU gap, 4 GPUs, rank 4 | Stock times out at 119.982 s | 110.387 s / 23 samples | Completes in 13.365 s |
| Native aggregate + CPU gap, 4 GPUs, rank 4 | Placement-groups-only times out at 74.961 s | 65.313 s / 14 samples | Completes in 13.365 s |
| Actor-only chain, 2 GPUs, no shuffle | Stock times out at 180.023 s | 169.684 s / 79 samples | Completes in 10.128 s |

Provenance note: native admission and PG-only use the exact current candidate;
incident-derived and actor-only admission results use predecessor candidates.
Stock controls use the pinned base Ray wheel. Every row named in the final
decision must be refreshed on the frozen review wheels. Appendix B records the
main pins and artifacts and identifies the remaining provenance gaps.

The wait graphs identify the owners and requests, not merely a lack of
application logs. In both stock incident shapes, the upstream GPU actor and
three ranks own all four GPUs while the fourth rank waits. In the
placement-group-only case, the complete four-GPU gang owns the pool while
upstream and downstream actors wait. In the actor-only case, two live map
actors own the pool while later map actors wait.

The actor-only result is important: shuffle configuration cannot be the general
answer because the same ownership cycle exists without a shuffle.

### Evidence when the pipeline does not deadlock

The 16-GPU performance experiments are a separate, deliberately
non-deadlocking capacity regime—not a claim that every test in this report
completes. They use seven shuffle ranks and actor-pool minimum size one. The
incident-derived minimum floors total 10 GPUs (one upstream actor, seven ranks,
one `map_groups` actor, and one terminal actor); the native-aggregate floors
total nine. Both fit within 16 GPUs, so the observed eager acquisition order
still leaves enough capacity for every required owner to make progress.

By contrast, at rank four the same floors require seven and six GPUs
respectively, but the liveness cluster has only four. Eager ownership can
therefore form the closed waits above. Every arm in the two specific 16-GPU
performance comparisons completes, while the constrained four-GPU and
two-GPU liveness arms remain the deadlock evidence:

| Workload | Arm | Pipeline time | Cluster-capacity GPU-seconds | Premature downstream GPU-seconds |
| --- | --- | ---: | ---: | ---: |
| Incident-derived | Stock | 71.895 s | 1,163.953 | 117.4 |
| Incident-derived | Admission | 53.379 s | 864.924 | 14.0 |
| Native aggregate + CPU gap | Stock | 33.624 s | 549.065 | 24.440 |
| Native aggregate + CPU gap | Admission | 30.674 s | 495.277 | 1.686 |

Provenance note: native admission uses the exact current candidate and incident
admission uses its immediate predecessor; both stock arms use the pinned base
Ray wheel. These are one-run screening comparisons, not final timing
distributions.

Cluster-capacity GPU-seconds integrates the GPUs present in the Ray cluster over
wall time; it is a capacity/time and cost proxy, not measured GPU compute
utilization or a sum of actor reservations. The premature-ownership column is
the separate direct lower-bound measurement of early actor ownership.

The premature-ownership metric is a measured lower bound, not a claim that the
GPU was physically idle for every second. For the incident-derived stock run,
the terminal actor alone was ready for 63.546 seconds before its first usable
input; two `map_groups` actors contribute another 53.887 seconds. GCS ownership
corroborates those intervals.

Admission therefore improves more than liveness on equal hardware:

- on the incident-derived workload it reduces wall time by 25.8% and
  cluster-capacity GPU-seconds by 25.7% versus stock;
- on the native aggregate workload it reduces wall time by 8.8%,
  cluster-capacity GPU-seconds by 9.8%, and the observed premature-ownership
  lower bound by 93.1% versus stock; and
- within each workload, every completing arm produces the same rows, schema,
  and deterministic content hash.

These are one-run DGX screening results. They establish the failure mechanism
and a performance direction; the hardened provenance refresh, scaling curve,
cloud autoscaling run, and final correctness matrix remain merge gates.

## Existing approaches compared with admission

The claim is not that every workaround always fails. A user can make a
particular DAG complete by changing its rank, adding capacity, or inserting a
barrier. The merge case is that none of those is a general, automatic, and
resource-efficient ownership policy for Ray Data.

Every existing workaround addresses one dimension—placement atomicity,
physical capacity, a particular rank, or an explicit phase boundary. In the
current fixed-capacity DGX screening evidence, admission is the only tested
mechanism that preserves the requested operator shape, orders ownership across
the physical DAG, fixes both shuffle and shuffle-free closed waits, and
improves equal-resource performance on both large incident-relevant controls.
The final-wheel refresh and live G6 comparison remain merge gates. The safe
actor-only control has a measured 3.4% admission regression.

| Existing approach | What it can solve | Evidence relative to admission | Why it is not sufficient |
| --- | --- | --- | --- |
| Current Core scheduling and proportional Ray Data allocation | Places valid individual requests and shares throughput budgets | Stock deadlocks at 4 GPUs; admission cuts wall time by 8.8-25.8% on the two 16-GPU controls | Neither layer infers DAG progress floors before eager acquisition |
| Atomic placement groups | Prevents partial shuffle-gang acquisition | PG-only still deadlocks in both 4-GPU shapes; admission cuts wall time by 3.2-9.2% when both complete | Atomicity answers “all ranks together,” not “which stage owns GPUs now” |
| Autoscaling or overprovisioning | Adds GPUs for pending feasible requests | Fixed-capacity evidence only; cloud comparison open. At a configured ceiling the measured cycles remain; on an already sufficient 16-GPU host admission uses fewer cluster-capacity GPU-seconds | Capacity supply does not order owners, and enough capacity for every eager stage is an expensive workaround |
| Lower shuffle rank or actor-pool sizes | Can make one small shape fit | Stock rank 1 completes the small control, but the scaled rank-1 control OOMs; the actor-only cycle has no rank to tune. Best completing stock-rank sweep remains open | There is no workload-independent safe rank, and pools already at `min_size=1` can still participate in the cycle |
| Explicit `materialize()` boundaries | Breaks a known cycle into phases | Essentially tied at 4 GPUs; admission cuts wall time by 4.7% and 17.4% on the two large no-spill controls | Requires DAG-specific surgery, loses streaming overlap, and cannot split inside a fused/grouped operator |
| Object spilling | Relieves object-store pressure while Arrow blocks queue | The 16-GPU headline controls do not spill, yet stock is slower; spilling does not revoke an actor or order the next GPU owner | Storage pressure and GPU ownership are different problems; admission still reuses Ray's spiller |
| Fuse shuffle and `map_groups` | Removes one GPU owner | The native aggregate lifecycle proxy still deadlocks under stock and PG-only; admission completes and cuts wall time by 8.8% at scale | Fewer owners reduce risk but do not order the remaining upstream gang and downstream actor |

### Current Core scheduling and proportional allocation

Ray Core correctly places actors, tasks, and submitted placement groups. The
current Ray Data allocator correctly distributes proportional budgets. The
missing information is temporal: which independently submitted requests belong
to one physical DAG, which owner is an unfinished ancestor, which edge is
backpressured, and when an owner can release.

Stock Ray enters a certified closed wait on the native-aggregate shape. When it
does complete at 16 GPUs, admission reduces wall time by 25.8% on the
incident-derived workload and 8.8% on the native-aggregate workload with the
same ranks, data, and hardware.

Conclusion: existing Core scheduling is the correct placement mechanism, but
it cannot by itself guarantee liveness for these Ray Data DAGs. Admission
should feed ordered requests into Core rather than replace it.

### Atomic placement groups

Atomic placement is necessary for an indivisible shuffle gang. It prevents
stock's partial-gang failure, but it does not decide when that complete gang may
own the cluster. Using workload elapsed time for the liveness comparisons:

- incident-derived PG-only times out at 180.152 seconds while admission
  completes in 18.554 seconds;
- native-aggregate PG-only times out at 74.961 seconds while admission completes
  in 13.365 seconds; and
- at 16 GPUs, admission reduces wall time by 9.2% versus PG-only on the incident
  workload and 3.2% on the native-aggregate workload.

Conclusion: the PR should retain atomic placement groups as a building block,
but placement groups alone are neither the liveness fix nor the full
performance fix.

### Autoscaling or adding more GPUs

Autoscaling answers how much physical capacity should exist. Admission answers
which DAG owner may acquire that capacity now. Autoscaling can help while a
feasible request is below the configured maximum, but it cannot add another GPU
after the fleet reaches that maximum. Every four-GPU closed-wait result above
is measured at that state.

Overprovisioning until every eagerly created stage fits can avoid the cycle, but
that makes aggregate concurrent stage demand—not useful work—the required fleet
size. The non-deadlocking 16-GPU results show the remaining cost: on identical
capacity, admission reduces cluster-capacity GPU-seconds by 25.7% and 9.8%
versus stock on the two workloads.

Code inspection shows another coupling.
`execution/operators/hash_shuffle.py::_get_total_cluster_resources()` uses the
autoscaler-v2 maximum configured resources when present, and
`gpu_shuffle/hash_shuffle.py::_derive_num_gpu_ranks()` uses that GPU count when
`gpu_shuffle_num_actors` is unset. Raising the maximum can therefore enlarge
the gang along with the fleet instead of creating guaranteed headroom. This is
a source-level observation; the G6 campaign must still measure the runtime
effect.

The candidate composes with autoscaling. It exposes a complete pending floor
when more nodes are needed, retains an already-created cold-start gang request
through upstream handoff, and starts the UCXX setup timeout only after the
placement group is ready. Focused tests cover that request path.

Conclusion: autoscaling remains the capacity mechanism, but it is not a
substitute for DAG ownership ordering. A real G6 autoscaling run is still
required to quantify node-launch and multi-host costs; the current evidence
does not claim that cloud result yet.

### User tuning of shuffle rank and actor pools

Rank tuning is a useful optimization, and the evidence should not pretend
otherwise. In the small four-GPU incident control, stock rank-1 final
materialization takes 11.519 seconds while the then-candidate full-rank final
materialization takes 15.989 seconds. For that one shape, tuning is the faster
workaround.

It is not a general policy:

- the corresponding four-GPU, one-billion-row rank-1 run fails in cuDF sort
  with GPU out-of-memory because one rank concentrates the partition, while
  admission preserves rank 4 and completes final materialization in 71.640
  seconds;
- the two-GPU actor-only chain has no shuffle rank to tune, yet stock times out
  after 180.023 seconds while admission completes in 10.128 seconds; and
- the native-aggregate deadlock already uses elastic actor pools with
  `min_size=1`, so the minimum cannot be lowered further without removing the
  GPU stage.

On a safe four-GPU actor-only shape with one actor per stage, stock completes
final materialization in 43.515 seconds and admission in 44.989 seconds, a 3.4%
admission regression. With two requested actors per stage on two GPUs, stock
enters the closed wait while admission preserves both actors and completes.
This is the intended trade: bounded lifecycle overhead on an already-safe shape
instead of silently changing requested parallelism on an unsafe one.

Admission preserves the chosen shuffle rank and the user's
`ActorPoolStrategy` bounds. A tuned pipeline can still use admission; users no
longer need rank selection to double as a liveness mechanism.

Conclusion: rank and pool sizing should remain performance and memory controls,
not a correctness obligation placed on every Ray Data user. The final scaling
curve must still compare admission against the best completing stock ranks, not
only rule out rank 1.

### Explicit `materialize()` boundaries

Materialization is a valid manual escape because it forces one GPU phase to
finish before constructing the next:

- on the small native-aggregate control, materialized stock has slightly lower
  pipeline materialization time than admission: 11.221 versus 11.295 seconds;
- on the two-billion-row native-aggregate control, admission takes 30.674
  seconds versus 37.150 seconds for materialized stock, a 17.4% wall-time
  reduction and a 17.6% reduction in cluster-capacity GPU-seconds; and
- on the large incident-derived control, admission reduces wall time by 4.7%
  and cluster-capacity GPU-seconds by 4.0%.

The workaround requires application-specific DAG surgery, removes streaming
overlap, and cannot be inserted between the internal shuffle and `map_groups`
parts of a grouped operation without rewriting that operation. It can still be
the right choice under severe object-store pressure, as described in the
spilling comparison below.

Conclusion: `materialize()` should remain an explicit tool and forced-spill
performance remains an admission optimization target. It is not an acceptable
default liveness protocol that users must discover and maintain.

### Object spilling

Spilling relieves object-store pressure by moving queued Arrow blocks to disk.
It does not revoke a GPU actor, atomically acquire a gang, or choose which
physical operator should own GPUs next. The two 16-GPU headline comparisons
complete without spilling, so spill overhead did not cause their stock
slowdowns. Independently, the owner graphs show that spilling cannot complete a
partial gang, revoke an actor, or choose the next GPU owner.

Admission keeps Ray's existing object spiller as part of the handoff: upstream
Arrow output can queue or spill on RAID while active work drains, after which
the GPU owner is released. This separates the storage mechanism from the
ownership decision.

Forced-spill performance is an honest limitation. With a 16 GiB object store,
build-to-final-materialization time is 77.280 seconds for materialized stock and
85.137 seconds for the then-current streaming candidate. Both spill roughly
71-74 GB.

Conclusion: spilling is complementary backpressure relief, not a resource
admission policy. The PR is needed for ownership ordering, while forced-spill
performance remains a merge optimization target.

### Fusing shuffle and `map_groups`

Fusion removes the separate `map_groups` actor pool, but the remaining shape
still contains an upstream GPU actor, a fixed GPU gang, ordinary CPU work, and
a downstream GPU actor. The native `GPUHashAggregateOperator` proxy directly
tests that resource lifecycle:

- stock and PG-only both enter structural closed waits at four GPUs;
- admission completes pipeline materialization in 11.295 seconds; and
- at 16 GPUs, admission reduces wall time by 8.8% versus stock, 3.2% versus
  PG-only, and 17.4% versus materialized stock.

Conclusion: fusion is complementary throughput work, not a replacement for
resource admission. A future arbitrary fused UDF still needs its own correctness
tests and must implement the small admission contract.

On the current evidence, placement groups, tuning, materialization, spilling,
and fusion are ruled out as general replacements: each either leaves a tested
closed wait, changes the requested rank or pool shape, requires DAG surgery, or
loses equal-resource scale performance. Autoscaling is structurally
complementary rather than equivalent, but its live cloud cost and latency
comparison remains the one major workaround result that has not yet been
measured.

## Proposed Ray Data change

### Why this policy belongs in Ray Data

Ray Core remains responsible for placing individual actors, tasks, and
placement-group bundles on nodes. Once Ray Data submits a placement group, Core
does know and atomically schedule that gang. It cannot infer that independently
submitted requests form one Ray Data progress floor, which owners are
unfinished ancestors, which streaming edge is backpressured, or when that floor
should be exposed.

The change adds the missing temporal policy inside Ray Data: decide which
physical operators may create or retain scarce owners, then let Ray Core place
the admitted requests normally. It does not replace Core scheduling,
placement-group semantics, or cluster autoscaling.

Stock execution starts resource-owning operators while constructing the
streaming topology. Actor pools can therefore create their initial actors and a
stock shuffle can request its rank actors before Ray Data has selected the owner
that must run first. The proportional resource allocator acts after this
acquisition has begun and can assign fractional budgets that are not usable by
a one-GPU actor or an indivisible rank group. The candidate separately adds an
atomic placement group for the ranks; admission decides when that group may
activate.

### Small internal contract

Each supported physical operator describes its complete minimum progress floor:

```python
ResourceAdmissionSpec(
    minimum_resources,  # complete CPU/GPU/memory floor
    unit_resources,     # one elastic unit; None for a fixed gang
    min_units,
    max_units,
)
```

The controller returns only:

```python
ResourceAdmissionGrant(
    max_units,   # resource-owner/request cap
    may_submit,  # whether input may be consumed or new work submitted
)
```

`PhysicalOperator` supplies three private hooks: report the specification,
apply a grant, and report whether its resources can be released. There is no
public Dataset API, operator-kind enum, parallel admission state machine, or
Ray Core change.

### Startup and topological admission

Execution now follows this sequence:

1. Build every operator state and connect queues without starting workers.
2. Construct the resource manager, inspect the complete topology, validate
   every declared specification and known incompatibility marker, and apply an
   initial zero-unit grant.
3. Start operators; for managed owners, the initial zero grant keeps startup
   local-only.
4. Find current resource claimants and every unfinished declared GPU ancestor,
   including ancestors separated by ordinary CPU operators.
5. Protect non-releasable owners and scan the remaining claimants in
   deterministic topological order.
6. Admit complete minimum floors until the first floor does not fit. That
   operator becomes the frontier; later operators cannot leapfrog it.
7. Give remaining capacity to admitted elastic pools through Ray's existing
   resource allocator. When proportional operator reservations are disabled,
   the floors remain active and the controller caps admitted pools directly.

This is not a global GPU mutex. If all demanded floors fit, operators may
overlap and elastic pools may grow. Serialization occurs only when complete
progress floors cannot coexist. As one narrow overlap optimization, a declared
single-input direct managed successor of a fixed gang may expose its floor once
the gang's inputs are complete while the gang continues draining or extracting.
This prewarms a Core request, not guaranteed GPU ownership, and does not recurse
into later GPU stages.

### Actor-pool lifecycle

Actor-based GPU `map_batches`, actor-based cuDF `map_groups`, and plugin-created
GPU actor maps that lower to `ActorPoolMapOperator` with static resources share
one adapter:

- one actor's declared CPU/GPU/memory is the elastic unit;
- `ActorPoolStrategy.min_size` is the protected floor and `max_size` remains
  the growth limit;
- the initial pool is not created while its grant is zero;
- growth cannot exceed `grant.max_units`;
- demotion cancels pending actors and removes idle actors;
- active calls drain, but a demoted or excess actor does not take another
  internally queued bundle; and
- fixed `size=N` and `min_size=N` semantics remain a real `N`-actor floor.

The controller preserves fixed-size and `min_size`/`max_size` bounds; it may cap
an elastic pool's initial acquisition until capacity is granted. Admission
completion is deliberately narrower than whole-operator execution completion:
after inputs, internal queues, and active calls drain, an actor pool can release
its GPUs while already-produced Arrow output remains buffered downstream.

### Shuffle and hash-aggregate lifecycle

`GPUShuffleOperator` reports one fixed floor equal to its complete rank count.
`GPUHashAggregateOperator` inherits the same behavior. Admission activates one
atomic `SPREAD` placement group with one GPU/CPU bundle per rank; each rank
actor is pinned to its bundle.

Placement-group readiness and UCXX setup are polled asynchronously. Before its
first input, a revoked grant can remove an idle or pending group. After its
first insert, the gang is sticky and its unfinished declared producer closure
is protected. When extraction generators finish, the placement group is
removed even if extracted Arrow blocks remain queued downstream. Its removal
releases every rank reservation, and any remaining actors are killed. Timeout,
cancellation, and partial setup failure use the same whole-gang cleanup. If a
later operator fails during two-phase startup, already started operators are
stopped in reverse order.

### Handoff and backpressure

Admission preserves Ray Data's host-side block contract:

```text
GPU actor map: cuDF in worker -> Arrow ObjectRefs
GPU shuffle/aggregate: Arrow -> GPU ranks -> Arrow ObjectRefs
GPU map_groups: Arrow group -> cuDF -> Arrow ObjectRefs
```

When the downstream GPU floor cannot run, upstream Arrow blocks may queue or
spill while the active owner finishes. A targeted output-backpressure escape
path lets an active task drain enough generator output to complete and release
its GPU rather than waiting forever for the blocked downstream stage.

The intended constrained-capacity handoff is:

```text
admit upstream floor
-> finish active calls and queue/spill Arrow output
-> release upstream owners
-> admit the complete shuffle/aggregate gang
-> remove the gang after extraction
-> admit downstream actor floor
```

### Autoscaling interaction

Autoscaling answers “how many physical GPUs should exist?” Admission answers
“which DAG owner may hold them now?” When a minimum floor exceeds current
capacity but fits explicit Ray Data execution limits, the frontier exposes
pending actor or placement-group requests that the cluster autoscaler can act
on. Admission neither chooses a node shape nor directly commands the
autoscaler. An already-exposed fixed-gang request remains pending while an
earlier owner drains, avoiding a loss of the whole-gang demand as new capacity
arrives. Without a finite execution limit, an infeasible request may remain
pending under normal Core semantics.

Placement-group provisioning time is excluded from the UCXX setup timeout.
The timeout begins after the group is ready, so slow node launch is not
misclassified as communicator failure. Once capacity exists, the same
topological ownership policy applies.

### Compatibility, rollback, and scope

The controller is intentionally conservative. The actor-pool and shuffle
adapters mark recognized dynamic resource callbacks, custom resources, label
selectors, accelerator constraints, and unsupported placement strategies that
cannot be represented by the aggregate contract. If a recognized incompatibility
appears beside a managed owner, the controller clears every declared
specification with one warning. Actor pools then acquire eagerly and candidate
shuffle gangs activate eagerly: this is placement-group-only behavior, not
stock Ray.

One private `DataContext` flag, also backed by
`RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL`, disables topological admission.
On the candidate wheel this restores eager actor acquisition and eager atomic
gang activation; it is the placement-group-only ablation, not stock Ray.
Current production coverage is declared GPU actor pools, GPU shuffle, and GPU
hash aggregate. The default operator contract declares no floor, so unknown or
undeclared GPU owners retain stock scheduling and are outside the deadlock
guarantee. Production GPU task-pool admission is not included.

| Ray Data area | Change | Reused mechanism |
| --- | --- | --- |
| Physical operator interface | Private specification and grant hooks | Existing operator state |
| Streaming executor/resource manager | Two-phase startup and topological frontier | Existing topology and allocator |
| Actor pool/autoscaler | Grant-gated start, growth, drain, and cleanup | Existing actor pool |
| GPU shuffle/hash aggregate | Atomic asynchronous gang lifecycle | Core placement groups |
| Backpressure | Resource-starvation escape path | Existing output queues and spilling |
| Data context | One private rollback flag | Existing context/environment configuration |

The detailed contract and invariants are in
`docs/generic-resource-admission.md`; this report connects that implementation
to the measured reviewer evidence.

## Why the maintenance cost is bounded

The evidence above is the reason to carry this policy in Ray Data: it replaces
multiple workload-specific correctness workarounds with one private contract
at the layer that owns topology, backpressure, and operator lifecycle. The code
cost is intentionally capped and reuses existing Core and Ray Data mechanisms.

- Production scope: ten nonzero files under `python/ray/data`; no Ray Core
  change.
- Production delta: 810 added, 166 deleted, `+644` net NCLOC.
- API surface: no public API; one private rollback flag.
- Exact-wheel checks: commit, source tree, wheel, patch, and every changed
  production-file hash match the pins.
- Focused validation: 18 controller/autoscaling/backpressure tests and one
  logical two-GPU streaming integration pass.
- Evidence-harness validation: 154 local/cloud unit tests pass. Separate
  one-GPU streaming and manually materialized smokes produced exact output, no
  spill, and clean teardown against the real candidate/stock wheels; their
  captured layouts satisfy the current hardened guard.
- Spill paths are on `/raid` (`/dev/md127`), while short Ray sockets use
  `/dev/shm`; cleanup proves no owned actors, placement groups, Ray processes,
  or spill directories remain.

The future-fusion workload, plan guard, oracle hardening, launcher registration,
tests, and this report add no Ray production LOC. The LOC artifact is
[`benchmark/review_artifacts/candidate-loc-final-31d444.json`](../benchmark/review_artifacts/candidate-loc-final-31d444.json).

`benchmark/results/` is intentionally gitignored because full logs, isolated
runtimes, and spill data are large. The original JSON records cited by this
report are committed as the compact
[`benchmark/review_artifacts/dgx-screening-v1.tar.gz`](../benchmark/review_artifacts/dgx-screening-v1.tar.gz)
bundle. Its [review index](../benchmark/review_artifacts/README.md) maps each
claim to the corresponding experiment and records the archive checksum and
provenance limitations.

## Merge decision and remaining evidence

The current evidence justifies continuing with the small Ray Data design, but
it is not yet the final merge bundle. These are the pass/fail gates:

| Gate | Required result | Current status |
| --- | --- | --- |
| Structural liveness | On frozen final wheels, stock or PG-only reproduces each certified closed wait and admission completes every focused incident-derived, native-aggregate, and actor-only trial | Mechanism shown; predecessor-wheel rows and hardened-plan rows need the focused refresh |
| Best stock configuration | At 4/8/16 GPUs and equal data per GPU, admission preserves the requested full rank, is faster than the best completing stock rank on the scaled incident-derived workload, and delivers at least a 10% primary-workload gain | Faster than default stock at 16 GPUs; best completing rank sweep is open |
| Safe-workload overhead | Equal-shape workloads that already complete regress by no more than 5% | Actor-only screening regression is 3.4%; final-wheel refresh open |
| Materialization and spill | Admission beats materialized stock on the large no-spill controls and forced-spill regression is no more than 5% | Large no-spill controls pass; current forced-spill candidate is 10.2% slower and must improve |
| Correctness and cleanup | Completing arms have identical schema, row count, deterministic content, and no actor, placement-group, process, or spill leak; failure paths clean up as well | Headline oracles and cleanup pass; final failure matrix open |
| Autoscaling | At the same G6 ceiling, separate node-launch time from scheduling time; after capacity is ready, admission completes and beats the best stock workaround on the incident-derived shape without hiding cost in extra GPUs | Source path and fixed-ceiling mechanism tested; live G6 run open |
| Maintenance | No public API, no Ray Core change, and at most `+650` net production NCLOC | Currently passes at `+644`; remeasure after the final code change |

The remaining work should run in this cost-efficient order:

1. Refresh the exact four-GPU liveness matrix with frozen wheels and hardened
   plan capture. For both the incident-derived and native-aggregate workloads,
   run stock, placement-groups-only, admission, and materialized stock. Refresh
   stock versus admission on the two-GPU actor-only chain as the shuffle-free
   control.
2. Run the same per-GPU data scale at 4, 8, and 16 GPUs for admission and every
   completing stock rank, then compare with the best stock rank at each scale.
   Run one randomized 16-GPU block containing stock, placement-groups-only,
   admission, and materialized stock for both headline workloads.
3. Profile and improve the forced-spill handoff until it meets the 5% gate;
   repeat the unchanged materialized-stock control after each candidate change.
4. Add timing repetitions only if the refreshed gaps or scaling curve are
   noisy enough to change the decision.
5. After the policy is frozen, run the targeted correctness, failure, and
   cleanup matrix.
6. Finally run one fixed and one autoscaling G6 campaign. Separate node request
   to node ready from node ready to first progress, and compare a warm replay
   on the already-expanded cluster.

The cloud decision answers one narrow question: after autoscaling has provided
the same maximum GPUs, does Ray Data admission still reduce premature
ownership and improve completion time? The DGX screening evidence says yes on
one host; the G6 run must test whether that survives real launch and network
costs.

## Appendix A: Detailed evidence

### Incident-derived workload

```text
GPU actor map
-> GPU shuffle/groupby
-> GPU map_groups
-> GPU actor map
```

The performance comparison uses 16 V100 GPUs, 2,000,000,000 input rows, 2,048
blocks, seven shuffle ranks, elastic GPU map pools with `min_size=1` and
`max_size=16`, no synthetic GPU work, and a 128 GiB object store. No arm spilled.

| Arm | Build start through final materialization | Cluster-capacity GPU-seconds | Result |
| --- | ---: | ---: | --- |
| Full admission | 53.379 s | 864.924 | 64 rows, matched |
| Placement groups only | 58.771 s | 953.354 | 64 rows, matched |
| Stock Ray | 71.895 s | 1,163.953 | 64 rows, matched |
| Stock Ray + manual materialization | 56.005 s | 901.290 | 64 rows, matched |

The full and placement-groups-only arms use the same candidate wheel and frozen
harness; the controller mode is the meaningful difference. Their 9.2% gap is
therefore the direct evidence that admission adds performance beyond atomic
shuffle placement.

The telemetry also observes a lower bound of 14.0 premature downstream
GPU-seconds with admission, versus 100.7 with placement groups only and 117.4
with stock. This measurement is directionally useful, but its actor coverage is
explicitly incomplete and must be described as a measured lower bound.

#### What the actor maps measure

The actor maps deliberately separate useful data work from resource-lifecycle
evidence:

- `AddKey` copies each cuDF batch and computes `key = id % groups`. The headline
  run disables its optional synthetic CuPy loop, so the comparison does not
  manufacture a GPU-compute advantage.
- `SumGroup` is the GPU `map_groups` UDF. It returns one row containing
  `(key, sum(id))` for each group.
- terminal `Identity` records its first usable input and returns that input
  unchanged. It is a downstream GPU-owner probe standing in for inference or a
  later cuDF transform, not useful application compute.

Every actor-map stage uses a persistent `ActorPoolStrategy` and reserves one
GPU for each live actor. In the stock 16-GPU artifact, the observed terminal
`Identity` actor became ready at 8.691 seconds but received its first and only
64-row input at 72.237 seconds. It therefore contributes a directly observed
lower bound of 63.546 prematurely reserved GPU-seconds. The two `SumGroup`
actors that were ready before their stage's first input contribute another
53.887 GPU-seconds by the stricter stage-first-input definition. Together, the
documented lower bound is 117.433 GPU-seconds. These intervals use a verified
single-boot monotonic clock and are corroborated by GCS `ALIVE` GPU ownership;
the lower-bound qualifier remains necessary because the elastic-pool telemetry
is not complete.

### Four-GPU incident liveness and placement-group ablation

At four GPUs and shuffle rank four, full admission completes in 18.554 seconds
of workload elapsed time, including 16.464 seconds of materialization.
Placement-groups-only times out after 180.152 seconds. For 177.465 seconds and
298 samples, every closed-wait predicate is true:

- all four logical GPUs have an observed owner;
- no logical GPU is available;
- GPU actor requests are pending; and
- the configured GPU ceiling has been reached.

The wait graph is concrete: one four-GPU shuffle placement group owns the whole
pool while `AddKey`, `SumGroup`, and terminal `Identity` actor creations each
wait for one GPU. Atomic acquisition prevents a partial shuffle gang, but it
does not decide *when* that gang may own the GPUs. The admission controller is
the layer that orders ownership across the Ray Data DAG.

Stock Ray independently exhibits the original failure mode. In the paired
16-million-row four-GPU run, stock rank four sustains a structural closed wait
for 112.925 seconds, while final materialization takes 11.519 seconds for stock
rank one and 15.989 seconds for the then-current candidate at rank four. This
supporting run predates the exact final wheel; it is useful evidence that rank
tuning can fix one small instance, not evidence that tuning is useless.

### Native aggregate and future-fusion proxy

Fusing the GPU shuffle and `map_groups` UDF would remove the separate
`SumGroup` actor pool. It would reduce contention, but it would not eliminate
the ownership-ordering problem. The relevant future resource-lifecycle shape is
now a directly tested benchmark:

```text
upstream GPU actor A
-> fused GPU shuffle/group operator F
-> ordinary CPU operators
-> downstream GPU actor B
```

The proxy uses public Ray Data APIs. A typed GPU actor expression computes the
key while preserving static schema inference, the native
`GPUHashAggregateOperator` performs shuffle plus partial/final reduction, a CPU
expression project multiplies the result by two, and a nontrivial GPU `AddOne`
actor consumes it. Every completing arm checks the exact per-key result
`2 * sum(id) + 1` and rejects duplicate keys. The harness also guards the exact
optimized physical shape; its rollout status is recorded in the provenance
section.

This is evidence for the fused operator's fixed-gang resource lifecycle, not a
claim about the behavior or performance of every possible future arbitrary
`map_groups` UDF. Such an operator must implement the same admission contract
and needs its own correctness and performance tests when it exists.

#### Four-GPU liveness

The four-GPU run uses 16 million rows, 32 blocks, rank four, and elastic GPU
actor pools with `min_size=1,max_size=4`.

| Arm | Workload elapsed outcome | Structural closed-wait tail | Result |
| --- | ---: | ---: | --- |
| Full admission | 13.365 s | none | 64 exact rows |
| Placement groups only | timed out at 74.961 s | 65.313 s / 14 samples | no progress |
| Stock Ray | timed out at 119.982 s | 110.387 s / 23 samples | no progress |
| Stock Ray + two materializations | 13.299 s | none | 64 exact rows |

Stock Ray partially acquires the indivisible aggregate: the upstream GPU
`Project` actor and three `GPUHashAggregateActor` ranks own all four GPUs while
the fourth aggregate rank waits for one. The placement-group-only arm prevents
that partial gang but still deadlocks: its four-GPU aggregate placement group
owns the complete pool while both the upstream GPU `Project` and downstream GPU
`AddOne` actor creations wait.

The resource-relevant suffix of the physical plan in every non-materialized arm
is:

```text
ActorPoolMapOperator[Project]
-> GPUHashAggregateOperator
-> TaskPoolMapOperator[Project]
-> ActorPoolMapOperator[MapBatches(AddOne)]
```

The intervening CPU operator therefore does not create a resource-lifecycle
boundary or prevent the closed wait. Explicit materialization is a valid manual
escape at this small scale.

#### Sixteen-GPU non-deadlocking performance

The scale run uses 16 V100 GPUs, two billion rows, 2,048 blocks, rank seven,
elastic GPU actor pools with `min_size=1,max_size=16`, and a 128 GiB object
store. Every arm completes without spill or restore.

| Arm | Build start through final materialization | Cluster-capacity GPU-seconds | Premature downstream GPU-seconds |
| --- | ---: | ---: | ---: |
| Full admission | 30.674 s | 495.277 | 1.686 |
| Placement groups only | 31.680 s | 511.362 | 28.104 |
| Stock Ray | 33.624 s | 549.065 | 24.440 |
| Stock Ray + two materializations | 37.150 s | 601.198 | 2.144 |

The premature-ownership values remain measured lower bounds. Admission reduces
that bound by 93.1% versus stock and 94.0% versus placement groups only.

Ordinary CPU operators do not create a Dataset execution boundary. They can
buffer or transform streamed blocks, but do not prevent `B` from being created
before usable input arrives, and output backpressure can propagate through
them. An explicit intermediate `materialize()`, a demand-driven and
topologically ordered actor lifecycle, or enough GPUs for every simultaneous
minimum floor removes this particular resource cycle. If the fused operator
has no GPU ancestor, acquires its gang before `B`, fully drains its output, and
releases the gang, that execution can complete; the defensible claim is
therefore “can deadlock depending on acquisition order,” not “always
deadlocks.”

The admission controller already walks ancestry through non-participating CPU
operators. A future fused owner must expose one fixed aggregate floor, apply
its grant to resource acquisition, and report when ownership is releasable; a
new physical operator that does not implement that small contract is not
protected automatically.

## Appendix B: Evidence provenance and current limits

- Current candidate: `31d4448482c1a47e034e9160d1d8f41e84cd1a25`
- Base Ray: `2741c6461d2bd3e5ff114af67be7a1190453dadd`
- Candidate wheel SHA-256:
  `e26e04f95572010151f379f044f10f01bcf716bde49caf2688f586781f500309`

The incident-derived admission and PG-only DGX measurements were captured from
immediate predecessor `21c102ed462f1821542aac1139d5269922275e8f`, wheel
`a6ea72a6d24045738220b2edb2884588ae8753b4498fd442a009ef1051afc9f4`.
The stock and materialized-stock controls use the base Ray wheel recorded in
their manifests.
The current commit retains an already-created cold-autoscaling gang request
until upstream handoff, corrects the setup-timeout documentation, and removes
one readiness log. The demand-retention branch is not entered by those
fixed-capacity cases.

The native-aggregate admission and PG-only measurements were captured from the
exact current candidate and wheel pinned above. The stock and
materialized-stock controls use the base Ray wheel recorded in their manifests.
The hardened harness now persists and validates the complete optimized physical
layout before terminal execution; manual-boundary controls checkpoint each
phase plan before blocking materialization. A completing run must also report
runtime stats containing `GPUHashAggregate(`. A silent CPU-aggregate fallback,
unexpected operator, or optimizer removal of the CPU gap therefore cannot count
as evidence.

The native headline runs predate that hard guard. Their executor logs record
the required physical sequence, runtime stats verify
`GPUHashAggregateOperator` for completing arms, and GCS state corroborates the
timed-out owner graphs. Separate one-GPU streaming and manual-boundary smokes
captured layouts that satisfy the current hard guard against the real wheels.
One focused headline refresh remains required so every published
`workload.json` embeds guard execution.

The stock native timeout logs a UCX bind error only after harness teardown
releases a GPU and lets the previously pending aggregate actor start. The
110.387-second closed-wait interval and partial-gang owner graph precede that
error, so the deadlock classification does not depend on the teardown symptom.

Correctness and cleanup are checked independently of timing:

- all four completing 16-GPU incident performance arms produce 64 rows, content digest
  `975f27249f6230ee4f66fe48efd8bbeccb24135fe32964d05750b9bcde25162b`,
  and schema hash
  `e970ea10b5b479c42a961541e65e56984196c60341a0c61cc59e4c7b26679b83`;
- all four 16-GPU native-aggregate arms produce 64 exact rows, content digest
  `13e02773a8544941f3e39369b44c183b0faca84799bee0ba1c5fb89d5a40e9a6`,
  and the same schema hash. The two completing four-GPU arms also pass the exact
  per-key oracle and match each other; and
- every headline job and isolated cluster proves cleanup. The 16-GPU headline
  runs report no spill or restore.

Raw artifact index:

- incident, placement-group ablation, and materialization:
  `benchmark/results/local/perf16-final-full-vs-pg-r7-w0-v1`,
  `benchmark/results/local/perf16-final-stock-r7-w0-v1`,
  `benchmark/results/local/perf16-final-stock-materialize-r7-w0-v1`, and
  `benchmark/results/local/final-liveness-full-vs-pg-g4-r4-v1`;
- native aggregate, placement-group ablation, and materialization:
  `benchmark/results/local/aggregate-cpu-gap-g4-r4-v1`,
  `benchmark/results/local/aggregate-cpu-gap-pg-g4-r4-v1`,
  `benchmark/results/local/agg-gap-stock-mat-g4-r4-v1`,
  `benchmark/results/local/agg-gap-perf16-r7-v1`, and
  `benchmark/results/local/agg-gap-stock-mat16-r7-v1`;
- hardened-layout smokes:
  `benchmark/results/local/aggregate-plan-smoke-v2` and
  `benchmark/results/local/aggregate-plan-materialized-smoke-v1`;
- rank and actor-pool controls:
  `benchmark/results/local/stock-workaround-16m-20260722a`,
  `benchmark/results/local/perf4-1b-v1`, and
  `benchmark/results/local/local-multigpu-actor-baseline-20260722a`; and
- forced-spill controls:
  `benchmark/results/local/perf16-2b-16g-stock-mat-r7-w0-v1`,
  `benchmark/results/local/perf16-2b-16g-min-r7-w0-v3`, and
  `benchmark/results/local/perf16-2b-16g-min-r7-w0-v4`.

The small rank-tuning and actor-only controls also predate the exact final
wheel. They establish the mechanism and the next focused comparisons, but the
final-wheel refresh remains a merge gate.

These results are screening evidence, not final statistics. A provenance
refresh, a 4/8/16-GPU scaling curve, real G6 autoscaling and multi-host UCXX,
and the final correctness/failure matrix remain open.
