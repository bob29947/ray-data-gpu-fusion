# Ray Data GPU Admission: DGX Screening Evidence

- Current candidate: `31d4448482c1a47e034e9160d1d8f41e84cd1a25`
- Base Ray: `2741c6461d2bd3e5ff114af67be7a1190453dadd`
- Candidate wheel SHA-256:
  `e26e04f95572010151f379f044f10f01bcf716bde49caf2688f586781f500309`

The fixed-capacity DGX measurements below were captured from immediate
predecessor `21c102ed462f1821542aac1139d5269922275e8f`, wheel
`a6ea72a6d24045738220b2edb2884588ae8753b4498fd442a009ef1051afc9f4`.
The current commit retains an already-created cold-autoscaling gang request
until upstream handoff, corrects the setup-timeout documentation, and removes
one readiness log. The demand-retention branch is not entered by these
fixed-capacity cases; the next repeated timings must nevertheless use the
current wheel.

## Current conclusion

The candidate has earned continued validation. It is not merely a placement
group wrapper or a deadlock-only change:

1. Atomic placement groups without admission still enter a certified closed
   wait at the four-GPU ceiling. Admission completes the same case.
2. On a non-deadlocking 16-GPU, two-billion-row workload, admission has 25.8%
   lower pipeline wall time than stock Ray, 9.2% lower than
   placement-groups-only, and 4.7% lower than stock Ray with explicit phase
   materialization.
3. The completing arms return identical rows, schemas, and deterministic
   content hashes.
4. The implementation is confined to Ray Data, adds no public API, and is
   `+644` net production NCLOC.

These are one-run DGX screening results, not merge-ready statistics. Cloud
autoscaling, multi-host UCXX, repeated trials, and the full correctness matrix
remain open.

## Incident-derived workload

```text
GPU actor map
-> GPU shuffle/groupby
-> GPU map_groups
-> GPU actor map
```

The performance comparison uses 16 V100 GPUs, 2,000,000,000 input rows, 2,048
blocks, seven shuffle ranks, elastic GPU map pools with `min_size=1` and
`max_size=16`, no synthetic GPU work, and a 128 GiB object store. No arm spilled.

| Arm | Build start through final materialization | Cluster GPU-seconds | Result |
| --- | ---: | ---: | --- |
| Full admission | 53.379 s | 864.924 | 64 rows, matched |
| Placement groups only | 58.771 s | 953.354 | 64 rows, matched |
| Stock Ray | 71.895 s | 1,163.953 | 64 rows, matched |
| Stock Ray + manual materialization | 56.005 s | 901.290 | 64 rows, matched |

Relative to full admission:

- full admission reduces wall time by 25.8% and GPU-seconds by 25.7% versus
  stock Ray;
- it reduces wall time by 9.2% and GPU-seconds by 9.3% versus
  placement-groups-only; and
- it reduces wall time by 4.7% and GPU-seconds by 4.0% versus manual
  materialization.

All four arms produce content digest
`975f27249f6230ee4f66fe48efd8bbeccb24135fe32964d05750b9bcde25162b`
and schema hash
`e970ea10b5b479c42a961541e65e56984196c60341a0c61cc59e4c7b26679b83`.
Job-level and isolated-cluster cleanup are proven for every arm.

The full and placement-groups-only arms use the same candidate wheel and frozen
harness; the controller mode is the meaningful difference. Their 9.2% gap is
therefore the direct evidence that admission adds performance beyond atomic
shuffle placement.

The telemetry also observes a lower bound of 14.0 premature downstream
GPU-seconds with admission, versus 100.7 with placement groups only and 117.4
with stock. This measurement is directionally useful, but its actor coverage is
explicitly incomplete and must be described as a measured lower bound.

Raw artifacts:

- `benchmark/results/local/perf16-final-full-vs-pg-r7-w0-v1`
- `benchmark/results/local/perf16-final-stock-r7-w0-v1`
- `benchmark/results/local/perf16-final-stock-materialize-r7-w0-v1`

## Liveness and why placement groups are insufficient

At four GPUs and shuffle rank four, full admission completes in 16.464 seconds
of materialization. Placement-groups-only times out after 180.152 seconds.
For 177.465 seconds and 298 samples, every closed-wait predicate is true:

- all four logical GPUs have an observed owner;
- no logical GPU is available;
- GPU actor requests are pending; and
- the configured GPU ceiling has been reached.

The wait graph is concrete: one four-GPU shuffle placement group owns the whole
pool while `AddKey`, `SumGroup`, and terminal `Identity` actor creations each
wait for one GPU. Atomic acquisition prevents a partial shuffle gang, but it
does not decide *when* that gang may own the GPUs. The admission controller is
the layer that orders ownership across the Ray Data DAG.

Raw artifact:
`benchmark/results/local/final-liveness-full-vs-pg-g4-r4-v1`.

Stock Ray independently exhibits the original failure mode. In the paired
16-million-row four-GPU run, stock rank four sustains a structural closed wait
for 112.925 seconds, while stock rank one completes in 11.519 seconds and the
then-current candidate rank four completes in 15.989 seconds. This supporting
run predates the exact final wheel; it is useful evidence that rank tuning can
fix one small instance, not evidence that tuning is useless.

Raw artifact: `benchmark/results/local/stock-workaround-16m-20260722a`.

## Why the existing alternatives are not equivalent

### Autoscaling

Autoscaling supplies physical capacity; admission controls temporal ownership
inside one Ray Data DAG. At the configured ceiling, an autoscaler cannot add a
GPU to a closed wait. The four-GPU wait graph demonstrates exactly that state.

There is also a configuration coupling in current Ray: when
`gpu_shuffle_num_actors` is unset, GPU shuffle rank is derived from the
autoscaler's maximum configured resources when available. Raising the cluster
maximum can therefore raise the default gang size along with capacity instead
of creating guaranteed headroom.

The candidate composes with autoscaling rather than replacing it. An infeasible
atomic gang exposes its complete minimum demand and retains that pending demand
when new capacity arrives but an upstream owner has not handed off yet. The
UCXX setup timeout starts only after its placement group is ready so node-launch
latency is not misclassified as communicator failure. Exact focused tests cover
this demand path. A real G6 cluster is still required to measure EC2 launch
latency and multi-host behavior. One remaining operational limitation is that
an explicitly impossible rank without a finite execution limit follows normal
Ray infeasible-request semantics and may remain pending; the UCXX timeout does
not cover placement-group provisioning.

### User rank and pool tuning

Rank tuning is a valid workload-specific workaround. It is not a general
liveness policy:

- the small four-GPU case completes when stock rank is reduced from four to
  one;
- the corresponding four-GPU, one-billion-row rank-one run fails in cuDF sort
  with GPU out-of-memory because the lower rank concentrates the partition;
- a two-GPU actor-only chain enters a 169.684-second structural closed wait
  without any shuffle, so no shuffle-rank setting can address it (this
  supporting run also predates the exact final wheel); and
- downstream `map_groups` and `map_batches` actors can reserve GPUs before
  receiving usable input even when the shuffle rank itself fits.

The controller preserves the user's chosen rank and `ActorPoolStrategy`
minimum. Its purpose is to make those owners acquire and release resources in a
progress-safe order, not to guess one globally correct rank.

Supporting artifacts:

- `benchmark/results/local/perf4-1b-v1`
- `benchmark/results/local/local-multigpu-actor-baseline-20260722a`

### `materialize()`

Manual materialization can break a cycle by forcing phase boundaries, and it
must remain an available workaround. On the exact no-spill 16-GPU comparison,
however, stock plus two explicit intermediate boundaries is 4.7% slower than
automatic admission. It also requires application-specific DAG surgery, gives
up streaming overlap, and cannot be inserted between the internal shuffle and
`map_groups` portions of the grouped operation without rewriting that
operation.

Materialization is not universally slower. With only a 16 GiB object store, the
stock materialized control completes in 77.280 seconds while the then-current
streaming candidate takes 85.137 seconds and spills roughly 71 GiB. An
experiment that strictly deferred the shuffle to mimic an automatic boundary
was worse still at 173.849 seconds because it forced a large intermediate
through backpressure and spill. That policy was removed.

The defensible claim is therefore not “materialize is always worse.” Admission
automates safe ownership while retaining streaming, wins the normal-capacity
incident control, and avoids requiring users to discover and maintain manual
phase boundaries. Forced-spill behavior remains an optimization target.

Supporting artifacts:

- `benchmark/results/local/perf16-final-stock-materialize-r7-w0-v1`
- `benchmark/results/local/perf16-2b-16g-stock-mat-r7-w0-v1`
- `benchmark/results/local/perf16-2b-16g-min-r7-w0-v3`
- `benchmark/results/local/perf16-2b-16g-min-r7-w0-v4`

## Maintenance cost and verification

- Production scope: ten nonzero files under `python/ray/data`; no Ray Core
  change.
- Production delta: 810 added, 166 deleted, `+644` net NCLOC.
- API surface: no public API; one private rollback flag.
- Exact-wheel checks: commit, source tree, wheel, patch, and every changed
  production-file hash match the pins.
- Focused validation: 18 controller/autoscaling/backpressure tests and one
  logical two-GPU streaming integration pass.
- Spill paths are on `/raid` (`/dev/md127`), while short Ray sockets use
  `/dev/shm`; cleanup proves no owned actors, placement groups, Ray processes,
  or spill directories remain.

LOC artifact:
`benchmark/results/local/candidate-loc-final-31d444.json`.

## Minimum next evidence

Do not run the full acceptance matrix yet. The next useful sequence is:

1. Repeat only the four 16-GPU arms above twice in randomized order. Keep the
   candidate only if the stock, placement-group, and materialization gaps
   survive.
2. Run the same per-GPU data scale at 4, 8, and 16 GPUs for the candidate and
   best completing stock shape. This supplies an actual scaling curve rather
   than one large endpoint.
3. Repeat the equal-shape actor-only control on the final wheel to bound normal
   actor-pipeline regression and retain the no-shuffle liveness case.
4. Only after the policy is frozen, run the targeted failure/cleanup matrix and
   broader correctness gate.
5. Finally run one fixed and one autoscaling G6 campaign. Separate node-request
   to node-ready time from node-ready to first progress, and compare warm
   replay on the already-expanded cluster.

The cloud decision should answer one narrow question: after autoscaling has
provided the same maximum GPUs, does Ray Data admission still prevent idle
ownership and improve completion time? The DGX evidence says yes on one host;
the G6 run must test whether that survives real launch and network costs.
