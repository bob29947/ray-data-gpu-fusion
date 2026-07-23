# Local GPU and 16-GPU scale evidence

This harness runs the incident-derived workload on isolated single-host Ray
clusters with up to 16 physically visible GPUs. The machine currently exposes
16 NVIDIA V100 32-GiB GPUs; each case sees only its explicitly selected subset.
The `dgx-scale` profile is the primary strong- and weak-scaling experiment when
AWS is intentionally skipped.

The local result can prove the scheduling deadlock, admission liveness,
rank-one floor, output correctness, GPU assignment, and cleanup. It cannot
prove L4 performance, node-launch behavior, multi-host networking, or a real
cluster autoscaler's provisioning latency. Those remain separate claims rather
than prerequisites for the single-host scaling result.

## Safety model

- Planning is the default. Only `--execute-local` starts Ray or installs wheel
  overlays.
- Every wheel is installed with `pip --target` into the ignored run artifact;
  the project `.venv` is not changed.
- Every case owns a fresh local Ray cluster. Ray session files and short Unix
  socket paths stay under `/dev/shm/ray-admission`; actual object spilling is
  explicitly directed to that case's unique
  `benchmark/results/local/<run>/cases/<case>/ray-spill` directory on the
  `/raid`-backed workspace. The runner never calls `ray stop`, so it does not
  kill other users' or tasks' Ray processes.
- Wait graphs and job-level cleanup audits read core GCS tables directly; the
  dashboard remains disabled.
- A timed-out workload gets `SIGTERM` first so its structural wait snapshot is
  retained. The cluster stays up for an actor/placement-group audit, then its
  owning daemon shuts down and verifies its exact process IDs are gone.
- The spill path must be an absolute, previously absent `ray-spill` leaf in the
  exact case directory. Each execution records its backing device, filesystem
  type, mount, available bytes, and physical spill-file count/bytes. Only after
  all owned Ray processes stop does the runner remove that exact guarded leaf;
  a live scan race is explicitly labeled, while the stable post-shutdown scan
  must be complete. Cluster cleanup is not proven unless removal is verified.
- The campaign stops immediately if cleanup cannot be proven.

## Tests and plan

Run the harness unit tests first; these do not start Ray:

```bash
.venv/bin/python -m pytest -q benchmark/local/tests
```

The wait graph uses the pinned Ray core-GCS accessor and protobufs, so the
derived wheels do not need dashboard frontend assets or HTTP State API extras.

Render the core randomized matrix. This performs only read-only wheel hashing
and GPU inventory; it does not start a cluster:

```bash
.venv/bin/python benchmark/local/run_evidence.py \
  --run-id local-core-v1 \
  --arms stock,pg-only,minimal,prototype \
  --capacities 1,2,4 \
  --workloads incident,actor-only \
  --ranks all \
  --repetitions 1 \
  --gpu-indices 0,1,2,3 \
  --case-timeout-seconds 75
```

Inspect `benchmark/results/local/local-core-v1/plan.json`. The final candidate
and preserved prototype wheels are now distinct and their hashes are pinned.
For each derived wheel the plan records three separate provenance values:

- `source_commit`: the candidate or prototype change commit.
- `base_commit`: the stock Ray commit to which that change was applied.
- `installed_ray_commit`: the value embedded in the wheel and exposed as
  `ray.__commit__`; this must equal `base_commit`, not `source_commit`.

After the plan and artifacts are correct, repeat the command with
`--execute-local`. A small one-GPU pilot should precede that complete run:

```bash
.venv/bin/python benchmark/local/run_evidence.py \
  --run-id local-one-gpu-pilot-v1 \
  --arms stock,minimal \
  --capacities 1 \
  --workloads incident,actor-only \
  --ranks all \
  --repetitions 1 \
  --gpu-indices 0 \
  --rows 1000000 \
  --blocks 8 \
  --case-timeout-seconds 75 \
  --execute-local
```

The pilot is useful for debugging but will correctly remain
`not_ready_for_cloud`; the hard gate requires all four arms and the complete
1/2/4-GPU rank sweep. For timing distributions, rerun the complete matrix with
at least five repetitions and a new run ID. Case order is randomized from the
recorded `--seed`, and each case records the wheel/commit/hash, physical GPU
selection, preflight allocation, workload telemetry, correctness oracle,
timeout classification, job cleanup, and cluster process cleanup.

The focused future-fusion proxy can be run without the full matrix:

```bash
.venv/bin/python benchmark/local/run_evidence.py \
  --run-id aggregate-cpu-gap-g4-r4-v1 \
  --arms stock,minimal \
  --capacities 4 \
  --workloads aggregate-cpu-gap \
  --ranks 4 \
  --map-actors-per-stage 1 \
  --map-actors-max-per-stage 4 \
  --repetitions 1 \
  --gpu-indices 0,1,2,3 \
  --rows 16000000 \
  --blocks 32 \
  --case-timeout-seconds 180 \
  --execute-local
```

Its physical shape is GPU actor key creation, fused GPU hash
shuffle/aggregation, a CPU expression map, and a real downstream GPU `AddOne`
actor. The harness snapshots and validates that optimized operator sequence
before execution, and the exact-output oracle checks `2 * sum(id) + 1` for
every unique group key.

For the focused performance gate, `--profile scale` defaults to five randomized
repetition blocks at four billion rows and 1,024 blocks. The keyed dataset has a
64 GB logical lower bound before cuDF and shuffle overhead and exceeds the 8 GiB
object store, so this is the evidence profile rather than a smoke test. Override
both `--rows` and `--blocks` together only while iterating:

```bash
.venv/bin/python benchmark/local/run_evidence.py \
  --run-id local-scale-proof \
  --profile scale \
  --gpu-indices 0,1,2,3 \
  --execute-local
```

## 1/2/4/8/16-GPU DGX scaling

`--profile dgx-scale` owns a fixed factor matrix so a run cannot accidentally
omit a GPU count or silently call one hand-picked stock setting "best." The
default is a one-repetition, 26-case screening pilot. Each randomized
repetition block uses a fresh cluster for every case:

- Strong scaling fixes the input at 4 billion rows and 1,024 blocks. The
  candidate uses full map and shuffle parallelism at 1, 2, 4, 8, and 16 GPUs.
  Stock measures all 14 unique useful edges of its safe frontier
  `3 * map_actors + shuffle_ranks <= GPUs`: one shape at 4 GPUs, four at 8,
  and nine at 16. For every feasible map-pool size, the grid measures rank one
  and the largest safe shuffle rank, removes duplicates, and only then reports
  the fastest completing shape.
- Weak scaling runs the candidate at all five capacities with 1 billion rows
  and 256 blocks per GPU. This isolates whether admission continues to turn
  additional GPUs into proportional throughput after the paired strong-scaling
  cases establish the speedup over stock.
- One 16-GPU actor-only pair gives stock and candidate the same safe fixed
  five-actor shape. Its three lifetime GPU map pools reserve at most 15 GPUs,
  so both arms can complete. It is the no-shuffle regression control and must
  stay within 5%. A full-16 candidate actor chain, if run separately, is
  liveness/scaling evidence, not a completing stock regression baseline.

Stock's one- and two-GPU incident cases are deliberately not repeated here.
Even the irreducible positive shape needs four GPUs (`3 * 1 + 1`), and the core
liveness profile owns the one-GPU structural-timeout proof. Those capacities
cannot provide a completing stock performance baseline.

Render the 26-case screening plan while another task owns the GPUs:

```bash
.venv/bin/python benchmark/local/run_evidence.py \
  --run-id dgx-scale-plan \
  --profile dgx-scale \
  --gpu-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --num-cpus 64 \
  --case-timeout-seconds 7200
```

This is plan-only: it hashes artifacts and inventories GPUs but starts no Ray
cluster and runs no CUDA work. Inspect `plan.json` and
`dgx-scale-evidence-report.json`. A smaller pilot may cap total weak-scaling
work explicitly, for example `--dgx-weak-max-rows 4000000000`. The plan and
report mark that curve as bounded and record the actual rows per GPU, so it
cannot be mistaken for the full 1-billion-rows-per-GPU result.

The full-grid evidence run uses five randomized blocks, or 130 cases:

```bash
.venv/bin/python benchmark/local/run_evidence.py \
  --run-id dgx-scale-proof \
  --profile dgx-scale \
  --repetitions 5 \
  --gpu-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --num-cpus 64 \
  --case-timeout-seconds 7200 \
  --execute-local
```

The report selects the fastest measured completing stock shape independently at
4, 8, and 16 GPUs, then pairs that shape with the full-utilization candidate by
repetition. It emits median time reduction, throughput, strong-scaling speedup,
parallel efficiency, weak-scaling efficiency, cluster and owned GPU-seconds,
spill/restore bytes, content/schema hashes, and job/cluster cleanup proof. Its
merge gates require at least five repetitions, at least 10% candidate speedup
at 4/8/16 GPUs, no more than 5% actor-only regression, complete metrics and
correctness, the configured 8 GiB of object store per GPU in every execution,
and cleanup for all 130 cases. At least 64 logical CPUs are required so the
16-GPU cases are not accidentally CPU-throttled.

The one-repetition default is screening, not proof. It emits the measured
ranking and up to the top two stock shapes at each capacity (4 GPUs has only
one safe shape). Run focused generic-profile comparisons for those shapes with
at least five randomized repetitions. Give stock and the candidate the same
selected map-actor count and shuffle rank in an equal-tuned control; this
separates admission benefit from simply changing the resource shape. Match the
DGX run's 8 GiB-per-visible-GPU object-store setting in those controls. The
deterministic full-grid proof remains available when the extra run cost is
acceptable.

For the causal speedup claim, each GPU UDF records its constructor-ready and
first-input event times at the caller. The first positive-row input event is
acknowledged once per actor, a conservative equal-arm benchmark barrier. The
driver then requires a stable, quiescent tracker snapshot with exact fixed-pool
constructor registrations. The report treats constructor-ready-to-first-input
GPU-seconds as a measured lower bound and attributes it separately to
`SumGroup/map_groups` and final `Identity/map_batches`, including each stage's
fraction of cluster GPU-seconds and peak observed premature GPUs. Every counted
actor must also appear in-window as a restart-free GCS `ALIVE` owner with the
expected job, operator class, and GPU reservation. Sampled
premature-GPU overlap while an earlier stage is active is a separate metric
joined to GCS `ALIVE` state; it does not claim that a particular pending request
was displaced. On the single host, a shared boot ID and monotonic origin verify
one process-independent clock domain. Cross-host wall-clock events remain
unverified and cannot satisfy the local causal-evidence gate.

Every plan stages a read-only, content-addressed copy of all Python harness
inputs. Each case verifies those hashes before starting, and any invocation
using a run ID whose execution has started is rejected. This prevents a
multi-day randomized campaign from mixing benchmark implementations.

## Ray Data logical-autoscaling simulation

The optional simulation runs the candidate admission path itself. It starts a
Ray Data actor map with `ActorPoolStrategy(min_size=2, max_size=2)` and an
explicit two-GPU execution ceiling while only one logical-GPU raylet exists.
It proves that two actor requests are visible, at least one is pending, the
single logical GPU is owned, and no map batch completes before adding a second
logical-GPU raylet. It then measures resource-request-to-node-visible,
node-visible-to-first-progress, node-visible-to-completion, and a warm replay
on the same expanded cluster:

```bash
.venv/bin/python benchmark/local/simulate_autoscaling.py \
  --result benchmark/results/local/autoscaling-simulation-plan.json

.venv/bin/python benchmark/local/simulate_autoscaling.py \
  --result benchmark/results/local/autoscaling-simulation.json \
  --execute-local
```

Execution installs the pinned candidate wheel into an isolated temporary
overlay and uses Ray's multi-raylet test cluster without the unsafe global
`ray stop` used by `AutoscalingCluster`. The overlay, owned processes, and the
short `/dev/shm/ray-admission` directory are audited and removed. State comes
directly from Ray's core GCS tables; no dashboard or HTTP State API is needed.

All logical raylets share this one host and the UDF does no CUDA work. The
result tests scheduler demand, candidate admission, and reaction to changing
logical capacity only. It must not be used as EC2 provisioning, multi-host,
or GPU-throughput evidence; those claims require the G6 cloud experiments.
