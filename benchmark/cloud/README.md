# Direct-AWS GPU admission evidence harness

This directory contains the reproducible cloud experiment, not production Ray
or plugin code. The launcher is dry-run-only unless `--execute` is present. A
normal invocation renders `plan.json` and one cluster YAML per case without
contacting AWS.

## One-time local configuration

Copy `local-config.example.json` to the ignored `local-config.json` and fill in
the AMI, subnet, security-group, and IAM instance-profile references. The AMI
must support NVIDIA G6 instances and the tracked CPython 3.11 environment.
Static AWS access keys, session tokens, passwords, and private-key contents are
rejected. Authentication comes from a named `aws_profile` or the ambient AWS
provider chain. An optional `ssh_private_key_path` is a path only; its contents
are never placed in a plan or tracked file.

Before an executed run, install the pinned launcher-side AWS provider packages
into the Ray CLI environment:

```bash
.venv/bin/python -m pip install -r benchmark/cloud/aws-runtime.requirements.txt
```

The rendered configs use the checked-in RAPIDS 25.12 explicit environment,
IMDSv2, encrypted gp3 root volumes with delete-on-termination, local NVMe for
Ray temporary/spill data where the instance provides it (with an isolated root
volume directory on an EBS-only CPU head), and unique run/arm/topology tags on
both instances and volumes. Structural wait graphs and post-job cleanup audits read
the actor, task-event, and placement-group tables directly from core GCS. The
dashboard is disabled and no HTTP State API endpoint is exposed.

The repository file mount excludes local Ray source checkouts, all-wheel
directories, worktrees, caches, and prior benchmark results. The selected
stock or candidate wheel is synchronized through its separate wheel mount, so
fresh-cluster setup does not copy unrelated build artifacts to every node.

## Experiment factors

The four arms isolate the scheduling change:

- `stock`: unmodified stock Ray wheel.
- `pg-only`: final minimal candidate wheel with DAG admission disabled. This
  preserves its placement-group behavior and answers whether atomic placement
  alone is sufficient.
- `minimal`: the same final candidate wheel with admission enabled.
- `prototype`: the separately preserved full `5b9ac4` prototype wheel with
  admission enabled. It is intentionally not the GPU-fusion plugin.

Preflight requires the final candidate grant schema (`max_units`,
`may_submit`) for `pg-only`/`minimal`, the legacy `max_units`-only schema for
the preserved prototype, and no admission module for stock. The exact installed
Ray commit, grant fields, manifest source commit, and wheel digest are recorded.

The full-matrix topologies are fixed 1, 2, and 4 GPU clusters; a GPU-head
autoscaler from 1 through 4 GPUs; and a CPU-head cold autoscaler from 0 through
4 GPUs. Shuffle ranks default to 1, 2, and 4, with ranks above topology capacity
omitted. This directly tests the claim that lowering shuffle rank count is an
adequate workaround.

Available workload shapes are `incident`, `actor-only`, `map-heavy`,
`shuffle-heavy`, `forced-spill`, and `fan-in`. The incident is the physical
owner chain from the candidate acceptance test: GPU actor pool, complete GPU
shuffle gang, GPU `map_groups` actor pool, and a final GPU actor pool. Its
aggregate demand exceeds the fleet while each minimum progress floor fits.

The defaults deliberately render the complete matrix and five repetitions;
that is a large and expensive campaign. Inspect the case count and stage small
subsets before executing, for example:

```bash
.venv/bin/python benchmark/cloud/run_evidence.py \
  --run-id admission-pilot \
  --arms stock,pg-only,minimal,prototype \
  --topologies fixed-1 \
  --workloads incident \
  --ranks 1 \
  --repetitions 1
```

The preserved prototype manifest and wheel may be pending while the candidate
is rebuilt. Plan-only mode records missing evidence artifacts; execution
requires every selected manifest and wheel to exist and match its SHA-256.

After inspecting the rendered plan, repeat the exact command with `--execute`.
That flag performs an authenticated, read-only AWS preflight before the first
launch. Clusters are run one at a time and every arm receives a fresh cluster.

## Focused scale proof

`--profile scale` is the opt-in merge-evidence campaign. It still only renders
a plan unless `--execute` is also present. It owns the arm, topology, workload,
and rank factors so a paid run cannot accidentally omit one side of a gate. Its
default input is four billion rows in 1,024 blocks. Override `--rows` and
`--blocks` together only for a deliberate smaller smoke run.

```bash
.venv/bin/python benchmark/cloud/run_evidence.py \
  --profile scale \
  --run-id admission-scale-proof \
  --repetitions 5
```

Each repetition is one randomized block of nine fresh-cluster cases:

- On identical fixed four-node `g6.4xlarge` clusters, stock Ray runs the best
  completing workaround (fixed map pool size 1 and shuffle rank 1), while the
  candidate uses four-way map and shuffle parallelism. The paired median time
  reduction must be at least 10%.
- Stock and candidate run the same fixed-size four-actor `actor-only` shape.
  Candidate regression must be at most 5%.
- Candidate fixed-4 and autoscale-1-4 cases run explicit rank 4 with an elastic
  actor pool (`min_size=1,max_size=4`). This permits useful work while workers
  launch. Fixed-size pool cases remain as semantics and startup controls.
- Stock gets an `autoscale-1-7` control with the same elastic map pool and rank
  4. Seven GPUs cover four eager shuffle actors plus three map-pool minimums.
  The report records that 1.75x capacity ceiling, observed peak instances,
  instance-seconds, GPU-seconds, and elapsed time beside the candidate's
  1-to-4 result.
- Candidate autoscale-1-4 also runs the default shuffle rank, showing whether
  raising the autoscaling ceiling expands the planned shuffle gang.

Fixed pools use `ActorPoolStrategy(size=N)`. Elastic pools use
`ActorPoolStrategy(min_size=N,max_size=M)`; the harness does not silently turn
one contract into the other. Scale execution is restricted to `g6.4xlarge`, so
each GPU slot is one identical L4 node.

The focused report uses exact repetition pairing from
`benchmark/evidence_statistics.py`. It fails unless both performance gates
have at least five completed pairs, all correctness oracles match, every
control completes, and every teardown is proven. A one-repetition pilot is
useful operationally but cannot be merge evidence.

The default large-data qualification represents at least 32 GB of source `id`
values and 64 GB after the `key` column is added, before cuDF, Python, shuffle,
and spill overhead. Each node has an 8 GiB Ray object store, so this shape is
materially larger than the fixed-four cluster's aggregate object store and is
designed to exercise local-NVMe spilling. Render the one-repetition plan first:

```bash
.venv/bin/python benchmark/cloud/run_evidence.py \
  --profile scale \
  --run-id admission-scale-4b-pilot \
  --repetitions 1
```

The five-repetition merge campaign is 45 fresh clusters: per repetition, five
fixed-4 cases, three autoscale-1-4 cases, and one autoscale-1-7 case. Run the
nine-case pilot first and use its slowest successful duration to choose an
explicit `--workload-timeout-seconds` for the final campaign. The default is
1,800 seconds. That timeout covers remote driver startup, materialization,
correctness validation, and the final checkpoint—not just the timed
materialization.

The dry-run summary and `plan.json` report price-neutral capacity bounds. At
the default 1,800-second timeout, a five-repetition campaign has 60 GPU-node
hours at topology minimums and 97.5 GPU-node hours at topology ceilings if
every workload consumes its full timeout. These are safety bounds, not a cost
forecast: cluster setup, preflight, and cleanup add time, while completing
early and gradual autoscaling reduce it. Multiply observed instance-seconds by
the selected region's current `g6.4xlarge` rate after the pilot.

## Evidence and cleanup

Each workload continuously checkpoints cluster capacity plus Ray actor, task,
and placement-group state counts. A timed-out run therefore retains the stable
closed-wait tail needed to distinguish a scheduling wait from a crash. Normal
results include elapsed time, throughput, correctness oracle/digest, Ray Data
stats, object spilling, GPU-seconds, availability, scale-up timing, and the
topology/rank/arm identity.

Checkpoint records retain full state counts but embed only nonterminal GPU
owners and requests. This preserves the closed-wait graph and GPU ownership
timeline without copying thousands of unrelated CPU tasks into every sample.
The plan reports the maximum number of samples implied by the timeout and
sampling interval; final artifact size remains data dependent.

Autoscaling results put EC2 observations and Ray progress on one timeline. The
harness records workload command request, remote driver start, materialization
demand start, EC2 pending/running transitions, first and peak Ray GPU
visibility, and first useful stage progress. Signed node-ready-to-progress
intervals show whether work overlapped worker launch; instance-seconds and
GPU-seconds expose the cost of solving an eager minimum by adding nodes. EC2
instance-seconds are poll-derived estimates, and the raw timeline records the
poll interval and observations needed to bound that uncertainty.

Cleanup always runs from `finally`: bounded `ray down`, an exact-tag EC2
termination backstop, three consecutive empty instance polls, then three
consecutive empty tagged-EBS polls. Returned resources are checked against the
project, run, arm, and cluster tags before they are accepted as teardown
targets. A cleanup verification failure fails the campaign and is recorded in
`teardown.json`.

Run the local safety suite with:

```bash
.venv/bin/python -m pytest -q benchmark/cloud/tests/test_run_evidence.py
```
