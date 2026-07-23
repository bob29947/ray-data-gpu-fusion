# GPU admission reviewer artifacts

This directory contains the compact evidence bundle referenced by
[`docs/dgx-admission-evidence.md`](../../docs/dgx-admission-evidence.md).

## Files

- `dgx-screening-v1.tar.gz` contains 62 original, unmodified JSON artifacts
  from the selected DGX screening runs.
- `candidate-loc-final-31d444.json` is the directly browsable production-NCLOC
  audit for Ray candidate `31d4448482c1a47e034e9160d1d8f41e84cd1a25`.
- `scale-evidence-report.json` is the directly browsable four-GPU scale and
  rank-control summary.

Archive SHA-256:

```text
9f12f53db1abe78c6b200b22e399c583f5c8599e19db498a44f38a203171725b
```

Each archived experiment includes its rendered `plan.json`, prerequisite
report, staged-harness manifest when available, and per-case `execution.json`.
An `execution.json` embeds the workload result and resource/ownership samples,
along with the executed commands, preflight result, and cleanup result.

## Evidence map

| Claim | Archived experiment |
| --- | --- |
| Four-GPU incident liveness; placement groups alone | `final-liveness-full-vs-pg-g4-r4-v1` |
| Four-GPU native aggregate liveness | `aggregate-cpu-gap-g4-r4-v1` |
| Native aggregate placement-group-only control | `aggregate-cpu-gap-pg-g4-r4-v1` |
| Four-GPU materialization control | `agg-gap-stock-mat-g4-r4-v1` |
| 16-GPU incident performance: admission vs placement groups | `perf16-final-full-vs-pg-r7-w0-v1` |
| 16-GPU incident stock control | `perf16-final-stock-r7-w0-v1` |
| 16-GPU incident materialization control | `perf16-final-stock-materialize-r7-w0-v1` |
| 16-GPU native aggregate: stock, placement groups, admission | `agg-gap-perf16-r7-v1` |
| 16-GPU native aggregate materialization control | `agg-gap-stock-mat16-r7-v1` |
| Small rank-tuning control | `stock-workaround-16m-20260722a` |
| Billion-row rank/OOM and safe actor-only controls | `perf4-1b-v1` |
| Shuffle-free actor lifecycle control | `local-multigpu-actor-baseline-20260722a` |

Extract without modifying the repository:

```bash
mkdir -p /tmp/gpu-admission-review
tar -xzf benchmark/review_artifacts/dgx-screening-v1.tar.gz \
  -C /tmp/gpu-admission-review
```

These are the screening artifacts behind the current report. Some runs used
predecessor candidate wheels, as recorded in their manifests and in the report.
They demonstrate the mechanism and motivate the PR, but do not replace the
frozen final-wheel merge-gate refresh.
