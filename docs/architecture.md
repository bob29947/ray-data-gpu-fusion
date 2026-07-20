# Architecture

`ray-data-gpu-fusion` is an optimizer and execution-backend extension for Ray
Data. It is not a Dataset interpreter and it does not run a scheduler beside
Ray.

## Where it hooks into Ray

```text
ordinary Ray Dataset API
        |
        v
Ray logical optimization and physical planning
        |
        v
plan-local rules installed by rgf.enable()
  1. lower eligible physical nodes to closed GPU candidates
  2. fuse compatible linear candidates opportunistically
        |
        v
Ray StreamingExecutor
        |
        v
Ray Core actor scheduling, ObjectRefs, resources, retries, metrics, shutdown
```

Calling `rgf.enable()` adds two importable rule classes to the current
`DataContext`. A Dataset captures that context when it is built. When a normal
action such as `materialize()`, `count()`, or `iter_batches()` asks Ray for an
execution plan, Ray invokes the rules as part of its own physical optimizer.
The resulting operators are ordinary Ray physical operators and are consumed
by the stock `StreamingExecutor`.

There is no plugin-owned action wrapper, graph runner, queue, resource manager,
or actor scheduler.

## Closed nodes before fusion

Lowering first makes every accepted node independently executable. Phase 0 has
two forms:

- A standalone eligible `MapBatches` is a Ray `ActorPoolMapOperator` using the
  already-planned stock Ray transformer. Its candidate also carries a native
  `TransformSpec` that can participate in fusion.
- A standalone eligible Parquet read is a Ray `ActorPoolMapOperator` around a
  closed plugin program: Ray-block work descriptors enter the actor, exact
  row groups are read with cuDF, and Arrow blocks leave the actor.

Turning fusion off with `rgf.enable(fusion=False)` therefore leaves a valid
Ray plan. Fusion is an optimization, never the condition that makes a node
runnable.

## Fusion

Each candidate declares:

- input and output payload kinds;
- a runtime transform key and immutable configuration;
- execution requirements, including actor-pool shape and Ray resources;
- required and provided data properties.

The fusion rule only folds a linear edge when payloads, properties, retry
semantics, actor-pool shape, resources, placement, runtime environment, and
backend identity compose exactly. A refusal leaves both original executable
nodes in the plan.

For a compatible `read_parquet -> map_batches -> map_batches` chain, Ray sees
one physical actor-pool operator. Inside one actor task the plugin runs:

```text
Ray work-descriptor block
  -> Parquet work import
  -> cuDF exact-row-group read
  -> cuDF batch UDF
  -> cuDF batch UDF
  -> Arrow block export
```

The intermediate frames stay on the GPU only inside this fused region.

## Two GPU regions that cannot fuse

Phase 0 deliberately uses an Arrow-backed Ray ObjectRef boundary:

```text
GPU region A -> Arrow Ray blocks -> GPU region B
```

The regions have separate Ray actor pools. The resource manager admits eligible
static-resource GPU pools in topological order against
the operator-reservation allocation. Every admitted pool receives at least a
one-actor floor and may scale only within its allocation. The first pool whose
floor does not fit is the frontier and may retain one queued actor request;
later pools are blocked so they cannot leapfrog it. Dormant, completed, and
blocked pools cancel pending actors and release idle actors, never active work.
Ray Core still decides actor placement, and independent stages can stream
concurrently whenever their floors fit.

This trades extra device/host conversion and Object Store pressure for a
simple ownership boundary and deadlock-free Phase-0 resource behavior. An
opaque device-resident boundary is reserved for a later phase.

If both actor floors fit, the two regions may overlap and stream through the
Arrow boundary. If the downstream floor does not fit, it waits at the frontier
and queued Arrow bundles can increase Object Store pressure while upstream work
drains. Each actor task also materializes all Arrow output before its first
yield so I/O, CUDA, and UDF failures are reported atomically through that task.
Object Store capacity and per-task host memory therefore remain explicit
benchmark dimensions for this prototype.

## Fallback and failures

- A recognition, compatibility, or materialization refusal during planning
  keeps the original stock Ray physical node.
- A fusion refusal keeps the independently executable GPU nodes.
- Once execution begins, I/O, CUDA, actor, and user-code failures follow Ray's
  normal retry and failure path. The plugin does not replay work on a second
  implementation.

`rgf.explain(dataset)` constructs the same optimized physical plan Ray would
execute and includes the selected standalone/fused regions and stable planning
refusal reasons.

## Ray candidate and local hooks

The pinned stock checkout is immutable. Derived Ray is split into two auditable
layers:

1. C is the standalone PR candidate for generic resource admission. It exposes
   capability version 1 and an internal `DataContext` rollback field. Today it
   adapts statically declared GPU actor pools and atomic GPU shuffle/hash-
   aggregate gangs. Safety floors remain enabled without proportional operator
   reservation, minimum-actor readiness is asynchronous after admission, and a
   user-supplied dynamic `ray_remote_args_fn` retains legacy scheduling with a
   warning; C contains no plugin dependency.
2. H1 adds plan-local physical optimizer rule classes on `DataContext`.
3. H2 adds a conservative Parquet external-scan descriptor that performs no
   I/O.

C is built independently as `wheels/pr-candidate`. H1/H2 are then applied to C
to build `wheels/hooked`, the only Ray wheel installed by bootstrap. The build
also derives stock+C+H1+H2 directly and requires byte-for-byte equality with the
layered hooked wheel. None of the three changes contains cuDF, CUDA, plugin, or
GPU-fusion imports. Everything specific to recognition, composition, GPU
execution, and future API adapters lives in the plugin distribution.

## Adding future Ray Data APIs

A new API adapter should do four things without changing the fusion engine:

1. recognize one exact logical/physical shape and decline unsupported options;
2. emit a native `TransformSpec` with payload and property contracts;
3. register an actor-local runtime for that transform key;
4. provide a standalone closed materialization path before enabling fusion.

Scalar expressions, encoders, and preprocessors can therefore share the same
composition and execution machinery. APIs that require exchanges, grouping,
ordering, multiple inputs, or a new ownership boundary add explicit payloads
and properties rather than special cases to the existing MapBatches adapter.
