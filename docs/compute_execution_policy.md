# Compute Execution Policy

Hephaestus treats paid compute as a governed experimental resource. Compute selection must minimize cost while remaining sufficient for the frozen scientific workload.

## Run-level immutability

Before a paid run starts, its compute specification is frozen as part of that run's execution manifest.

The frozen specification includes at minimum:

- provider and datacenter;
- pod count and parallelism;
- exact GPU model;
- GPU VRAM;
- image/runtime;
- hourly price ceiling;
- per-pod and whole-run spend ceilings;
- hard wall-clock limits.

Retries of the same scientific run must reuse that compute specification. Hephaestus must not silently substitute a different GPU model, increase VRAM, increase pod count, widen a GPU allowlist, increase parallelism, or raise cost ceilings merely because capacity is unavailable or a previous attempt failed.

## Compute changes require a new recorded run decision

Compute may change only when evidence shows the frozen compute specification is insufficient for the workload, for example verified out-of-memory failure, unsupported precision/runtime, or a measured throughput constraint that prevents scientifically valid completion within the frozen wall clock.

A compute change must:

1. state the evidence requiring the change;
2. create a new execution/run configuration rather than mutating the existing run identity;
3. record the old and new compute specifications and the reason for the change;
4. preserve scientific inputs, seeds, datasets, gates, and evaluation boundaries unless they are independently authorized to change;
5. prefer the smallest sufficient increase in compute.

Capacity scarcity or temporary provider unavailability is not, by itself, evidence that more powerful hardware is required. In that case Hephaestus should retry or wait for the same frozen hardware rather than silently upgrading.

## Cost minimization

Hephaestus should select the cheapest hardware that is demonstrably sufficient for the workload. More expensive GPUs are not preferred merely because funds are available.

Paid runs should avoid unnecessary parallelism. Parallel execution is justified only when its time savings are worth the additional simultaneous burn rate and remain inside the frozen experiment budget.

The whole-run budget is authoritative. Per-pod limits do not substitute for a shared experiment-level spending cap.

## Required launch report

Whenever paid pods are created, the operator report must include for every pod:

- role or workload;
- pod ID;
- exact GPU model;
- VRAM;
- hourly price;
- creation time;
- termination time when known;
- elapsed runtime;
- estimated or actual cost.

The report must also show the combined hourly burn rate and the experiment-level projected maximum spend.

If GPU identity, VRAM, or hourly price cannot be determined, that missing evidence must be made explicit rather than presenting the launch as fully characterized.

## No silent budget expansion

Restored or increased account funds do not authorize a larger experiment budget. Budget ceilings and compute specifications remain frozen until a separate explicit run decision changes them.
