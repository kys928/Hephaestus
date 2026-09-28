# Interface Repair V2 execution path

Interface Repair V2 preserves the frozen Interface Repair V1 scientific protocol and changes only execution mechanics.

## Frozen scientific inputs

- Same 7,040-case pack and canonical SHA-256.
- Same train/certification/regression partitions.
- Same parent role-mastery stack.
- Same LoRA rank/alpha/dropout, optimizer, learning rate, seed, gradient accumulation and optimizer-step counts.
- Same role decoding envelopes and certification/live-certification thresholds.
- Diagnosis remains frozen. Production promotion and automatic role dispatch remain disabled.

## Compute-efficiency changes

1. **Dynamic-length training**: microbatch size remains one, but each example is processed at its real token length instead of being padded to 2,048 tokens. The 2,048 value remains a hard maximum-length validity check.
2. **Contract-aware early stopping**: generation keeps the original maximum-new-token ceilings, but stops once an exact seven-key contract JSON object is complete. This avoids paying for text after a valid contract without narrowing the original decoding envelope.
3. **Sharded evidence persistence**: evaluation samples are persisted as 32-case JSONL shards. Writes are verified with object length plus SHA-256 metadata instead of a synchronous PUT followed by a full GET for every sample.
4. **Immutable baseline cache**: pre-training interface/regression summaries are keyed by pack hash, role, exact model revision, parent-adapter hash, seed, decoding configuration and partition geometry. They may be reused only on an exact identity match.
5. **Persistent network-volume cache**: the RunPod network volume is mounted at `/workspace`; Hugging Face caches persist under `/workspace/hephaestus-cache` across paid runs.
6. **One-pod full stack**: planner, evaluator, judge, controller and live certification execute sequentially on one A40 pod. Each role runs in a fresh Python process, so CUDA/model memory is released between roles while image/bootstrap cost is paid only once.
7. **Stage telemetry**: model materialization, model load, baseline, training, adapter persistence and post-certification timings are persisted with runtime/VRAM evidence.

## Paid preflight gate

Preflight is a separate paid authorization boundary and cannot automatically launch the full experiment.

Frozen preflight shape:

- GPU: exactly one NVIDIA A40, no substitution.
- Hard wall: 600 seconds.
- Spend ceiling: $0.10.
- Role: planner (the observed bottleneck role).
- Evaluation sample: 8 certification cases distributed across the partition.
- Training sample: 5 optimizer steps = 40 examples at accumulation 8.
- It measures evaluation seconds/case, generated-token length, complete-contract rate, early-stop rate, training seconds/optimizer-step, peak training VRAM and projected full planner-role runtime.
- It rejects the full-run recommendation if projected planner runtime exceeds 5,400 seconds, any generation hits the deadline, or complete-contract JSON rate is below 100%.

A full paid launch additionally requires the successful preflight `result.json` S3 key and the exact repository SHA to be written into `interface_repair_v2.launch.json`. The full launcher reads that evidence before creating a pod and refuses to launch if the result is absent, rejected or belongs to another commit.

## Current safety state

Both `paid_preflight_allowed` and `paid_full_launch_allowed` are false. Both launch-marker authorizations are false. No paid V2 pod can be intentionally created until the operator explicitly authorizes the corresponding step.
