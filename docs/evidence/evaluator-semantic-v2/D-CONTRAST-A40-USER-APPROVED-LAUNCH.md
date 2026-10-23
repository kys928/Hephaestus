# Approved A40 inference-only D contrast — cross-region S3 and one-use $1.90 ceiling

**Date:** 2026-10-23. Explicit user instruction: choose any genuinely available GPU with sufficient VRAM and spend up to roughly $2 credit to obtain frozen D evaluator scientific contrast results. **No training is authorized.**

## Choice and current limitation

The prior Blackwell Server Edition was a hardware-comparability control, not required by Granite 3.3 8B inference. A40 48GB has BF16 tensor cores and ample inference VRAM for a BF16 8B base and small LoRA, although Ampere BF16 outputs can differ from Blackwell and must be reported as **hardware-changed**. NVIDIA A40 in EU-SE-1 was verified by exact `gpuTypes(input:{id:"NVIDIA A40"})` one-GPU Secure Cloud GraphQL request in [read-only Actions #39973104184](https://github.com/kys928/Hephaestus/actions/runs/39973104184): `stockStatus=Low`, `uninterruptablePrice=$0.59/hour`. Subsequent [read-only Actions #39973815940](https://github.com/kys928/Hephaestus/actions/runs/39973815940) found its quote null/unavailable. **Availability is transient; no assertion of real-time stock beyond the specific checks.** Only a fresh positive A40 in-region quote can permit allocation.

[RunPod REST v2 per-datacenter inventory #39972878519](https://github.com/kys928/Hephaestus/actions/runs/39972878519) separately showed positive per-region stock for A40 EU-SE-1 and other GPUs; REST availability does not independently establish Secure Cloud inventory. Use exact legacy GraphQL one-GPU secure quote as a fail-closed allocation preflight.

The existing immutable network volume `cviwpryzao` is anchored to **EU-CZ-1** and cannot mount in EU-SE-1. The pod request therefore explicitly removes `networkVolumeId` and `volumeMountPath`, and pins compute to EU-SE-1. All source/checkpoint/evidence S3 GET/HEAD/PUT requests continue to use the existing **https://s3api-eu-cz-1.runpod.io/** endpoint and `RUNPOD_DATACENTER_ID=EU-CZ-1`, using the existing S3 key. Nothing moves or edits original frozen model/training datasets. Ephemeral pod disk is 180GB for dependencies, HF cache, and scratch. This **cross-region S3 path was statically/CPU-validated but not tested on a GPU pod yet**; failure means no model evidence, not a scientific failure.

## Scientific scope (unchanged)

- Frozen `ibm-granite/granite-3.3-8b-instruct` base revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`, plus trained D step-96 LoRA archive SHA-256 `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`; no training.
- 60 single-scientific-variable counterfactual pairs / 120 appearances deduplicated to **56 fixed greedy prompts**, with **48 expected scientific transitions and 12 invariants**. Identical sample/labels/protocol SHA, no sealed certification data. Max 112 new tokens/prompt.
- Durable append-only S3 timing receipts for cache lookup, Hugging Face 8B snapshot download, S3 adapter download/verify/extract, CUDA, tokenizer, BF16 weights, LoRA attach, and eight-output shard generations. The last-started stage is retained on timeout; no raw output/secret logging.
- Independent **CPU/S3 read-only** scientific rescore for exact-format validity, both-end pair correctness, 60 transition consistency metrics after full completion. No certification, training, model changes, production merge or autonomous second run.

## Exact spend and one-use boundary

- 1× NVIDIA A40 (48GB), **Secure Cloud EU-SE-1** only, BF16. Require measured model GPU free memory floor **40 GiB**, appropriate for an 8B BF16 model and this inference-only workload; no quantization or precision changes.
- Hard **$0.59/hour** actual pod rate. User funds estimated ~$2 (not independently verified). Limit this run to **$1.90 estimated GPU total** including a 120-second teardown reserve. At $0.59/hour, 10,800 seconds of launcher lifetime projects ~$1.77 and reserve ~$0.02; expected budget <=$1.90. **Actual billing may differ** and storage charges are separate.
- Launcher hard wall 10,800 seconds, runner hard wall 10,450 seconds, single pod, `create_with_capacity_retries(...,1,0)`, atomic S3 `IfNoneMatch="*"` approval claim before allocation. No allocation repetition after claim or create request. Teardown always attempts deletion and verifies absent, including errors.
- New distinct approval ID `heph-D-contrasts-A40-EUSE1-20261023-ONE-PAID-ATTEMPT`; the old attempt `heph-D-contrasts-20261019-capacity-retry-1` already consumed and cannot be reused. User authorized this new paid attempt and the $1.90 limit on Oct 23.
- CPU run [#39973815792](https://github.com/kys928/Hephaestus/actions/runs/39973815792): **64 passed**, paid=false dry-run, exact configured GPU without volume mount, no spend. A new exact read-only stock gate runs **both before dependencies and immediately before one-shot GPU request**.

## Approval and success conditions

The single-use Actions workflow `.github/workflows/evaluator-D-contrast-instrumented-approved-once.yml` triggers only on commit message marker `[explicit-launch-approved-D-instrumented-once]` on branch `experiment/D-contrast-flex-gpu-budget2-20261023`. It first checks real live A40 Secure Cloud stock, runs CPU science/approval/budget tests, rechecks stock, runs the once-claimed GPU inference, then independent S3 scientific evaluation; no GPU can be purchased if the regional stock guard fails. If unavailable, the job is a preallocation failed capacity check and can be rerun later if the single-use claim and pod request are proven untouched.

**Success requires** `contrast_inference_complete` terminal evidence, all 56 uniquely hashed outputs/60 paired rescore, and verified pod deletion under estimated cap. Merely starting the GPU or obtaining partial shards is not scientific success. No guarantee of availability or inference success is possible while real provider stock is Low/None or while Hugging Face snapshot download remains unmeasured on a live pod.
