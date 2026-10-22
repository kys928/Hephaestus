# D contrast: timed-out model-load audit and instrumented paid-run proposal

**Status as of 2026-10-22: PREPARED, NOT AUTHORIZED, NOT LAUNCHED.** No GPU claim or paid allocation has been made on this branch. Keep this PR in **draft**; do not merge.

## 1. Evidence and root-cause boundary

Historical pod `lw476giazga8e4`, [Actions #39299580607](https://github.com/kys928/Hephaestus/actions/runs/39299580607), GPU NVIDIA RTX PRO 6000 Blackwell Server Edition in EU-CZ-1. Estimated GPU charge **$0.6501000356**, verified pod deletion, 939.9 s including teardown. Workflow failed at `launcher_hard_wall_seconds=930`.

[Read-only forensic Actions #39815121442](https://github.com/kys928/Hephaestus/actions/runs/39815121442) found last persisted bootstrap stage `runner_started` at 2026-10-19 18:06:42 UTC, **73.5 seconds** after pod creation. No later post-load manifest, no inference progress, no saved shards, no terminal result. No durable model generations can be recovered. It does **not** prove that no transient model generation happened; the first unfinished shard (up to seven completions) was not persisted.

Code-path audit: `run_evaluator_contrast_probe.py` looks up frozen D source evidence, executes `prepare("D")`, resolves the base via `materialize_parent_cached`, calls `runtime.load_model`, freezes/evaluates the LoRA model, writes the **post-load** scientific manifest, and only then starts 56 generations. Model materialization first invokes `snapshot_download(..., local_files_only=True)`, then falls back on any failure to downloading a pinned 8B model into **ephemeral** Hugging Face cache. Frozen step-96 LoRA comes from a checked archive in persistent cache or S3 download into ephemeral disk; then HF tokenizer, base BF16 weights, and PEFT LoRA are loaded on GPU. Earlier S3 evidence cannot separate those blocking stages.

[Read-only source/cache Actions #39834590189](https://github.com/kys928/Hephaestus/actions/runs/39834590189) confirmed historical D source training complete, its step-96 adapter SHA256 and exact archived size (**61,535,036 bytes**). **No objects were listed under two expected volume cache prefixes** for pinned base snapshot and adapter archive. That increases concern about cold-loading but is **not definitive**, because S3 listing is not equivalent to an on-pod Hugging Face local-only snapshot lookup (POSIX symlinks, cache layout and mount behavior).

## 2. Instrumentation implemented (no scientific change)

- New `scripts/evaluator_D_durable_timing.py`: emits UTC + monotonic elapsed timing checkpoints to S3 `<run>/evaluation/timing/events/NNNN.json` and `timing/latest.json`, with a bounded metadata vocabulary and readback checks. The last **started** stage remains visible if it hangs or is terminated.
- Optional observer in `materialize_parent_cached`: local HF lookup start/cache hit/miss; pinned ephemeral base download start/end; config verified; persistent adapter probe, S3 adapter download, byte/SHA verification and extraction start/end.
- Optional observer in `run_interface_repair_v1.load_model`: CUDA validation, tokenizer load, BF16 base weights, PEFT LoRA attachment and readiness start/end. Uninstrumented callers retain their old signature behavior.
- `run_evaluator_contrast_probe.py`: stages cover frozen source fetch and check, curriculum preparation, full model materialization/load, post-load manifest write, first token generation, each 8-record S3 shard, and terminal result. Source SHA and scientific identities are unchanged.
- Unknown event types/metadata, missing S3 evidence persistence, and approval violations fail closed. Timing records contain **no raw prompts, labels, generated text, tokens, secrets or weights**.

**Important:** The S3 timing event writes occur during a future approved pod and add some unmeasured runtime overhead; the current CPU tests prove API shape and persistence behavior under mocks but **not** GPU download or load throughput.

## 3. Exact proposed next run: NOT TRAINING

| Area | Frozen or proposed value |
|---|---|
| Operation | **Inference-only diagnostic**, no training or adapter updates |
| Base model | `ibm-granite/granite-3.3-8b-instruct` (~8B) |
| Base revision | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |
| Ancestor LoRA adapter SHA256 | `913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed` |
| Frozen D step-96 adapter SHA256 | `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1` |
| Historical model training lineage | `evaluator-semantic-v2-D-38945578442`, `ffa0458d5850f9d572a06d5ee6660fd63fc8acc6` |
| Historical D training | 768 selected cases, six ordered 128-case blocks, 96 AdamW optimizer steps, LR 2e-6, microbatch 1 × gradient accumulation 8, BF16, LoRA rank 16 / alpha 32, max length 2048. **Already completed; not part of next run.** |
| **New** diagnostic data | 60 single-variable matched counterfactual pairs (120 appearances), reduced to **56 unique prompts** for fixed greedy decoding |
| Pair content | **48** expected state transitions, **12** precedence-invariant pairs, throughput delta a controlled secondary distractor; six scientific factors, eight scientific decision states |
| Prompt data policy | CPU-authored frozen contrasts, no answer labels/expected states in model-visible prompts, no sealed test/certification data |
| Decoding | Greedy `do_sample=False`, max 112 new tokens per unique prompt, no training |
| Metrics | Strict JSON versus exploratory parsing, correct two-end state pairs, pair-transition consistency; independent CPU S3 read-only rescore after complete result |
| Hardware | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, Secure Cloud, **EU-CZ-1**, network volume `cviwpryzao`, no fallback |
| Rate limit | **$2.49/hour** maximum |
| Proposed cost limit | **$0.75 estimated total**, including 120-second teardown reserve (planning value only; **not approved**) |
| Time limits | 850 s runner, 930 s launcher; automatic verified pod teardown |
| S3 evidence | A new run ID, manifest, progress, shard samples, independently scored result, immutable-by-run timing record |

## 4. Prelaunch gates and concrete evidence

- [CPU-only Actions #39834590196](https://github.com/kys928/Hephaestus/actions/runs/39834590196): **64 passed**, compilation, immutable oracle/data/protocol tests, timing receipt tests, dry-run pod request. Log explicitly reports `actual_allocation=false`, `paid=false`, `gpu_spend_authorized_now=0`.
- [Read-only S3 Actions #39834590189](https://github.com/kys928/Hephaestus/actions/runs/39834590189): D saved training `training_complete`; source step-96 SHA and size verified via S3 GET/HEAD; cache prefixes not visible; GPU allocations 0.
- Future `.github/workflows/evaluator-D-contrast-instrumented-approved-once.yml`: remains **SKIPPED** until an **explicit launch-marker commit** `[explicit-launch-approved-D-instrumented-once]`. Even with a marker, execution code will refuse a pod until a fresh distinct approval ID is marked `authorized_once`, `additional_allocations_authorized=true` and `runpod_launch_enabled=true`. Current values remain pending/false/false.
- Prior approval `heph-D-contrasts-20261019-capacity-retry-1` was **consumed**; this proposal uses a distinct **unapproved** ID `heph-D-contrasts-instrumented-20261022-PROPOSED`, no S3 claim.

## 5. Unresolved risk and user decision

**No guarantee** that all 56 generations fit within $0.75. The previous attempt cost about $0.65 merely to reach timeout with no post-load manifest. If cache is still cold, the next inference-only job may consume the proposed cap obtaining only load-stage timing. Its scientific value would then be root-cause diagnosis, *not* a full paired score.

Options requiring new user approval: (A) keep $0.75 and accept a **diagnostic-first** run with possible incomplete inference, or (B) explicitly authorize a separately budgeted cache-warm/full-inference plan based on the risks. No larger ceiling is assumed or configured, and no pod will be requested from this preparation alone.

The run is **staged but not launched**. Do not merge, resume the consumed approval, bypass cache provenance, use sealed evidence, train, or certify.
