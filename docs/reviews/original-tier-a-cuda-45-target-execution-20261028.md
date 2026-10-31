# Original Tier A GPU target science and full five-role native replay

Status: **implemented behind explicit external resource grants, not executed on GPU** (2026-10-28). This does **not** replace historical/private-source certification, nor mint an external attestation.

## Proven source and unchanged role identities

- Native original CPU reference: [45/45 run #40753588466](https://github.com/kys928/Hephaestus/actions/runs/40753588466), actual 96 AdamW updates, original ZIP archive #12845610807.
- Original Tier A readback: [run #40762010344](https://github.com/kys928/Hephaestus/actions/runs/40762010344), original source SHA `sha256:70e80d8f23f9ff47202b1c4778df81d2306fb2a9e8707553ea3ee6006f82c4e6`, exact chat/EOS training SHA `sha256:ee98849e7141cfdaba13eafc4647cc82707da82c8cdc9dd0ec88c7caac413b73`.
- Original frozen independent five-role CUDA models and original LoRA adapter keys: unchanged `configs/models/hephaestus_role_model_stack_v1.json`, `configs/models/hephaestus_role_mastery_runtime_v1.json`, `docs/evidence/role-model-stack-v1/certification.json`.
- The separately evaluated SmolLM2-135M target remains *not* an original certified Controller, Diagnosis, Planner, Evaluator or Judge.

## New real CUDA scientific owners

The new `CudaExactTokenTrainer` in `src/hephaestus/training/tier_a_cuda_target.py` trains the real pinned base on `cuda:0` with LoRA rank=8 alpha=16, AdamW lr=0.0005, fixed 96 sample indices and chat-prefix masked/EOS-supervised tokens. Every actual step saves a versioned immutable checkpoint (adapter, optimizer, CUDA+CPU RNG and content-addressed manifest) and an append-only original SQLite receipt; restart validates *every* prior step and sample-index prefix rather than trusting an isolated latest pointer. Any missing CUDA device is a hard failure, never a CPU fallback.

`CudaProspectiveEvaluator` executes both the original pinned baseline and actual finalized candidate on `cuda:0`, decoding with identical greedy chat64 rules and verifying strict valid JSON and exact typed correctness on the same original *already-known* 24-example generated evaluation. NLL remains a diagnostic. It verifies the candidate against the original final optimizer receipt and never derives expected hash values from mutable candidate bytes.

`CudaResearchTrainingRole`, `CudaResearchMonitorRole`, `CudaResearchEvaluatorRole`, `CudaResearchJudgeRole` and `CudaResearchEvidenceCollector` replace corresponding native scientific stage delegates *only when an explicit, separately authorized `CudaTargetScope` is supplied*. The original `OriginalCudaResearchObservation` wrappers independently run all five frozen role models on five actual native domain-observation stages. Before target training or generation, the role generator is explicitly unloaded to prevent two role/target models from occupying the same GPU.

`src/hephaestus/production/tier_a_cuda_campaign.py::run_or_resume_original_tier_a_cuda_campaign` is the opt-in native original 45-stage execution entrypoint. Its caller must inject the real original `scripts/run_cpu_research_campaign.py::load_context` and an independently scoped `StagedApprovalService`, a read-only original S3 adapter client, model/cache paths and a **persistent user-owned work directory**. It has no GPU provisioning permission, contains no bundled credentials, and does not create grant objects. The SDK caller/RunPod worker must allocate a GPU and bind all resources only after authentic owner budget approval, with explicit cancellation and quota enforcement.

The new `verify_native_tier_a_cuda_trace` requires an authentic contiguous native 45/45 campaign, the original 45 substep IDs, five real original pinned CUDA inference receipts with exact revisions/adapter hashes, 96 independent original CUDA optimizer receipts, finalized adapter SHA, unchanged authorized Tier A source and both baseline/candidate generated on CUDA under identical 24-case evaluation. A green CI/test matrix or a five-role *binding* inventory never counts as GPU execution.

## Approval separation

1. Original **Tier A data admission** may be automated only for the original generated task source, deterministic validator and exact permitted recipe. It never grants compute rights.
2. **GPU spending** requires a different externally trusted, per-operation `StagedApprovalService` and content-addressed, bound grant with workflow, run, source policy, purpose, `cuda:0`, operator identity and time ceiling. Unsupported/missing grants stop before model execution. This research approval is not a historical-source approval.
3. **Historical scientific certification** separately requires the actual four sealed reference datasets (train, dev, diagnostic, holdout), independent original-text contamination testing, Granite comparisons, external signature and human approval. No generated task, native 45/45, or GPU campaign alone grants these.

## Actual execution status and next deployment gates

As of this draft, all newly added CUDA services have **CPU-only structural and denial tests**. Real CUDA optimizer updates, real original-role inference and the new contiguous 45/45 GPU campaign **have not executed**. There is no fake 45-operation GPU record. Nothing has been merged, promoted, released, or self-certified.

Before a genuine launch: integrate a trusted RunPod worker and user-owned persistent volume, pin and prefetch the original five immutable HF model snapshots and S3 LoRA archives, pin compute quota/teardown and all per-operation grants, create a *new* 45-stage config with its own workflow/run/experiment IDs, and run the injected native entrypoint against the actual CUDA device. Audit all original SQLite/source/role/checkpoint receipts from the original persistent workdir, then perform an independent byte-for-byte remote readback and promote only public-safe evidence indexes to GitHub. A temporary GitHub Actions artifact alone is **not** permanent evidence custody.

Existing CPU-only baseline 45/45 stays frozen. No historical references or role adapters are silently substituted.
