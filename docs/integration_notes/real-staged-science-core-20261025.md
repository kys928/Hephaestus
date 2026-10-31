# Evidence-backed scientific core for the 45-step staged campaign

Parent: draft PR #96 (`ade1ba4290ede32c22171fd125b5590c3a4414ae`).

## Delivered

- `StagedScientificEntry` reads a caller/operator-pinned, SHA-addressed `staged-diagnosis-input.v1` document and checks every source artifact's bytes, scope and kind. Judge entry admits inspection only. The real `IntegratedDiagnosisService` generates a diagnosis from those artifacts; completeness and blocking issues are not silently overridden.
- `StagedPlannerHandoff` reopens the persisted original diagnosis report and refuses incomplete/mismatched reports.
- `StagedRuntimeObservation` requires the persisted training lifecycle's completed status, exact checkpoint equality with the staged poll, and verification of finalized checkpoint-manifest bytes. It emits actual observed runtime and metric references, not synthetic success.
- `StagedRecoveryAdvice` gives no-recovery advice only if the verified training record is complete with no recorded incidents; it calls existing `BoundedRecoveryService` on observed incidents without executing a recovery action.
- `StagedSemanticJudge` reopens the immutable semantic comparison and calls the existing `SemanticComparisonJudgeAdapter` before emitting a verdict. Promotion is not permitted without an independent scientific certificate; the synthetic tiny campaign cannot mint one.
- `compose_verified_staged_dependencies` can create these missing core services when given an immutable `diagnosis_request_ref`; after that it still requires the existing actual discovery, local data factory, PEFT, generation, and model-backed role adapters.
- The incomplete evaluator placeholder was removed: a checkpoint-only adapter is an interim private assembly object, and no completed production inventory can use it as an evaluation service.
- Production composition now insists on the actual frozen `CertifiedRoleReasoner` and `PinnedTransformersRoleGenerator(device="cuda:0")`, and actual Transformers generation backend; a legitimate-looking wrapper around a fixture role generator is rejected.

## Why the requested all-real 45-stage CPU proof is still blocked

The frozen role mastery v1 contract hard-pins five specific 3B–8B model identities and the only certified model-backed role generator in `src/hephaestus/llm/certified_runtime.py` explicitly **requires one CUDA GPU**. It will refuse CPU inference, including on a GitHub hosted CPU runner. A tiny CPU PEFT model can demonstrate real training, generation, semantic evaluation and Judge logic, but it **cannot be passed off as one of these five frozen role models**.

A separate explicitly labeled CPU-only exploratory role contract would be needed to exercise all 45 steps with actual tiny-model role inference on CPU. That contract must never be allowed to certify the real Granite Evaluator or count as certified production five-role inference. Until such a separately governed scientific test is implemented, one contiguous 45-stage real-worker CPU proof is **not achieved**.

The diagnostic source-input tests contain synthetic cases that were written as immutable blobs; they prove admission and diagnosis execution, not independent original-data custody or a real experiment. Actual original-text near-duplicate checks, independent holdout custody and scientific certification are still external responsibilities. Never infer their completion from these CPU CI checks.

## Guardrails

- No automatic approvals, dataset downloads or network model acquisition.
- No unapproved live training or GPU deployment.
- No changes to frozen protected eval packs or model role identities.
- No simulated scientific outcome is called an observed real result.
- No production activation, certificate issue, or PR merge.
