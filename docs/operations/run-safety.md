# Safe operation and incident boundaries

**Default posture:** local inspection and CPU tests are safe starting points; live provider access, training, artifact mutations and promotion are separate governed actions.

## Before any real experiment

1. Freeze a scoped objective, success/failure gates, candidate/base identities, seeds, tokenizer, train data and **independently protected** eval data. Keep data rights and leakage status explicit.
2. Admit model/data revisions by stable reference/hash and record stage-specific manifests, preprocessing and training configuration.
3. For paid execution, require a new explicit approval identifying provider/region, exact GPU SKU/VRAM, run identity, currency/rate/budget ceiling, wall-clock/pod ceiling and termination plan. No silent upgrade for capacity scarcity.
4. Verify durable evidence/checkpoint storage identity and admission, independent re-read and cleanup plan before spending money.
5. Confirm trained-model activation, role dispatch, eval-pack mutation and checkpoint promotion are **not** inferred from a successful training or infrastructure run.

[Compute execution policy](../compute_execution_policy.md), [action boundary](../autonomy_boundary_policy.md), [promotion gates](../promotion_gate_policy.md) and [durable storage design](../durable_model_storage_and_repository_evidence.md) are normative.

## Secrets and provider access

Keep RunPod, S3, Hugging Face, GitHub or other provider credentials behind the deterministic execution layer; model role context gets only approved typed intents and safe metadata. No secrets, protected training examples, private holdouts, model weights, checkpoints or generated runtime SQLite belong in source control.

On 2026-10-31, [PR #157](https://github.com/kys928/Hephaestus/pull/157) retired 13 historical one-off workflows from the default branch; only five reviewed CPU CI definitions remain. **Do not recover an old paid workflow or reuse an old approval to launch work.**

## If something fails

- Preserve status and authentic event/receipt references first. Never synthesize missing optimizer updates, stages, role inferences or completion markers.
- Distinguish provider allocation failure, model load failure, data/format mismatch, genuine optimizer instability and evaluation invalidity.
- Before retry, inspect whether a Pod exists, whether billing can continue, whether an idempotency/once-claim has already been consumed and whether the exact frozen run supports resume.
- Recover only via a bounded, policy-allowed action with fresh approval when required. Unknown outcomes are `inconclusive` and block automatic promotion.
- Verify provider deletion/teardown independently; mark price estimates as **estimates**, not paid invoices.

[Bounded recovery](../autonomous_recovery.md), [replay verification](../replay_verification_policy.md) and [R10 executed-science report](../evidence/original-cuda45-r10/README.md) provide concrete reference evidence.

## Do not overclaim

Internal R10 finished 45/45 native steps, 5/5 CUDA model roles and 96/96 target updates. This does not authorize other runs, prove general model improvement, satisfy independent original-private-data contamination checks or enable unrestricted autonomous dispatch. External certification remains tracked by [#102](https://github.com/kys928/Hephaestus/issues/102); prospective improvement belongs to [#152](https://github.com/kys928/Hephaestus/issues/152).
