# Verified scientific training proposal finalization and genuine tiny-model CPU test

Parent: draft PR #94 (SHA `51f3b28e6c57f10b26224771418e1d3c8dda4506`).

## Production behavior

`StagedExperimentFinalizer` performs the previously missing transition from
the Planner's unresolved proposal to an executable `ExperimentProposal`.
It derives identities from immutable SQLite + content-addressed workflow
records and checks source-file contents rather than trusting proposal prose.

Inputs include:
- Planner's original experiment proposal, with exact workflow/run/lineage IDs;
- actual deterministic dataset and model selection records and discovered
  candidate payloads;
- an independently scoped acquisition approval tied to the selected dataset;
- verified local source and Data Factory preprocessing receipts;
- processed training data, trainable-data contract and processing evidence
  with matching SHA-256 digests;
- positively checked tokenizer compatibility and pinned local tokenizer bytes;
- pinned model directory bytes and immutable revision;
- an independently allowlisted, SHA-addressed PEFT recipe with matching model
  selection and bounded training steps;
- an independently allowlisted baseline identity.

Any unresolved selection, absent approval, unchecked tokenizer, mismatched
model bytes, unsupported remote acquisition, unauthorized recipe, or changed
training data blocks finalization. The finalizer never calls a network
provider, launches training, certifies improvement or changes role routing.

When all contracts agree, it produces an approved training proposal containing
the **actual** `TransformersTrainingLifecycleService` field names
(`trainable_data_contract_ref/hash`, `processed_dataset_ref/hash`,
`processing_evidence_ref/hash`, `tokenizer_id/revision`, model/revision,
LoRA mode, fixed recipe and explicit seed).

`ProductionRuntime.with_training_services(...,` 
`finalize_verified_selections=True, approved_recipe_refs=..., approved_baseline_refs=...)`
activates this new step. The existing default behavior remains unchanged until
explicitly selected by the deployment owner. The runtime-monitor stage now
polls the actual training lifecycle; unrelated monitoring is delegated.

## Tests and valid claims

The core CI validates the real source preprocessing, selection, approval,
receipt and training-field join with adversarial inputs.

The opt-in PEFT CI installs actual CPU PyTorch/Transformers/PEFT dependencies,
constructs a real tiny GPT2 model, trains LoRA through the finalized real
dataset/model/recipe proposal, verifies PEFT adapter checkpoint bytes and
generates tokens using the actual Transformers inference backend.

A separate **test-only** frozen evaluation pack is built for bounded,
short-context measurements. The protected production semantic pack and
scientific-state holdout are untouched. The test compares the base and adapted
model's generated samples and submits the independent semantic
`ExperimentComparison` to the governed `SemanticComparisonJudgeAdapter`.

That is an integration test of real optimizer/model weights, not a claim that
the tiny model becomes scientifically superior or that the 45-step autonomous
recovery campaign has independently produced an unseen-holdout certificate.

## Genuine blockers

1. Full 45-substep production dependency composition is still incomplete;
   the existing simulator-backed three-cycle campaign in PR #91 is not the
   same test as this real tiny-weight test.
2. An operator-governed genuinely untouched Evaluator scientific-state
   holdout, separate from all D diagnostics/development cases, is not
   available in this branch. No real Evaluator scientific certification may
   be declared without it.
3. Frozen certified Granite Evaluator GPU inference and PEFT adaptation have
   not been run under this integrated campaign; activating the real role
   remains forbidden until separate approval and verification.
4. The first full local data processor does not enable a remote Hugging Face
   acquisition path. Offline sources require explicit selection and approval.

No RunPod Pod or GPU was created, no production adapter replaced, no PR merged,
and protected benchmarks remain unchanged.


## Independent scientific-state custody and external certification gates

Added `evaluation/independent_holdout_review.py`. This module **does not
self-certify a candidate**. It requires an operator-pre-registered immutable
holdout digest and separately provisioned Ed25519 custodian public key, then
authenticates the custodian's signed statement over the precise holdout,
train/development exposure manifests, diagnostic provenance, baseline/candidate
checkpoints, and repeated prediction evidence. The signed custodian claims
that labels remained blinded until predictions were fixed, no training access
was granted, and checkpoint bytes were checked by an independent evaluator.

It detects exact training/development/diagnostic collisions using case IDs and
example fingerprints, recomputes the existing eight-state individual,
protected-state and contrast scores from content-addressed predictions and
checks repeatability across at least three matched seeds. A preregistered
practical gain and a one-sided paired sign test across **distinct contrast
pairs** (not repeated seed-pairs) must pass before the report says
`eligible_for_external_certification_review`. The output remains
`not_certified` with `live_activation_authorized=false`.

The `verify_external_evaluator_certificate` entry point requires a
*second*, different operator-trusted Ed25519 certifier identity/key. It
verifies the separately signed externally issued certificate, its timestamp,
exact candidate checkpoint/holdout/assessment IDs and independently re-runs
the raw review. It cannot mint an external certificate.

`EvaluatorRoleActivation.activate(simulated=False)` now **refuses** to
activate a role without the exact independent certificate ref and the
separate custodian/certifier trust roots provisioned onto the manager by an
operator. Even then it must separately pass existing lineage promotion,
checkpoint integrity, activation authorization, and real GPU model-probe
checks. Simulated CPU tests do not satisfy any live-activation requirement.
An active verified route also checks that the certificate receipt bytes still
exist under their original hash. The frozen registry remains unchanged.

### Limitations and required real-world evidence

The positive signature/certification-verifier tests use **explicitly fabricated
small scientific-state examples signed by test-generated keys**, which proves
signature and enforcement wiring but cannot establish any real Evaluator
improvement. There is **no real external custody signature, independently
unseen D holdout, signed research certifier verdict or Granite GPU validation**
in this PR, and therefore **no valid real Evaluator certificate**.

A digital signature establishes which trusted key signed a statement; it does
not prove that the signer's claim is truthful, that all semantic contamination
was eliminated, or that the contrast pairs are independent statistical
draws. Trust-root provisioning, offline holdout custody, record of exactly
which examples every training/evaluation system was exposed to, and external
scientific review remain separate real operational responsibilities. The
statistical test assumes independent contrast-pair observations.

No route was activated in production and no costed infrastructure was used.
