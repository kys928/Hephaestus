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
