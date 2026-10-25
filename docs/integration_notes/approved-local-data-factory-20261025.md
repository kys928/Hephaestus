# Approved local dataset → actual preprocessing → trainable-data contract

Parent: draft PR #93. This slice extends the existing eight-phase, 45-substep
orchestrator; it introduces no competing scheduler.

## What works in this draft

- After allowlisted discovery and deterministic selection, the existing staged
  `acquisition_approval_gate` requires a **separate operator decision**, even
  when the selected local source has a known acceptable license.
- `StagedApprovedLocalDataFactory` loads immutable discovery/selection/
  approval records from the campaign's durable repository, verifies their
  original artifact hashes and exact run/workflow identity, and authorizes
  **one local_fixture source only**.
- Its local acquisition uses `acquire_approved_local_candidate`, which checks
  approval scope, the source file, byte budget, record format and the declared
  source SHA-256. No network request is made.
- The existing `AutonomousDataPreprocessor.process` reads the **real local
  source bytes**, performs its existing normalization, exact deduplication,
  bounded audit, optional contamination checking and chunking, and produces
  the real dataset manifest, preprocessing report and trainable-data contract.
- Downstream staged steps reconstruct and compare the processed dataset and
  processing-evidence file against immutable receipts. Changes to original or
  processed bytes fail closed. Disabled execution never accesses data.

## Verified tests

GitHub Actions run 40177559993, head `68d82df55a7b5bc4356afd0ac60929feb8f3c29d`:
**5 focused passed, 612 full CPU tests passed, 9 skipped.**

The positive source is **synthetic JSONL bytes written by the test**. All
acquisition/preprocessing code, source hashing, immutable storage and data
transformation logic are the actual production implementations. Operator
approval is an explicit test fixture rather than real organization permission.

## Limitations that must not be disguised as completion

- The local-fixture provider must be independently configured/allowlisted and
  discovered. This slice does NOT make Hugging Face acquisition and its
  remote approval/reconciliation path operational in the staged campaign.
- For a Planner intervention that requires no new dataset, this adapter does
  not rebind an already-certified trainable-data artifact; it blocks.
- Tokenizer compatibility reports `not_requested` unless a pre-existing
  checker is installed and configured. This is **not evidence of tokenizer
  compatibility** and must not be treated as a verified green training gate.
- The data-factory result is **not yet automatically joined** into the
  Planner's originally unresolved experiment proposal and exact PEFT model/
  tokenizer/recipe identities. Training deliberately cannot start from such
  an unresolved proposal.
- Full end-to-end candidate training, generation, genuine independent
  scientific-state holdouts, certified promotion and active role replacement
  remain unverified. No GPU, RunPod pod, remote dataset download, protected
  evaluation mutation, merge, or production role change occurred.

Recommended next integrated slice: policy-admitted proposal finalization,
binding the approved dataset manifest and pinned model selection into the
existing training service's detailed data/tokenizer/recipe validation. Only
then run real tiny CPU PEFT within the entire campaign and its actual
generation/decision path.
