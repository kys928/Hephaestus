# Real staged data/model discovery and PEFT lifecycle integration

Parent: draft PR #91 at `50151d890b5bd527c3f796a692c231d18eb2815a`.

## Implemented in this PR

- `ProductionRuntime.with_discovery_services(...)` binds the **existing**
  allowlisted `DatasetProviderRegistry`, `ApprovalAwareDatasetSelectionService`,
  model discovery methods, and `DeterministicModelSelectionService` to
  actual staged operation names.
- The Planner's `search_request` is read from exact workflow/run/lineage/
  stage records and verified against immutable content-addressed artifact bytes,
  not arbitrary caller-provided memory.
- Discovery persists real provider candidates and blocking issues. Selection
  is performed against those exact verified candidates, preserving existing
  licensing/metadata/approval policies. Absent Planner requests generate an
  explicit `not_required` result, **never** fabricated candidates.
- `ProductionRuntime.with_training_services(...)` binds the existing guarded
  `TransformersTrainingLifecycleService` (PEFT-capable since draft PR #88)
  to `training_input_binding`, `launch_configuration`,
  `lifecycle_launch`, `training_status_poll`, and `checkpoint_resolution`.
- It reconstructs an exact `ExperimentProposal` from immutable stage records,
  checks its run/lineage identity, execution readiness, bounded explicit seed
  and necessary training input fields.
- Training is **disabled by default**; with explicit execution enable, a
  previously known run is reconciled via lifecycle status before any launch.
  A foreign/failed run cannot be overwritten. Running work is polled and
  remains a durable waiting cycle. Completed work is handed to the evaluator
  only after the real finalized checkpoint manifest and every component byte
  pass verification.
- One common immutable workflow-record reader is shared across stages.

## Actual tests and scope

The CPU tests use **real composition, staged orchestrator, SQLite state,
artifact records, real dataset/model policy selectors, and real checkpoint
manifest verification**. Providers/generation/training workers are clearly
identified deterministic CPU test doubles where external resources would
otherwise be needed.

They cover:
- Real allowlisted offline dataset and model discovery and deterministic
  selection, including revisions and approval metadata.
- Correct explicit `not_required` discovery outcomes.
- Missing providers and tampered staged records.
- Launch-disabled default and interrupted-cycle persistence.
- Resuming a completed training job without re-launching a second job.
- Missing data, unresolved discovery and required approvals refusing training.
- Tampering a finalized checkpoint refusing evaluator handoff.

The existing tiny-model CPU LoRA tests in #88 independently prove that the
real PEFT lifecycle can optimize and resume, but that real optimization is not
yet exercised inside the 45-stage workflow in this PR.

## Remaining non-negotiable wiring

The current implementation is **not yet full autonomous self-repair**.
The original Planner may produce an unresolved draft `ExperimentProposal`;
an approved dataset/model selection and verified preprocessing must be joined
into a **final policy-admitted training proposal** before the training handoff.
Until then, this adapter deliberately blocks `unresolved_discovery`.

Still required:
1. Real governed dataset acquisition and preprocessing staged substeps,
   including durable approval and checksum receipts.
2. Experiment-proposal finalization from independently verified selections
   and processed training data (no fabricated source datasets).
3. Real baseline/candidate generation resolving their respective completed
   training/evaluation receipts from the immutable workflow, and subsequent
   independent comparison.
4. Role-backed Judge/certification gate using the genuine comparison and
   operator-scoped approval; automatic activation remains separately gated.
5. A full CPU campaign that invokes actual PEFT optimization in-loop, then a
   separately approved and budget-capped GPU validation.

No RunPod resource was created, no external dataset downloaded, no protected
evaluation pack altered, no PR merged and no production model activated.
