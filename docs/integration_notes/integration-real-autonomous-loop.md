# Integration note: real autonomous experiment loop

## Branch and scope

Branch: `integration/real-autonomous-loop`.

This branch composes the independently implemented autonomous-experiment subsystems while preserving the existing mandatory control spine, approval policy, promotion gates, lineage governance, replay verification, and frozen evaluation rules.

Merged feature pull requests:

- #39 — evidence-based diagnosis
- #41 — governed autonomous data factory
- #37 — semantic and behavioral evaluation
- #38 — governed model discovery and real training lifecycle
- #36 — closed-loop experiment planner
- #40 — bounded execution infrastructure

None of those changes were merged into `main` directly. They were retargeted and merged into this integration branch.

## Review findings corrected by the integration layer

### Key-aware diagnostic evidence

The feature diagnosis normalizer accepts heterogeneous evidence. Generic string truth handling could invert positive fields such as `eval_integrity_verified="verified"` and could miss the data factory's nested `tokenizer_compatibility.compatible` result. `IntegratedDiagnosisService` and `TruthNormalizingEvidenceAdapter` normalize these values before deterministic diagnosis.

### Approval-gated dataset selection

The data selector correctly identified otherwise-suitable candidates that require license, provenance, or trust approval, but returned no selected candidate IDs. That prevented the planner from carrying the approval requirement forward. `ApprovalAwareDatasetSelectionService` records such candidates as selected subject to approval. Acquisition remains impossible without a concrete `DatasetAcquisitionApproval`, so this does not weaken the boundary.

### Training-control evidence failures

Malformed or missing restored resume/job evidence could escape as a persistence exception. `GuardedTrainingLifecycleService` converts those failures into blocking `ContractIssue` records and a failed `TrainingRunHandle`.

### Generic baseline references

The shared `ExperimentProposal.baseline_ref` is intentionally generic, while semantic comparison consumes run handles. `AutonomousExperimentCoordinator.compare` accepts a baseline resolver, records the original source reference, and passes a concrete baseline run ID to the evaluator.

### Processed-data binding

The local training smoke lifecycle consumes the processed dataset artifact itself. `bind_training_inputs` uses `DataFactoryResult.processing_evidence.processed_dataset_ref` and its actual content hash instead of passing the JSON `TrainableDataContract` record as training bytes.

### Unstructured model requirements

Planner-generated free-form problem statements are not treated as exact model capability labels. The coordinator prefers structured task requirements from diagnosis metadata. When those are absent, it clears the free-form exact-match constraint, records a missing-evidence issue, and caps model-selection confidence rather than rejecting every candidate or pretending a precise capability match.

## Public integration entry point

Use:

```python
from hephaestus.control import AutonomousExperimentCoordinator
```

The coordinator requires explicit injected services and an append-only `IntegrationRecordSink`. It does not replace the existing orchestrator and does not silently launch work.

Main operations:

1. `diagnose_and_plan(request)`
2. `discover(diagnosis, intervention)`
3. `build_experiment(diagnosis, intervention, discovery)`
4. `prepare_selected_dataset(...)`
5. `bind_training_inputs(...)`
6. `launch_approved(proposal, approval_evidence)`
7. `compare(proposal, runs, baseline_resolver=...)`

Every decision-critical output is appended to the configured integration record sink.

## Authority boundaries

- Diagnosis observes and hypothesizes; it does not execute interventions.
- Planner proposes; it does not call providers or launch training.
- Providers discover; selectors select.
- Dataset acquisition requires explicit approval evidence.
- Model and dataset selection do not approve themselves.
- Training launches only `ready` or `approved` proposals and only when all named approvals have evidence.
- Evaluator compares; it does not promote.
- Existing Judge, approval, action-boundary, promotion, rollback, branch, restart, lineage, and replay policy remain authoritative.

## Infrastructure wiring

The infrastructure branch provides adapters rather than a forced migration:

- `InMemoryJobQueue` and `LocalWorker` for local bounded jobs
- `FileSystemArtifactStore` for immutable SHA-256 artifacts
- `JsonLineStateRepository` for locked single-record appends
- secret references resolved only at runtime
- structured event sinks, telemetry, health, and container boundaries

The coordinator's record-sink protocol can be adapted to `JsonLineStateRepository` or existing state stores. This branch deliberately does not replace all existing stores.

## Validation

Each feature branch reported its own focused and full-suite checks before integration. Those results are useful but do not prove the composed branch.

The integration branch adds `tests/test_autonomous_experiment_integration.py` for:

- positive diagnostic truth normalization
- nested tokenizer compatibility evidence
- approval-gated dataset selection
- structured training-control failure handling
- generic baseline resolution
- launch approval enforcement
- unstructured model-requirement handling

A new PR-only GitHub Actions workflow compiles the source and runs the full `pytest` suite on the composed branch. The integration PR should remain draft until that check passes.

## Known limitations

- The real training lifecycle currently proves bounded process/checkpoint/resume behavior with a tiny dependency-free fixture model; useful Hugging Face model training still requires a later backend implementation.
- Dataset remote acquisition is intentionally absent; the built-in real path is approved local data, while the Hugging Face provider is metadata-only.
- Production-grade PII and contamination checkers remain injected hooks.
- Semantic evaluation requires recorded generation artifacts; the fixture trainer does not itself generate language-model samples.
- The local job queue and lock are process-local and do not claim distributed guarantees.
- The coordinator is a safe integration facade, not a replacement for the existing staged orchestrator. Narrow phase-level orchestrator adoption can happen after the composed PR passes review and CI.

## Merge guidance

Do not merge this branch into `main` until:

1. the composed full test suite passes;
2. the integration regression tests pass;
3. frozen eval-pack hash verification passes;
4. reviewers confirm that approval and promotion boundaries remain unchanged;
5. the tiny real training smoke run is exercised in a normal checkout;
6. generated datasets, checkpoints, state, secrets, caches, and model weights are absent from the diff.
