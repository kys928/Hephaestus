# Integration note: real training lifecycle and governed model discovery

## Scope and ownership

Branch: `feature/real-training-lifecycle`.

This feature implements model discovery/selection and a bounded real local training lifecycle. It does not choose an experiment hypothesis, modify shared `autonomous-experiment.v1` contracts, wire the orchestrator, change approval or promotion policy, or enable an external registry by default.

## Provider and selector classes

- `CatalogModelProvider`: loads an explicit local JSON catalog or injected catalog entries and normalizes them into `ModelCandidate` records.
- `FakeModelProvider`: returns deterministic, download-free fixture candidates.
- `ExternalModelRegistryProvider`: optional adapter around an injected registry callable. It raises while disabled and performs no implicit network access.
- `DeterministicModelSelectionService`: implements `ModelSelectionService` and ranks only candidates that pass mandatory license, revision, architecture, tokenizer, context, compute, backend, provenance, integrity, runtime, and smoke-test checks.

The selector returns `blocked` when all candidates are incompatible. Unknown license and unavailable revision are explicit rejection reasons. Candidate order cannot change ranking because total score and candidate ID form a deterministic sort key. Score components are preserved in decision metadata.

## Training service and backend configuration

`LocalTrainingLifecycleService` implements the shared `TrainingLifecycleService` protocol with `launch`, `status`, and `control`. Its backend ID is `local_fixture`. `FakeTrainingLifecycleService` is the deterministic consumer-test implementation.

Construct the local service with an explicit artifact root:

```python
service = LocalTrainingLifecycleService(Path("artifacts/autonomous-training"))
```

An `ExperimentProposal` must have status `ready` or `approved`. Its `training_constraints` must provide:

- `backend_id`
- `model_id` and `model_revision`
- `architecture_family`
- `tokenizer_ref`
- `training_recipe_ref`
- `data_contract_ref` and its actual SHA-256 `data_contract_hash`
- positive bounded `max_steps`

Optional fixture controls include `learning_rate`, `step_delay_seconds`, `force_exit_code`, and `omit_artifacts`. The latter two exist to exercise incident and integrity behavior in tests; production integration should not expose them to untrusted callers.

## Model fixture and process boundary

`hephaestus.training.fixture_worker` is a tiny pure-Python linear next-byte predictor. It performs real gradient updates in a separate OS subprocess but has only two trainable parameters. It needs no external model download, GPU, PyTorch, or Transformers dependency. This is deliberately a lifecycle smoke fixture, not a claim of useful language-model quality.

The service persists `prepared_job.json` before spawning the worker. Existing run evidence blocks implicit overwrite. Status is honest: `launch` returns `running` only after a process exists; active controls use `interrupting`; resume starts as `resuming`; completion is reported only after process exit and artifact validation.

## Artifact root and lifecycle evidence

Each run writes under `<artifact_root>/<run_id>/`:

- `prepared_job.json`: validated launch and compatibility inputs
- `runtime.log`: subprocess stdout/stderr reference
- `events.jsonl`: runtime state events
- `metrics.jsonl` and `metrics_summary.json`: step and final metrics
- `checkpoint_step_<n>.json`: fixture checkpoint
- `checkpoint_record.json`: checkpoint identity, step, metric evidence, computed SHA-256, integrity level, trainer/config references, tokenizer/model revision, resume compatibility, and partial-write status
- `resume_token.json`: checkpoint hash, configuration fingerprint, and compatibility evidence
- `runtime_result.json`: worker exit status
- `incidents.jsonl`: normalized non-zero-exit and missing/integrity incidents
- `handle.json`: latest `TrainingRunHandle`

Heavy evidence stays out of the shared handle. The handle contains references only.

## Control and resume semantics

- `status`: polls the real process and validates terminal artifacts.
- `interrupt`: sends a graceful interrupt, records `interrupting`, and becomes `interrupted` only when the worker checkpoint and resume evidence are available.
- `cancel`: requests graceful termination and becomes `cancelled` after the process exits.
- `resume`: allowed only from `interrupted` and starts a new subprocess from the verified checkpoint.

Resume validates the actual checkpoint SHA-256 and exact backend, model ID, model revision, architecture family, tokenizer, training recipe, data contract reference, and data contract hash. It also validates the prepared-job configuration fingerprint. Missing or mismatched evidence refuses resume; path existence alone is never called hash verification.

## Contracts consumed and produced

Consumed:

- `ModelSearchRequest`
- `ModelCandidate`
- `ExperimentProposal`
- `TrainingControlRequest`
- `autonomous-experiment.v1` issue vocabulary

Produced:

- normalized `ModelCandidate` sequences
- `ModelSelectionDecision`
- evolving `TrainingRunHandle` records
- referenced runtime, metric, checkpoint, resume, log, and incident evidence

## Failure modes

- pending/unapproved proposals fail prepared-job validation
- missing or hash-mismatched data fails before process launch
- unsupported backend or invalid step/learning-rate budget fails before launch
- unknown license, missing revision, or incompatible runtime blocks model selection
- non-zero process exit creates a persisted `runtime_failure` issue and incident
- missing required terminal artifacts fail an otherwise completed run
- malformed or hash-invalid checkpoint evidence fails the run
- insufficient or incompatible resume evidence refuses resume
- absent run IDs return an explicit failed handle with `invalid_request`

## Final wiring instructions

The final integration branch should instantiate approved providers and one selection service, persist the selection decision, then create `LocalTrainingLifecycleService` with the configured artifact root. It must pass only planner-issued, readiness-approved proposals. Keep `force_exit_code`/`omit_artifacts` test-only, preserve approval gates before launch, and let evaluator/judge phases consume the returned evidence references without promoting directly from lifecycle status.
