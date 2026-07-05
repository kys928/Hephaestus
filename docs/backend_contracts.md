# Backend Contracts

Backends are bounded execution adapters. They provide data acquisition, preprocessing, training launch, runtime events, artifacts, checkpoint candidates, and intermediate evaluation signals, but they do not make governance decisions.

## ExecutionBackend protocol

An execution backend implements these methods:

- `resolve_target(launch_config) -> BackendTarget`
- `acquire_dataset(run_id) -> dict`
- `preprocess(run_id) -> dict`
- `prepare_training_job(experiment_plan, data_contract, training_plan, launch_config) -> PreparedBackendJob`
- `launch_training(prepared_job) -> BackendRunResult`
- `stop(run_id) -> None`

The protocol is intentionally backend-agnostic. Local processes, dry runs, Hugging Face causal LM jobs, and external runtime adapters must all fit behind this boundary.

## Backend target

`BackendTarget` identifies the backend selected for a launch:

- `backend_name`: stable backend identifier.
- `dry_run`: whether the target is simulated.
- `config`: JSON-serializable launch configuration.

Backends may enrich `config`, but they must not hide governance-critical parameters outside serialized launch records.

## Prepared job

`PreparedBackendJob` is the handoff between training engineering and runtime launch:

- `run_id`
- `backend_name`
- `artifact_root`
- `expected_artifacts`
- `execution_spec`

The prepared job must contain references and execution metadata only. It must not embed training datasets or model checkpoints.

## Run result

`BackendRunResult` reports bounded runtime output:

- `run_id`
- `status`
- `events`: typed `RuntimeEvent` instances.
- `artifact_refs`: path references produced by training/evaluation.
- `checkpoint_candidates`: candidate checkpoint records, usually with `checkpoint_ref` and optional scores.
- `intermediate_eval`: short metric/probe/deterministic-check references and inline scalar fallback values.

The evaluator expects `intermediate_eval` to contain or point to `probe_score` and `toxicity` evidence. If referenced metrics are missing and no inline metrics are present, evaluation marks evidence limitations rather than inventing metrics.

## Runtime-event expectations

Runtime events must use `RuntimeEvent` and `RuntimeEventCategory`. Implemented categories are consumed by monitoring policy as follows:

- `INCIDENT` events are converted to `IncidentRecord` values.
- `DETERMINISTIC_CHECK` events whose message contains failure text can trigger hard abort classification.
- `METRIC`, `PROBE`, and other status events may carry `payload_ref` values and are indexed as artifacts.

Subprocess output can be parsed from lines with the format `EVENT|category|step|message|payload_ref`.

## Dry-run backend behavior

The built-in `DryRunBackend` is the reference minimal implementation. It returns:

- A synthetic approved dataset identity, source ids, quality score, risk list, total example count, and dataset-manifest artifact ref.
- A preprocessing operation list and processed-dataset artifact ref.
- A prepared job with expected metric, probe, and deterministic-check artifact paths.
- Runtime events for status, metrics, probe output, and deterministic checks.
- Two checkpoint candidates and intermediate eval values for `probe_score=0.68` and `toxicity=0.04`.

Dry-run outputs demonstrate control flow and schema boundaries only. They are not evidence of real model training.

## Backend invariants

- Backends must return JSON-serializable metadata and typed events.
- Backends must represent heavy outputs by path reference.
- Backends must not decide promotion, certification, rollback, branching, or approval status.
- Backends must not mutate lineage stores or decision stores directly.
- Backend-specific failures must surface as status, runtime events, incidents, missing artifacts, or explicit exceptions handled by the control layer.
