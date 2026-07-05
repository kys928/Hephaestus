# Control Spine

The control spine is the mandatory execution skeleton for every Hephaestus workflow. It is not a recommendation: it is the boundary that prevents training, evaluation, and governance from collapsing into an opaque orchestrator.

## Mandatory phase order

The implemented order is:

1. `judge_entry`
2. `planner`
3. `data_acquisition_audit`
4. `data_preprocessor`
5. `training_engineer`
6. `runtime_monitor`
7. `evaluator`
8. `judge_exit`

`SPINE_ORDER` is the source of truth. `StateMachine.advance()` follows `transition_rules.next_phase()` and returns `None` after `judge_exit`.

## Phase ownership

Each phase has one primary role owner:

| Phase | Role owner | Coordination owner | Durable outputs |
| --- | --- | --- | --- |
| `judge_entry` | `JudgeEntryRole` | `DefaultSpineCoordinator` | decision record |
| `planner` | `PlannerRole` | `DefaultSpineCoordinator` | experiment-plan report |
| `data_acquisition_audit` | `DataAcquisitionAuditRole` | `DefaultSpineCoordinator` | dataset manifest, dataset-profile report, artifact index |
| `data_preprocessor` | `DataPreprocessorRole` | `DefaultSpineCoordinator` | preprocessing report, trainable-data-contract report, artifact index |
| `training_engineer` | `TrainingEngineerRole` | `DefaultSpineCoordinator` | training-plan report, launch-config report |
| `runtime_monitor` | `RuntimeMonitorRole` | `DefaultSpineCoordinator` | incidents, runtime artifact refs |
| `evaluator` | `EvaluatorRole` | `DefaultSpineCoordinator` | eval report, checkpoint artifact ref, intermediate eval refs |
| `judge_exit` | `JudgeExitRole` | `DefaultSpineCoordinator` plus policy modules | decision, approvals, lineage/run/memory updates |

Role logic must remain in `src/hephaestus/roles/`; phase sequencing and cross-role wiring must remain in `src/hephaestus/control/`.

## Phase-result contract

Every phase returns a `PhaseResult` with:

- `phase`: the `SpinePhase` enum value that ran.
- `status`: a short status string; implemented successful phases use `ok`.
- `artifact_refs`: a list of path references to heavyweight outputs.
- `output`: a short JSON-serializable payload or `None`.

`PhaseResult.output` is not a place for raw datasets, model weights, full logs, or probe generations. Those must be represented by path references.

## Context passing

`ControlContext` carries `run_id`, `lineage_id`, `stage_name`, `artifact_root`, and an in-memory `outputs` dictionary. Later phases read prior phase outputs by phase name. This context is execution glue, not durable memory. Decision-critical facts must be appended through state stores.

## Implemented data flow

1. Entry judge reads lineage state, recent failures, checkpoint repeatability, and selected memory records before choosing entry mode.
2. Planner emits an experiment plan for the run and stage.
3. Data acquisition/audit calls the backend for dataset metadata and persists a normalized manifest.
4. Data preprocessing calls the backend, emits a preprocessing report and a trainable data contract.
5. Training engineer creates a backend-specific launch config and training plan.
6. Runtime monitor resolves stage policy, prepares and launches the backend job, classifies health, derives incidents, and recommends stop/continue behavior.
7. Evaluator reads stage profile and backend training outputs, evaluates deterministic gates, selects a checkpoint, and builds an eval report with scorecard.
8. Exit judge evaluates promotion and action policy, records the decision/approval path, updates lineage/run state, and builds memory records from the run.

## Governance boundaries

The control spine may route data and call policies, but policy decisions are not embedded in role code. Stage gates come from `StagePolicy`; runtime outcomes from `RuntimePolicy`; judge actions from `JudgePolicy`; promotion state from `PromotionPolicy`; high-impact boundaries from action/approval policy. This separation is required so operators can audit why a run proceeded, stopped, promoted, rolled back, branched, or required approval.
