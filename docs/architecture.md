# Architecture

Hephaestus is a bootstrap-stage training-governance system organized around explicit role boundaries, typed payloads, policy modules, and filesystem-backed state. The implemented system is intentionally small and inspectable: orchestration is concrete, many domain subsystems remain typed stubs, and decision-critical records are serialized as deterministic JSON/JSONL rather than hidden in process memory.

## Control spine is the top-level invariant

All workflow execution preserves this fixed phase order:

1. `judge_entry`
2. `planner`
3. `data_acquisition_audit`
4. `data_preprocessor`
5. `training_engineer`
6. `runtime_monitor`
7. `evaluator`
8. `judge_exit`

The phase order is declared by `SPINE_ORDER` and must not be collapsed into a generic monolithic training loop. Coordination belongs in `src/hephaestus/control/`; phase-specific behavior belongs in `src/hephaestus/roles/`. The default coordinator invokes one role per phase and stores phase outputs in `ControlContext.outputs` only as short JSON-serializable records and artifact references.

## Directory responsibilities

- `src/hephaestus/control/` owns sequencing, branching, rollback, restart, promotion application, replay verification, and state-machine transitions.
- `src/hephaestus/roles/` owns role-local phase behavior for the mandatory spine.
- `src/hephaestus/schemas/` owns cross-component payload shapes. Payloads that cross role, backend, policy, or state boundaries must be represented by an explicit schema class or normalizer here.
- `src/hephaestus/policy/` owns all governance decisions: stage resolution, runtime classification, judge actions, approvals, promotion, restart, and code-edit boundaries.
- `src/hephaestus/state/` owns durable decision-critical records. JSONL stores are append-oriented; lineage state is a single JSON document indexed by lineage.
- `src/hephaestus/backends/` owns execution backends behind `ExecutionBackend`. Backends may train, dry-run, or adapt external runtimes, but they do not decide promotion or governance outcomes.
- `src/hephaestus/runtime/` owns subprocess launch helpers, event stream parsing, incident derivation, stop recommendation, and runtime session structure.
- `src/hephaestus/evaluation/` owns deterministic metric reading, regression checks, checkpoint selection, eval-pack loading, and stage interpretation.
- `src/hephaestus/scoring/` owns deterministic gates and score aggregation primitives.

## Implemented behavior at a glance

The default workflow coordinator executes each phase in order, appends decision records from judge phases, stores plans and reports, indexes artifact references, and updates lineage after judge exit. Dataset acquisition and preprocessing are delegated to the configured backend. Training launch is prepared by the training role and executed by the runtime-monitor role through the backend. Evaluation consumes backend training outputs, deterministic gates, and eval-pack policy to produce an `EvalReport` with an embedded deterministic scorecard.

The built-in `dry_run` backend returns synthetic dataset metadata, preprocessing output, runtime events, intermediate metric/probe/deterministic-check artifact references, and checkpoint candidates. This makes the control path runnable without claiming to perform real training.

## Schema boundaries

A component may pass dictionaries internally, but externally visible records must conform to schemas in `src/hephaestus/schemas/`. Important schema boundaries include:

- Backend target/job/result payloads in `schemas/backend_contracts.py` and `backends/base.py`.
- Dataset profile, dataset manifest, preprocessing report, and trainable data contract for data phases.
- Experiment plan, training plan, and launch config for planning/training phases.
- Runtime event and incident record for runtime monitoring.
- Eval pack, eval report, checkpoint resolution, regression summary, scorecard, gate result, and metric summary for evaluation.
- Judge entry, judge exit, decision record, approval request, approval decision, promotion gate report, lineage state, run record, and memory record for governance and persistence.

Unknown or missing fields are handled by each schema or policy loader according to its domain. For example, scorecards reject unknown fields when reconstructed with `Scorecard.from_dict`, and stage profiles require `strictness`, `eval_pack`, `deterministic_gates.min_probe_score`, and `deterministic_gates.max_toxicity`.

## Artifact handling invariant

Heavy artifacts are never embedded in control records. Control records may include `artifact_ref`, `payload_ref`, `metrics_ref`, `probe_ref`, checkpoint paths, or other path-like references. The artifact index stores `{run_id, kind, ref}` records so operators can locate heavy evidence without expanding state memory.

## Operator-governance model

The judge roles propose actions; policy modules constrain them; approval policy and action-boundary evaluation can require operator approval or block unsafe actions. Promotion is not a raw metric comparison: deterministic gates, confidence, evidence completeness, promotion bundles, repeatability, variance risk, stage certification settings, approval decisions, and lineage history all influence the effective action.

Operator governance is therefore explicit and auditable:

- Entry judge chooses how the lineage should proceed based on lineage status, failures, stable checkpoints, and relevant memories.
- Exit judge chooses a proposed action based on monitor outcome, eval report, promotion state, and recent failure context.
- Promotion gates and action-boundary policy can downgrade or block requested actions.
- Approval requests and decisions are persisted separately from ordinary decision records.
- Lineage state records best, stable, and certified checkpoints plus trust/status changes.

## Bootstrap constraints

The codebase prefers typed stubs and TODO markers over fake-complete implementations. Documentation must distinguish implemented behavior from intended future behavior. At present, dry-run execution and deterministic JSON records are implemented; several modules in data, LLM, safety, and evaluation remain explicit scaffolds.
