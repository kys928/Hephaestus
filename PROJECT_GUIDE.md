# PROJECT_GUIDE.md

## Mission
Hephaestus is a disciplined research-engineering system for diagnosing, repairing, and evaluating model development cycles with explicit safety constraints and auditability.

## Role separation
The system is role-bound and phase-bound. Each role consumes typed inputs and emits typed outputs; no role should directly mutate global state outside store interfaces.

Mandatory sequence:
- Judge(entry) -> Planner -> Data Acquisition & Audit -> Data Preprocessor -> Training Engineer -> Runtime Monitor -> Evaluator -> Judge(exit)

## Stage-aware evaluation
Evaluation must be stage-aware:
- stage profile selects valid checks,
- evaluation pack defines probes/metrics,
- deterministic regression gates remain explicit and reproducible.

## Compact memory model
Compact control memory stores:
- run summaries,
- key decisions,
- manifests,
- report metadata,
- lineage links.

Large logs/checkpoints/reports stay in `artifacts/` and are referenced by path.

## Finite judge actions
Judge actions are finite and reviewable:
- approve and continue,
- hold for missing evidence,
- rollback,
- restart with bounded plan,
- block promotion.

## Safety rules
Safety guards must explicitly check:
- stage transition validity,
- dataset contract compatibility,
- checkpoint compatibility,
- evaluation gate status,
- promotion/restart policy constraints.

## V1 success condition
A single disciplined cycle can execute end-to-end with explicit typed boundaries, persisted control memory, artifact traceability, and a final judge decision backed by evidence references.
