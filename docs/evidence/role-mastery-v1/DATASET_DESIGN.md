# Phase I Role Mastery V1

Five frozen foundation models are trained independently. No model may compensate for another role.

Each role receives 1,024 deterministic role-specific training cases and 128 untouched certification cases across eight capability families. Total: 5,120 training cases plus 640 certification cases.

Certification cases are never used for gradient updates.

Controller covers exact execution, approvals, stale state, lineage/checkpoint integrity, idempotency, budgets, artifact integrity and no-op semantics.

Diagnosis covers causal restraint, tokenizer compatibility, evaluation integrity, data coverage, wrapper serialization, undertraining, checkpoint integrity and model-family limitations.

Planner covers evidence restraint, one-variable interventions, dead-end memory, evaluation repair, bounded continuation, rollback, governed data change and approval-preserving tokenizer planning.

Evaluator covers hard gates, evidence completeness, genuine improvement, regression, variance/repeatability, provenance, equivalence and certification readiness.

Judge covers hard gates, approval boundaries, valid promotion, incomplete evidence, provenance, rollback, variance and stage-policy boundaries.
