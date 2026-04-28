# Autonomy Boundary Policy

Hephaestus autonomy is **finite-action autonomy**, not freeform agency.

## Core principle

Before any judge-exit action can execute, it is classified against a finite registry:

1. `auto_allowed`
2. `approval_required`
3. `high_risk_approval_required`
4. `forbidden`

Unknown actions are not auto-allowed.

## Boundary behavior

- Every requested action is evaluated by the action registry boundary evaluator.
- Forbidden actions are blocked regardless of confidence or operator preference.
- Unknown actions are treated as non-auto actions and blocked from silent execution.
- Approval-required actions must have approval metadata before they can be executed.
- High-risk actions always require approval.

## Protected surfaces

The boundary policy explicitly protects critical integrity surfaces, including:

- frozen eval packs,
- run history,
- decision history,
- deterministic retrieval memory provenance,
- unapproved datasets.

These protections are represented as forbidden actions and cannot be bypassed by ordinary action flow.

## Decision traceability

Judge-exit decisions persist boundary classification metadata:

- `action_boundary`
- `action_category`
- `action_forbidden`
- `action_requires_approval`
- `action_high_risk`

This makes action gating auditable in persisted decision records.

## Scope limitation for this phase

This policy introduces only finite action definitions and boundary checks.

It does **not** implement:

- autonomous code editing,
- broad planner autonomy expansion,
- self-healing execution,
- reward-model judging,
- mutation of training data or eval packs.

Code editing remains deferred to a later constrained protocol.
