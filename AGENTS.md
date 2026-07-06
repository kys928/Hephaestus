# AGENTS.md — Hephaestus Operator Rules

## Required reading

Before making architectural changes, read:

- `PROJECT_MISSION.md`
- `docs/architecture.md`
- `docs/control_spine.md`
- `docs/stage_policy.md`
- `docs/evaluation_policy.md`

## Mandatory control spine (non-negotiable)

All workflow execution must preserve this exact phase order and boundary ownership:

1. Judge (entry)
2. Planner
3. Data Acquisition & Audit
4. Data Preprocessor
5. Training Engineer
6. Runtime Monitor
7. Evaluator
8. Judge (exit)

Do not collapse these roles into a monolithic orchestrator.

## Architectural rules

- Keep role logic in `src/hephaestus/roles/` and coordination logic in `src/hephaestus/control/`.
- Persist decision-critical records through `src/hephaestus/state/` only.
- Keep heavy evidence/artifacts as path references; do not inline them in control memory records.
- Every cross-component payload must use an explicit schema in `src/hephaestus/schemas/`.
- Policy decisions must remain in `src/hephaestus/policy/`.

## Safety and governance rules

- Do not launch real training unless the task explicitly asks for it.
- Do not mutate eval packs without explicit approval.
- Do not apply patches directly from the operator console.
- Do not bypass approval governance.
- Do not weaken replay verification.
- Do not silently rewrite loader/checkpoint behavior.
- Do not commit generated state, artifacts, datasets, checkpoints, model weights, caches, or secrets.

## Bootstrap expectations

- Prefer typed stubs and TODO markers over fake-complete implementations.
- Keep files small, explicit, and inspectable.
- Use deterministic, JSON-serializable structures for all persisted records.
- Prefer narrow, testable, PR-sized changes over broad rewrites.
- If evidence is weak, say the result is inconclusive instead of pretending certainty.
