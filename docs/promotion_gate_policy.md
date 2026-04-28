# Promotion Gate Policy

Hephaestus Judge recommendations are **advisory** until promotion gates evaluate evidence and safety constraints.

## Core principle

- `JudgeExitRole` still proposes `next_action`.
- `evaluate_promotion_gates(...)` converts persisted evidence into executable constraints.
- The orchestrator persists a `promotion_gate_report` in `DecisionRecord.metadata`.
- `effective_action` may differ from `requested_action` when gates or approval block execution.

## Gate categories

1. Deterministic scorecard gate
   - Missing deterministic scorecard or `deterministic_passed=False` blocks promotion-like actions.
2. Eval pack integrity gate
   - `insufficient` integrity blocks promotion-like actions.
   - `reference_only` / `inline_unhashed` lower confidence ceiling.
3. Scorecard integrity gate
   - `insufficient` integrity blocks promotion-like actions.
   - weaker integrity lowers confidence ceiling.
4. Data manifest gate
   - Missing manifests do not crash execution but lower confidence.
   - `insufficient` manifest integrity blocks direct `promote_checkpoint`.
5. Checkpoint candidate gate
   - Promotion-like actions require a candidate checkpoint reference.
   - Rollback requires stable or best checkpoint target.
6. Lineage trust/status gate
   - `poisoned`, `deprecated`, `archived`, `suspect`, and `blocked` statuses block promotion-like actions.
7. Approval gate
   - Existing approval policy remains authoritative for high-impact actions.
   - Pending/rejected/expired/superseded approval states block high-impact execution.
8. Repeatability/variance gate
   - unstable repeatability or high variance blocks stable/certified promotion paths.

## Safety outcomes

- Promotion updates are prevented whenever blocking promotion gates fail.
- Fallback actions are constrained to existing action vocabulary (`continue_lineage_best`, `reject_checkpoint`, `branch_new_experiment`).
- Branch creation does not inherit certified stability.
- Gates constrain execution only; they do not introduce new autonomous planning behavior.
