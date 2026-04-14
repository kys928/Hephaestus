# Lineage Policy

## Stage 6 behavior (current)

- Lineage current truth is compact and persisted in `lineage_state.json` plus per-lineage map in `lineage_states.json`.
- Run and decision history remain append-only in JSONL stores.
- `best_checkpoint_ref` and `last_stable_checkpoint_ref` are tracked separately.
- Promotion is conservative:
  - deterministic regression failure blocks promotion,
  - low-confidence results stay candidate/inconclusive,
  - stable requires stronger confidence than best.
- Rollback targets explicit checkpoint refs and defaults to `last_stable_checkpoint_ref`.
- Branching creates a new lineage identity with parent linkage and branch origin checkpoint.
- Restart is explicit and auditable via lineage status/pathology updates.

## Still simplistic

- Stable checkpoint evidence is confidence-threshold based and does not yet aggregate long-horizon replay evidence.
- Rollback selection is policy-light (stable-first), not yet stage-specialized.
- Branch naming and restart semantics are deterministic but basic.

## Not yet production-grade

- No signed provenance or tamper-evident audit log.
- No external policy engine for governance approvals.
- No multi-operator conflict resolution for lineage actions.
