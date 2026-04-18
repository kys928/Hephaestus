# Lineage Policy

## Stage 7 behavior (current)

- Lineage current truth is compact and persisted in `lineage_state.json` plus per-lineage map in `lineage_states.json`.
- Run and decision history remain append-only in JSONL stores.
- Checkpoint truth is explicit and separated:
  - `best_checkpoint_ref` = strongest promoted candidate under bounded evidence,
  - `last_stable_checkpoint_ref` = stricter trusted checkpoint,
  - `certified_stable_checkpoint_ref` = stable checkpoint that passed certification evidence gates,
  - `last_certification_result` = latest bounded certification status.
- Promotion and certification remain conservative:
  - deterministic regression failures block both promotion and certification,
  - incomplete or noisy evidence yields inconclusive certification,
  - stable requires stronger evidence than best,
  - certified stable requires stronger evidence than stable.
- Certification evidence remains bounded and filesystem-backed (no scheduler or external DB):
  - eval-pack bundles define regression/certification checks,
  - minimum evidence and optional recheck consistency can block certification.
- Rollback targets explicit checkpoint refs and defaults to `last_stable_checkpoint_ref`.

## Still simplistic or heuristic

- Stability confidence is still a bounded heuristic (`confidence * evidence_completeness`), not a calibrated reliability model.
- Recheck consistency is local and count-based; it does not yet include long-horizon distribution drift testing.
- Certification status is scoped to current stage policy and eval-pack bundles, not cross-system governance.

## Not yet production-grade

- No signed provenance or tamper-evident audit log.
- No external approval workflow for certification sign-off.
- No cross-cluster replay validation for independent certification confirmation.
