# Lineage Policy

## Stage 9 behavior (current)

- Lineage current truth is compact and persisted in `lineage_state.json` plus per-lineage map in `lineage_states.json`.
- Run and decision history remain append-only in JSONL stores.
- Checkpoint truth is explicit and separated:
  - `best_checkpoint_ref` = strongest promoted candidate under bounded evidence,
  - `last_stable_checkpoint_ref` = stricter trusted checkpoint,
  - `certified_stable_checkpoint_ref` = stable checkpoint that passed certification evidence gates,
  - `last_certification_result` = latest bounded certification status.
- Repeatability-aware certification now exists as enforced policy, not report-only labels:
  - eval reports carry compact repeated-evidence fields (`repeated_eval_count`, `consistency_score`, `variance_risk`, `repeatability_sufficient`),
  - stage profiles and eval packs can require rechecks and minimum consistency,
  - certification can return `certification_recheck_required`, `certification_inconclusive_due_to_variance`, or `certification_blocked_by_inconsistency`.
- Promotion and certification remain conservative and separated:
  - deterministic regression failures still block both promotion and certification,
  - best/stable/certified_stable remain distinct,
  - stable can progress while certified_stable remains blocked/deferred by repeatability policy.
- Recheck realism remains bounded and local (filesystem-only):
  - recent certification attempts are queryable by checkpoint,
  - repeatability summaries are derived from append-only decision history,
  - no scheduler, queue, vector DB, or distributed orchestrator is introduced.
- Operator governance for high-impact lineage actions is now explicit and bounded:
  - approval requirements are policy/config driven (`configs/policies/approval_policy.yaml`),
  - high-impact requests persist as explicit approval request records,
  - operator decisions persist as explicit approval decision records (`approved`, `rejected`, `expired`, `superseded`, plus explicit override outcomes),
  - lineage current state remains compact while tracking only compact governance truth (`pending_approval`, `last_approval_status`, `last_high_impact_request_id`),
  - high-impact transitions do not auto-execute while approval is pending or rejected.

## Still simplistic or heuristic

- Variance risk currently uses bounded spread thresholds over available probe scores; this is not a full statistical variance estimator.
- Consistency scoring is deterministic pass-rate based and bounded to local evidence windows.
- Override behavior is intentionally bounded to explicit per-request outcomes and does not provide unrestricted admin bypass.

## Not yet production-grade

- No signed provenance or tamper-evident audit log.
- No authenticated external identity/authorization provider for operator approvals.
- No independent multi-cluster replay certification.
- No formal statistical confidence intervals or power analysis for repeatability claims.
