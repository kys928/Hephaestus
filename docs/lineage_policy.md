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

## First-class lineage record contract

Lineage is the durable model-family record. `RunRecord` describes one run. `DecisionRecord` describes one decision. `LineageState` describes evolving family truth across runs.

Canonical lineage status values:
- `exploratory`
- `promising`
- `stable`
- `suspect`
- `poisoned`
- `archived`
- `deprecated`
- `blocked`

Canonical trust levels:
- `unknown`
- `low`
- `medium`
- `high`
- `certified`

Compatibility note: older persisted state can still contain legacy statuses (`active`, `degraded`, `unstable`, `restarted`). Lineage loading remains backward compatible and normalizes missing fields by schema defaults.

Persisted lineage records now include first-class fields for:
- identity and parent/child linkage,
- origin and timestamps,
- contracts (`architecture_contract_ref`, `tokenizer_contract_ref`, `data_policy_ref`, `training_recipe_ref`, `eval_policy_ref`),
- run/checkpoint tracking,
- decision and governance tracking,
- reliability/repeatability signals,
- failure memory (`recent_failures`, `known_pathologies`),
- `major_interventions`,
- structured `metadata`.

A lineage may be continued, branched, rolled back, restarted, archived, deprecated, or marked poisoned. Trust level is not identical to status. Certified stability must not be inherited by branches unless explicitly revalidated.

## Still simplistic or heuristic

- Variance risk currently uses bounded spread thresholds over available probe scores; this is not a full statistical variance estimator.
- Consistency scoring is deterministic pass-rate based and bounded to local evidence windows.
- Override behavior is intentionally bounded to explicit per-request outcomes and does not provide unrestricted admin bypass.

## Not yet production-grade

- Stage 10 does not yet provide subject-level ProvenanceRecord persistence. Current provenance is limited to manifests, artifact refs, decision records, run records, and artifact index entries.
- Checkpoint replay guarantees are reference-only unless checkpoint content hashes are explicitly recorded as evidence.
- No signed provenance or tamper-evident audit log.
- No authenticated external identity/authorization provider for operator approvals.
- No independent multi-cluster replay certification.
- No formal statistical confidence intervals or power analysis for repeatability claims.
