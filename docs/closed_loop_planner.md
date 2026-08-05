# Closed-loop experiment planner

The closed-loop planner converts a persisted `DiagnosisReport` into deterministic, ranked `InterventionProposal` records. It proposes actions but never executes training, data changes, lineage transitions, evaluation mutation, or discovery-provider calls.

## Planning rules

- Exactly one primary variable is named for every intervention. All other variables are held constant or explicitly documented as unavoidable.
- Missing evidence blocks training-like interventions and raises a diagnostic `collect_more_evidence` proposal.
- Inconclusive or blocked diagnoses cannot be reused to create discovery or training work.
- Dataset discovery is emitted only for `replace_or_mix_dataset`; model discovery is emitted only for `change_model`.
- A selected dataset or model decision is required before the corresponding experiment can be proposed.
- Every selection decision must reference the exact deterministic search request and contain no blocking issue.
- Experiments require a baseline reference or an explicit baseline justification.
- Known dead ends are rejected unless the diagnosis identifies evidence not present in the dead-end record.
- Poisoned, deprecated, or archived lineages may produce conservative evidence, branch, restart, or stop recommendations, but not continuation work.
- Frozen evaluation, deterministic scorecards, and baseline comparison remain required evidence.
- Unsafe or incomplete proposal construction raises `ExperimentPlanningError`; no experiment is executed or silently repaired.
- Contract status remains `pending`; approval requirements and the planner's non-authoritative approval state are preserved separately.

## Ranking semantics

Ranking combines diagnosis support, information gain, expected benefit, bounded compute/data/storage/evaluation/time cost, risk, reversibility, evidence completeness, baseline quality, prior attempts, stage suitability, and approval burden.

Every numeric component is a bounded heuristic used for deterministic ordering. It is not a calibrated probability, measured cloud quote, or guarantee of improvement. The basis and estimate confidence are preserved in proposal metadata.

## Conservative outcomes

The planner may recommend evidence collection, evaluation repair, data repair, dataset replacement or mixture changes, preprocessing/tokenizer/training/model changes, resume, rollback, branch, restart, or stop. The service refuses to turn `stop` into an experiment. High-impact lineage actions retain the existing approval-policy result.
