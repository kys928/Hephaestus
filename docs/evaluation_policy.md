# Evaluation Policy

Hephaestus evaluation is based on explicit eval packs, deterministic scorecards, checkpoint resolution, and stage-aware certification signals. Evaluation summarizes evidence and gate outcomes; it does not directly mutate lineage or approve promotion.

## Eval pack policy

Eval packs are loaded by name from configuration and normalized through the `EvalPack` schema. Persisted eval packs are immutable-by-default:

- `EvalPackStore` forces persisted packs to `frozen=true` if necessary.
- Missing mutation policy defaults to `immutable_without_approval`.
- Runtime must not silently rewrite eval packs for an existing persisted run.

Eval packs carry explicit identity and integrity fields:

- `eval_pack_id`
- `version`
- `content_hash`
- `hash_type`
- `source_ref`
- `integrity_level`

Integrity status is interpreted as:

- `content_hash_verified` when a content hash exists and identity is complete.
- `reference_only` when only an external source reference is available.
- `inline_unhashed` when inline payload exists without content hash.
- `insufficient` or incomplete identity when the pack cannot identify evidence reliably.

Eval-pack loaders reject unknown fields and validate required metrics, regression bundles, certification bundle configuration, evidence minima, repeatability requirements, and stage tolerances.

## Required metrics and deterministic gates

The implemented deterministic evaluation path expects `probe_score` and `toxicity`. Stage profiles require these gates:

- `probe_score >= min_probe_score`
- `toxicity <= max_toxicity`

Metric evidence is read from an intermediate artifact reference when available. If referenced metrics are missing but inline scalar metrics exist, evaluation can use the inline fallback. If neither valid referenced nor inline metrics exist, evaluation records missing-evidence limitations and lowers confidence instead of inventing values.

## Regression summaries

`build_regression_summary()` runs deterministic checks over metrics and gate thresholds. It records:

- whether deterministic checks passed,
- failed check names,
- notes/details,
- per-bundle pass/fail results,
- missing checks within requested bundles.

Regression bundle failures are blocking for promotion trust paths.

## Checkpoint selection

Checkpoint selection is deterministic:

- No candidates returns an empty checkpoint ref with reason `missing_checkpoints`.
- One candidate returns that candidate with reason `single_checkpoint` unless already specified.
- Multiple candidates require all candidates to include `probe_score`; otherwise selection is inconclusive with reason `inconclusive_multiple_checkpoints`.
- When all candidates are scored, the candidate with the highest `probe_score` is selected with reason `best_probe_score` unless already specified.

## Scorecard semantics

Every evaluator pass emits a deterministic scorecard. A scorecard records:

- eval-pack identity and integrity,
- checkpoint reference,
- deterministic pass/fail boolean,
- failed and passed gates,
- scalar metrics and thresholds,
- per-gate result objects,
- optional structural/repetition/continuation/ranking booleans,
- evidence references,
- warnings,
- completeness score,
- metadata.

`Scorecard.enforce_semantics()` ensures any failed gate forces `deterministic_passed=false`, records missing required scorecard fields, computes completeness, and downgrades integrity when eval-pack identity is incomplete. Missing deterministic gate results are recorded as warnings.

## Certification and repeatability

The evaluator combines eval-pack requirements and stage certification profile to produce certification signals:

- `certification_readiness`
- `recheck_recommended`
- `observed_consistent_runs`
- `repeated_eval_count`
- `consistency_score`
- `repeatability_ready`
- `repeatability_blocked`
- `repeatability_sufficient`
- `recheck_needed`
- `variance_risk`
- `stability_confidence`

These signals are inputs to promotion policy. They are not by themselves permission to promote.

## Dry-run behavior

Dry-run evaluations may use synthetic or limited evidence. The evaluator must still emit an eval report, deterministic scorecard, explicit warnings where evidence is missing, and a boolean deterministic result. Dry-run limitations reduce confidence and completeness rather than being hidden.

## Boundary invariants

- Deterministic failures are blocking for promotion and judge trust paths.
- Reward/ranking metrics can supplement review but cannot override deterministic failures.
- Eval reports and scorecards contain short summaries and evidence references, not full artifacts.
- Evaluation does not update lineage directly; judge exit and promotion policy consume evaluation outputs to determine the effective action.
