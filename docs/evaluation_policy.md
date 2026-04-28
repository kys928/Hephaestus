# Frozen eval packs and deterministic scorecards

Hephaestus evaluation is based on explicit eval packs and deterministic scorecards, not loose runtime-only output.

## Eval pack policy

- Eval packs are immutable-by-default (`frozen=true`) for persisted runs.
- Eval packs identify prompts/tests/settings used in evaluation.
- Eval packs carry explicit identity (`eval_pack_id`, `version`) and integrity status:
  - `content_hash_verified` only when a `content_hash` exists.
  - `reference_only` when only source reference is available.
  - `inline_unhashed` when inline payload exists without hash.
  - `insufficient` when identity is missing.
- Runtime must not silently rewrite eval packs.

## Deterministic scorecard policy

- Every evaluator pass emits a deterministic scorecard with explicit gate outcomes.
- Scorecards record metrics, thresholds, failed/passed gates, and completeness.
- Deterministic failures remain blocking for promotion/judge trust paths.
- Reward/ranking metrics may supplement review, but cannot erase deterministic failures.
- Missing deterministic evidence lowers confidence/completeness and is recorded as warnings.

## Dry-run behavior

- Dry-run evals may use limited/noisy evidence.
- Dry-run scorecards and eval-pack integrity are marked with explicit limitations and warnings.
- Dry-run reports still include deterministic scorecards and explicit boolean `deterministic_passed`.
