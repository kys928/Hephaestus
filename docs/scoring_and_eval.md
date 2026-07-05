# Scoring and Evaluation

Scoring is the deterministic substrate beneath evaluation and promotion. It provides small, auditable checks over scalar metrics and aggregates those checks into reports consumed by the evaluator and judge policy.

## Deterministic metrics

The implemented deterministic path focuses on two required scalar metrics:

- `probe_score`: higher is better and must meet `min_probe_score`.
- `toxicity`: lower is better and must not exceed `max_toxicity`.

Stage policy supplies thresholds. Eval packs declare required metrics and regression bundles. The evaluator reads metrics from referenced artifacts or inline scalar fallback values and records limitations when evidence is missing.

## Gate semantics

A gate is deterministic when the same metric payload and threshold configuration always produce the same result. Deterministic gate failures are blocking for promotion trust paths. Non-deterministic or reward-style signals may be recorded as supporting context but cannot cancel a deterministic failure.

## Regression bundles

Regression summaries group deterministic checks into named bundles, such as promotion or certification bundles. A bundle passes only when all required checks are present and passed. Missing checks are explicit failures for that bundle.

## Scorecards

The evaluator emits a `Scorecard` for every eval report. Scorecards preserve metrics, thresholds, per-gate results, pass/fail gate lists, evidence refs, eval-pack identity, warnings, and completeness. Scorecards enforce these invariants:

- Any failed gate forces `deterministic_passed=false`.
- Missing metrics, thresholds, gate results, or deterministic pass state lower completeness.
- Missing eval-pack identity lowers integrity and adds warnings.
- Optional ranking/continuation fields may be absent, but absence lowers completeness.

## Promotion interface

Promotion policy consumes deterministic pass/fail, confidence, evidence completeness, bundle status, certification readiness, repeatability sufficiency, variance risk, and stage thresholds. Scoring does not promote checkpoints; it only supplies auditable evidence.
