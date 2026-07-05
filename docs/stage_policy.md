# Stage Policy

Stage policy defines per-stage evaluation strictness, eval-pack identity, deterministic gates, allowed actions, and certification requirements. It is loaded from named configuration by `StagePolicy` and returned as a `StageProfile` schema.

## Required stage-profile fields

Every stage profile must provide:

- `strictness`: policy label used by evaluation interpretation.
- `eval_pack`: named eval pack to load for the evaluator.
- `deterministic_gates`: object containing both `min_probe_score` and `max_toxicity`.

If these fields are missing, or if deterministic gates are not an object, stage resolution fails with a configuration error.

## Optional fields

A stage profile may also define:

- `name`: explicit display/record name; defaults to the requested stage name.
- `allowed_next_actions`: list of action names allowed from the stage.
- `certification_profile`: certification and repeatability policy.
- `eval_pack_ref`: external eval-pack reference; defaults to `eval_pack`.
- `required_evidence`: integer evidence requirements keyed by purpose.
- `stage_thresholds`: numeric confidence thresholds.
- `deterministic_gate_config`: explicit gate config; defaults to `deterministic_gates`.

`allowed_next_actions` must be a list. `certification_profile`, `required_evidence`, and `stage_thresholds` must be objects.

## Certification profile semantics

The loader normalizes certification settings into these fields:

- `eligibility`: defaults to `standard`; values such as `disabled`, `none`, or `ineligible` prevent certification.
- `require_recheck`: whether certification requires recheck evidence.
- `min_consistent_runs`: integer at least 1.
- `repeatability_required`: whether repeatability is a blocking requirement.
- `required_rechecks`: integer at least 0.
- `min_repeat_consistency`: float between 0.0 and 1.0.
- `variance_sensitivity`: one of `low`, `medium`, or `high`.
- `certification_recheck_policy`: one of `always`, `never`, `required_if_repeatability_unmet`, or `required_if_variance`.

Invalid values fail during stage resolution rather than later in the run.

## Deterministic gates

The implemented evaluator treats `min_probe_score` and `max_toxicity` as required deterministic gates. A checkpoint cannot be promoted through deterministic trust paths if deterministic gates fail. Reward, ranking, or qualitative signals may provide context, but they cannot erase deterministic failures.

## Policy boundary

Stage policy lives in `src/hephaestus/policy/stage_policy.py`. Roles and backends may read a resolved `StageProfile`, but they must not silently invent stage gates or certification thresholds. Any new cross-component stage fields must be added to `StageProfile` before use.
