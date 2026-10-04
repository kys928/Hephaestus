# Branch Record

- Branch: `diagnostic/evaluator-sealed-v1-autopsy`
- Canonical label: `diagnostic/2026-09-evaluator-sealed-autopsy-v1`
- Category: diagnostic / evaluator certification
- Lifecycle: CURRENT RESEARCH LINEAGE
- Original tip before this metadata commit: `18613e00a798ddd98bf3c4050fe92ebf408fb3c8`
- Recorded: 2026-10-04
- Main integration policy: do not merge this branch wholesale. Only validated implementation changes, selected model configuration, and final scientific outcomes should later be integrated into `main` after the current model-selection and fine-tuning cycle is complete.

## Purpose

Preserve the evaluator-sealed-v1 autopsy and the follow-on diagnostic/repair work around evaluator certification behavior.

## Orientation

This is a research branch, not the stable product branch. The exact experiment sequence, evidence, failures, and conclusions remain authoritative in this branch's commits and evidence files; this record intentionally does not rewrite those results.

## Relationship

This lineage descends from the Phase II interface-mastery work and diverged from the targeted-interface-repair line. It contains substantial unique work and must not be treated as a disposable temporary branch.

## Naming convention

Future branches of this kind should use: `diagnostic/<YYYY-MM>-<subject>-v<N>`.
