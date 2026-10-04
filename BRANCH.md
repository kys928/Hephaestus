# Branch Record

- Branch: `phase-ii-targeted-interface-repair`
- Canonical label: `phase/2026-09-targeted-interface-repair-v1`
- Category: Phase II / targeted repair and recertification
- Lifecycle: OPEN DRAFT / ACTIVE PHASE-II LINEAGE
- Original tip before this metadata commit: `93b656dbba5c278247732203e83578bb2be65918`
- Recorded: 2026-10-04
- Related PR: #69

## Purpose

Carry the evidence-backed Phase II recovery path after interface-mastery testing exposed cross-role contamination under real model-to-model handoffs: normalization A/B, targeted repair, and adversarial recertification.

## Relationship

This branch descends from `phase-ii-interface-mastery`. The evaluator-autopsy research line later diverged from the same broader Phase II work, so neither branch should be assumed to supersede the other without evidence.

## Main integration policy

Do not merge wholesale while model-selection and fine-tuning tests remain active. After the cycle completes, integrate only the validated implementation, selected model configuration, certification/evaluation outcome, and appropriate evidence into `main`.

## Naming convention

Future phase branches: `phase/<YYYY-MM>-<objective>-v<N>`.
