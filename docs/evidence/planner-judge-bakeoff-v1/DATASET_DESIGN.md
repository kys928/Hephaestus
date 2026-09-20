# Planner/Judge Bakeoff V1 dataset design

This source-defined curriculum is deterministic and frozen by canonical SHA-256. Generated dataset artifacts are not committed.

Each role has eight independent capability families. Every family contains four zero-shot variants, ten micro-LoRA training variants, and six untouched held-out variants.

Per role:
- zero-shot: 32 cases
- micro-LoRA train: 80 cases
- held-out: 48 cases
- total: 160 cases

Across Planner + Judge: 320 deterministic cases.

## Planner capability families

P1 missing-evidence restraint; P2 one-primary-variable discipline; P3 dead-end memory and model admission; P4 evaluation-integrity precedence; P5 bounded undertraining continuation; P6 checkpoint-integrity rollback; P7 governed dataset-mixture change; P8 approval-preserving tokenizer planning.

The Planner is rewarded for proposing controlled, reversible, evidence-backed interventions. It is penalized for executing actions, inventing causality, repeating known dead ends, changing several primary variables, ignoring admission/approval boundaries, or training through invalid evaluation.

## Judge capability families

J1 hard-gate supremacy; J2 missing promotion approval; J3 valid promotion approval; J4 incomplete-evidence blocking; J5 immutable provenance; J6 rollback after repeated failure; J7 repeatability/variance restraint; J8 stage-policy action boundaries.

The Judge is rewarded for finite policy decisions, not creativity. Aggregate score, stakeholder pressure, or approval cannot override deterministic gates, provenance, evidence completeness, or stage policy.

## Anti-overfitting structure

Case IDs and evidence refs are split-specific and disjoint. Held-out scenarios use different lifecycle contexts from the training set. The same underlying decision boundary is exercised with changed numbers, distractors, stage names, stakeholder requests, and evidence ordering.

The global legal contract vocabulary is exposed to models at runtime; case-specific expected values are never exposed.
