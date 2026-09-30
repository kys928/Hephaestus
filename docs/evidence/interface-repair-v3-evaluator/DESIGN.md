# Interface Repair V3 — Evaluator Boundary

Status: implemented for validation; no production promotion
Date: 2026-09-30

## Evidence that triggered V3

Interface Repair V2 produced a valid terminal Evaluator result before the full-stack launcher later hit its cost ceiling. The Evaluator improved substantially but failed certification:

- post-interface quality: `84.140625`
- post-interface exact contract: `0.7578125`
- post-regression quality: `83.75`
- post-regression exact contract: `0.75`
- post-interface schema compliance: `1.0`
- post-interface evidence grounding: `0.9921875`
- semantic escalation: `0.125` on both interface and regression partitions

Shard-level diagnosis showed no recurring decision or primary-variable confusion. The residual failures were two repeated action mappings:

1. `improved` was correctly inferred, but the model proposed `continue_lineage_best` instead of `continue_from_checkpoint`.
2. `equivalent` was correctly inferred, but the model proposed `certify_candidate` instead of `continue_lineage_best`.

The second error accounts exactly for the 16/128 = 12.5% semantic-escalation rate.

## Architectural decision

V3 separates scientific judgment from finite action policy.

The Evaluator model owns:

`verified evidence -> scientific decision + primary variable + uncertainty`

The deterministic Evaluator boundary owns:

`scientific decision + verified machine facts -> role-local effective action`

The model still emits an `action` field during V3 because action-policy agreement is useful research telemetry. That field is advisory only. It cannot override `src/hephaestus/control/evaluator_boundary.py`.

The base projection is:

- `scientific_rejection -> reject_candidate`
- `incomplete_evidence -> request_recheck`
- `improved -> continue_from_checkpoint`
- `regressed -> continue_lineage_best`
- `equivalent -> continue_lineage_best`
- `inconclusive -> hold_candidate`
- `recheck_required -> request_recheck`
- `certification_ready -> certify_candidate`

Verified machine facts retain precedence. A hard-gate failure forces rejection, incomplete evidence forces a recheck, invalid provenance forces a hold, and `certification_ready` cannot produce `certify_candidate` unless deterministic gates, completeness, provenance, and recheck status are all explicitly true.

## V3 curriculum

The V2 curriculum correlated context and semantic class through the same modulo index. V3 removes that shortcut.

The V3 training partition contains 1,536 cases:

- 1,024 fully crossed cases: 16 contexts x 8 scientific states x 8 deterministic wording/order variants;
- 512 focused residual cases that oversample the `improved` and `equivalent` boundaries diagnosed in V2.

Evaluator examples now contain adversarial Planner advisory records. These deliberately recommend plausible but wrong actions, including the exact two V2 confusions. Evidence order and wording vary deterministically. Target rationales are state-specific rather than one repeated generic sentence.

## Holdout hygiene

The V2 Evaluator certification/regression cases have been inspected directly and are therefore diagnostic data, not valid future holdout evidence.

V3 prohibits reusing those cases for certification claims. It uses:

- 128 fresh certification cases across 16 new certification contexts;
- 128 fresh regression cases across 16 different regression contexts;
- context sets disjoint from training and from each other;
- new case IDs and evidence IDs;
- varied evidence wording and order.

The pack validator and CI assert these properties.

## Split certification

V3 reports two independent certifications.

### Model certification

Measures whether the Evaluator infers the scientific state correctly:

- semantic quality;
- semantic exactness (decision + primary variable + confidence band);
- schema compliance;
- evidence grounding;
- hallucination;
- regression preservation.

The model-proposed action agreement rate and model semantic-escalation rate are retained as report-only diagnostics.

### System certification

Measures whether the deterministic boundary makes the system safe and correct:

- boundary projection rate = 1.0;
- system action exactness = 1.0;
- system semantic escalation = 0.

Both model and system certifications must pass for the V3 Evaluator candidate to be certified.

## Artifact lineage

V3 continues from the exact persisted V2 Evaluator adapter because the residual repair is intentionally incremental. That V2 adapter remains explicitly labelled `diagnostic_uncertified_v2_candidate`; V3 does not retroactively certify it.

The V2 Planner candidate is preserved unchanged because it already passed interface and regression certification at 100/100. V3 does not retrain Planner.

The research candidate registry is `configs/models/hephaestus_phase_ii_repair_candidates_v1.json`.

## Governance

- production promotion remains disabled;
- automatic role dispatch remains disabled;
- the Phase-I source registry is not mutated;
- the preserved Planner candidate is not mutated;
- V2 burned holdouts cannot be used to certify V3;
- paid V3 execution requires a separate explicit launch marker after code validation.
