# Control spine, role authority and decision gates

## Non-negotiable phase order

The canonical eight phases are **Judge entry → Planner → Data acquisition and audit → Data preprocessor → Training engineer → Runtime monitor → Evaluator → Judge exit**. Orchestration belongs to [control code](../../src/hephaestus/control/); domain services live in [production](../../src/hephaestus/production/). The staged native operation trace used in CUDA45 does not replace this order.

| Phase | Primary responsibility | Cannot silently do |
| --- | --- | --- |
| Judge entry | Assess admissibility, identity and existing gates | Invent an approval |
| Planner | Diagnose bottleneck and choose bounded intervention | Treat an unapproved resource as selected |
| Data acquisition & audit | Candidate provenance, rights, contamination and source identity | Admit unknown data by prompt text alone |
| Preprocessor | Produce versioned materialization and trainable-data contracts | Mutate frozen holdout |
| Training engineer | Configure actual optimizer, resource, checkpoint and data usage | Increase paid budget or swap GPU without a new decision |
| Runtime monitor | Record events, failures, bounded retry and recovery | Turn an incident into an unlimited retry |
| Evaluator | Compare fixed evidence, metrics, probes and regressions | Override deterministic failures with reward score |
| Judge exit | Recommend accept/reject/branch/hold/rollback | Self-certify or execute an ineligible mutation |

## Who has authority?

A language model can hypothesize about causal explanations, interpret ambiguous observations, choose among proposals and explain confidence. It **cannot** authenticate a checksum, assert evidence existence, mint an approval, change lineage ownership, decide whether an action idempotency key is spent or promote itself.

The [deterministic role boundary](../deterministic_role_boundary.md) verifies source references and legality before a role receives trusted facts, then re-checks the proposed action. The [finite action registry policy](../autonomy_boundary_policy.md) classifies actions as auto-allowed, approval-required, high-risk-approval-required or forbidden. Unknown actions do not receive implicit approval.

The [evaluation policy](../evaluation_policy.md), [promotion gates](../promotion_gate_policy.md) and [replay verification](../replay_verification_policy.md) remain separate from a model's natural-language conclusion. A blocked deterministic gate stays blocked even if every model role agrees otherwise.

## Original five frozen CUDA roles

R10 independently accepted **Controller, Diagnosis, Planner, Evaluator and Judge** model inferences using the pinned role stack, with immutable receipt identities. The Evaluator's selected Granite adapter was fine-tuned **before** R10; R10's **96 optimizer steps trained the separate target model**. One accepted Evaluator inference is not a universal reliability certificate.

Per the [R10 source report](../evidence/original-cuda45-r10/README.md), automatic global role dispatch and checkpoint promotion were not authorized by that result. Any future operational change must pass a separate explicit policy and evidence gate.

## Change invariant

Do not collapse phase boundaries, bypass typed [schemas](../../src/hephaestus/schemas/), mutate protected evaluation data, trust role-generated approval strings, or silently change the model/tokenizer/config contract. Weak evidence must be represented as `inconclusive`, not promoted to certainty.
