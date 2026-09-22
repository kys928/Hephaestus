# Phase II targeted interface repair

Status: **normalization A/B running; repair training remains gated**.

The Phase II V1 run `interface-mastery-v1-35734212862` completed as an experiment but did not certify the live role chain. Evidence grounding was perfect, while downstream exact-contract preservation and schema compliance degraded under real model-to-model handoffs. The repair program therefore targets composability, not broad role retraining.

## Order of operations

1. **Paired handoff-normalization A/B.** Each producer output is generated exactly once and reused for both consumer arms. Arm A receives the raw projected producer output; Arm B receives a deterministic `interface-handoff.normalized.v1` envelope. No weights change.
2. **Controller interface repair.** Continue the certified Controller adapter on targeted source-role contamination examples.
3. **Evaluator containment.** Continue the certified Evaluator adapter on evidence-vs-governance role-boundary examples.
4. **Judge interface repair.** Continue the certified Judge adapter on real Planner/Evaluator-shaped handoffs and governance traps.
5. **Phase II-B adversarial recertification.** Assemble an evaluation-only stack from the three repaired adapters plus unchanged Phase I Diagnosis/Planner adapters and run a fresh 32-case holdout.

## Data separation

Three evidence domains are deliberately separate:

- **Phase II V1:** the original 32-case failed certification set. It may be used for the paired diagnostic normalization A/B, but never as repair training data.
- **`hephaestus_interface_repair_v1`: 120 targeted repair samples.** Each of Controller, Evaluator, and Judge receives 32 train cases and 8 held-back repair-dev cases: 96 train + 24 dev total. Every sample stores raw and normalized presentations so the A/B result selects the representation without relabeling data.
- **Phase II-B:** 32 new adversarial certification cases, eight per interface. They are evaluation-only and are not read by repair training.

Canonical repair-data SHA-256: `58cce9a9a4b43ac4874a50af86f6c6080f50d541d56dfd343d01539e2821335f`.

Canonical Phase II-B SHA-256: `e71f02a1965e53d1c157e5c6f1d3adba23216560569325c25b18661967b1a9ed`.

## Normalization boundary

Normalization is non-authoritative. It parses and canonicalizes a producer record and labels its source role, but never translates producer values into consumer vocabulary and never decides which action is legal. The envelope explicitly marks `authority=untrusted_model_output` and `schema_copy_allowed=false`.

The deterministic role boundary remains unchanged and remains the final authority for machine-verifiable execution legality.

## Repair selection boundary

Repair training cannot begin until the paired A/B result is completed and frozen into the repair config. Controller, Evaluator, and Judge are repaired **sequentially**. Each begins from its certified Phase I selected LoRA adapter, trains only on its 32 repair-train examples, and is selected only on its eight repair-dev examples. Phase II V1 and Phase II-B are never checkpoint-selection data.

The repair-dev gate requires quality >= 90, schema compliance = 100%, grounding = 100%, exact-contract pass rate >= 87.5%, and zero hallucinated evidence references.

## Recertification boundary

Phase II-B remains inaccessible to repair training. Only after all three selected repair adapters pass repair-dev does the aggregate become eligible for Phase II-B stack assembly. Diagnosis and Planner remain unchanged Phase I adapters. Automatic model-backed dispatch and production promotion remain disabled regardless of the Phase II-B result; any later rollout is a separate governed decision.
