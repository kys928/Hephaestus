# Phase II targeted interface repair

Status: **prepared; normalization A/B precedes all weight updates**.

The Phase II V1 run `interface-mastery-v1-35734212862` completed successfully as an
experiment but did not certify the live role chain.  Evidence grounding was perfect,
while downstream exact-contract preservation and schema compliance degraded under
real model-to-model handoffs.  The repair program therefore targets composability,
not broad role retraining.

## Order of operations

1. **Paired handoff-normalization A/B.** Each producer output is generated exactly
   once and reused for both consumer arms.  Arm A receives the current raw projected
   producer output.  Arm B receives a deterministic
   `interface-handoff.normalized.v1` envelope.  No weights are changed.
2. **Controller interface repair.** Repair data stresses Judge-vocabulary copying,
   stale/missing approval, stage-illegal actions, and idempotent replay.
3. **Evaluator containment.** Repair data stresses the distinction between reporting
   scientific evidence and prematurely turning that evidence into governance.
4. **Judge interface repair.** Repair data uses Planner/Evaluator-shaped records that
   contain valid source-role vocabulary but require a different Judge contract.
5. **Phase II-B adversarial recertification.** A fresh frozen holdout, never used for
   training, retests all four interfaces before any dispatch or promotion decision.

## Data separation

Three evidence domains are deliberately separate:

- Phase II V1: the original 32-case failed certification set. It may be used for the
  paired diagnostic normalization A/B, but never as repair training data.
- `hephaestus_interface_repair_v1`: 96 targeted repair samples, 32 each for
  Controller, Evaluator, and Judge. Every semantic sample stores raw and normalized
  presentation so the A/B result can choose the eventual training representation.
- Phase II-B: 32 new adversarial certification cases. These remain evaluation-only
  and must stay unseen by repair training.

Exact IDs are disjoint and validation tests enforce the boundary.

## Normalization boundary

Normalization is deliberately non-authoritative. It may parse and canonicalize a
producer record and label its source role, but it does not translate producer values
into the consumer vocabulary and does not decide what action is legal. The normalized
envelope explicitly marks the record as `untrusted_model_output` and
`schema_copy_allowed=false`.

The deterministic role boundary remains unchanged and remains the final authority for
machine-verifiable execution legality.

## Advancement rule

Repair training must not begin until the normalization A/B result is recorded. If
normalization materially improves downstream exact-contract preservation without
regressing schema validity, evidence grounding, or deterministic-boundary safety, the
repair corpus should be rendered using the normalized presentation. Otherwise repair
should target the raw interface directly.

Phase II-B remains untouched until all selected repair stages have completed.
