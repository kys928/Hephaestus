# Scientific-state assessment implementation (CPU-only, not certification)

This module is a **supplemental scientific evaluator**. It cannot certify a
model, grant promotion, issue an approval, or change an active role adapter.

## Starting point

Built on draft PR #88 (commit `12d46a4e0d95a48a9479ba040ce6a5eeeeb98aaa`).
The interrupted Sol session's two locally edited files were not published and
were not available to recover.

## Interface

`hephaestus.evaluation.scientific_state_certification.assess_scientific_states`
consumes:
- A **content-addressed** case-pack manifest, externally pinned by policy.
- Exact baseline/candidate prediction artifacts for the same samples and seeds.
- An immutable generation-settings artifact.
- A diagnostic-case identity manifest used to detect **known** overlap.
- Explicit numerical scoring thresholds supplied by the independent caller.

It enforces exact case coverage, two-member contrast groups, distinct repeated
seeds, constant checkpoint and decoding identities, per-state correctness,
counterfactual pair correctness and zero protected regressions.

The output is a content-addressed `scientific-state-assessment.v1` receipt with
complete input references and recomputed metrics.

## What this does NOT prove

- Declaring `split=sealed_holdout` in a JSON file does **not** establish that
  the samples were genuinely withheld from training or model selection.
- The diagnostic manifest can detect overlap with **listed** cases only.
- Thresholds supplied by a caller are not a substitute for approved frozen
  scientific thresholds in the stage policy.
- Successful CPU fixture scoring does not demonstrate a real Evaluator
  improvement or approved scientific certification.
- The evaluator currently does not consume arbitrary model generation transcripts
  directly: the caller must provide a validated, immutable state-prediction
  extraction artifact.
- Role activation and rollback are **not** implemented here. In particular, the
  existing `promote_checkpoint` action authorizes lineage metadata changes,
  **not active model routing**.

Therefore even a numerical pass emits
`certification_state=not_certified` and
`holdout_unseen_provenance_independently_verified=false`.
Only `eligible_for_external_certification_review=true` may be set.

## Next integration boundaries

1. Pin an externally approved sealed holdout and scoring profile via the
   existing stage-policy/certification owner, without altering frozen packs.
2. Adapt independently validated generation receipts to
   `scientific-state-predictions.v1`, including extraction schema/lineage evidence.
3. Connect this score to the existing independent science gates, without
   substituting it for `ActionAuthorizer`, `PromotionPolicy`, or approval records.
4. Design and implement **distinct** high-risk operator authorization for
   `activate_role_adapter` and `rollback_active_role` before changing live
   role routing; preserve current registry's `automatic_role_dispatch_enabled=false`.
5. Build a real campaign driver through all 45 staged operations. Neither this
   assessment nor PR #88 by itself demonstrates a multi-cycle campaign.

No RunPod pods were launched, no adapters were activated and no production
registry or protected scientific benchmark was changed.
