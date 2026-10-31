# Staged production composition, 45-operation receipt chain and semantic custody

Parent: draft PR #95 (`63babaf4b44d2bb2250d17bcbfe5512b90ede085`).
This work is stacked, draft, and **does not** claim an independently
completed real-weight 45-substep experiment.

## What is now implemented

- `production/staged_composition.py` exposes one explicit assembly path for
  previously separate verified discovery, approved local preprocessing, actual
  guarded PEFT proposal finalization/lifecycle, frozen baseline/candidate
  generation and scoring, and certified role-model wrappers.
- A 45-slot binding inventory checks every operation in the existing
  `PHASE_SUBSTEPS` schedule. Five orchestration-native guard steps are
  labeled as such. Absent or recognizable simulated/test workers are refused;
  approval, independent pinned baseline and recipe allowlists, and model/backend
  implementations must be provided. **This is a wiring check, not a real-model
  or service-behavior proof.** Simple class/module checks cannot authenticate
  the origin of a model's weights or eliminate cleverly disguised fakes.
- `production/staged_trace.py` verifies each of the 45 finished step-result
  receipts against exact content-addressed artifact bytes, workflow/run/
  lineage/stage IDs, unique operation IDs, successful final attempts and
  required decision-critical records. The production cycle now includes a
  `operation_trace_ref` pointing to this immutable trace. The three-cycle
  restart test exercises this on the **existing deterministic scientific
  simulation worker**. That test is explicitly **not** proof of real-weight
  45-operation execution.
- Before a science review can report eligibility, independent holdout custody
  must sign the exact content hash of a semantic-leakage audit scoped to the
  sealed holdout, training/development exposure and diagnostic manifest.
  The audit must attest full inventory coverage, examination of original
  private text, lexical near-duplicate and semantic similarity analysis,
  blinded human review, method provenance, and zero unresolved matches.
  Missing/false/unscoped/outdated audits fail closed. The external-certificate
  verifier recomputes this gate. Synthetic fixture tests use generated keys
  and fabricated scan results, **not** real independent data review.

## Unfinished prerequisites for the requested unsimulated 45-step proof

1. The production-owned implementations for the early scientific roles and
   Judge/recovery substeps still require full staged adapters. This factory
   intentionally **does not** substitute generic catch-all workers for
   Judge entry, evidence collection, diagnosis, readiness, Planner operations,
   runtime governance, recovery and Judge exit.
2. Approved, content-pinned local model/data selection and acquisition still
   need to be exercised inside **the same uninterrupted 45-step campaign**,
   not through a seeded downstream partial fixture.
3. Execute a tiny CPU campaign using actual five-role inference **and**
   genuine PEFT training, generation and comparison, with actual persisted
   launch/checkpoint/model receipts for **every operation**. Refuse test-double
   metadata and record which original weights and inference backend ran.
4. Separately conduct the **real Granite Evaluator** campaign with operator-
   authorized compute, protected frozen settings, fresh exposures and genuine
   independent scientific results. A successful CPU tiny-model test does not
   establish real Granite improvement.
5. An independent human data custodian/auditor must actually perform and sign
   the private-text and embedding similarity audit with auditable method/source
   provenance; attestations and text hashes alone cannot *prove* the absence
   of semantic contamination. Separate external scientific certifier/trust
   roots and an untouched holdout remain mandatory. Any suspicious overlap
   or missing exposure inventory must refuse certification.

## Explicit non-actions

No protected eval pack was modified; no synthetic audit was presented as a real
external finding; no production role was activated; no paid GPU, cloud Pod,
external dataset access, or model training was launched; no pull request merged.
Both training and generation stay opt-in and require existing governance.
