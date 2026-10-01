# V3.1 Evaluator burned-case readiness audit

Status: diagnostic evidence only; not certification.

Source run: `v3-neutral-holdout-36766350227`

Frozen adapter SHA-256: `913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed`

Neutral pack SHA-256: `0bae520eda2b59f9317baacfd9d77a24f51f0b7c4a2b5e7e057e0a8cee2d7492`

Persisted audit: `hephaestus/scientific/v3/neutral_holdout/v3-neutral-holdout-36766350227/readiness-audit-v1.json`

## Scope

The 128-case neutral holdout is burned diagnostic material. This audit performs no new inference, no training, no adapter mutation, no certification claim, and no production promotion. It inspects the already-persisted case rows and reconstructs their paired source cases.

## Residual failure inventory

Six of 128 cases contain at least one residual scoring failure.

- 3 scientific-state decision failures.
- 4 primary-variable failures; 2 overlap state failures and 2 are primary-only.
- 1 evidence-reference grounding failure.
- 1 effective-system-action failure; it overlaps a scientific-state failure.
- 0 schema failures.
- 0 confidence-band failures.
- 0 upstream-copy failures.
- 0 system semantic-escalation failures.

## Scientific-state failures

### Scientific rejection -> recheck required

The evidence states that a non-negotiable deterministic gate is reproducibly false. The model predicts `recheck_required` instead of `scientific_rejection`. The deterministic boundary correctly observes `deterministic_gate_passed=false` and forces `reject_candidate`, so the model-state error is contained at the system-action layer.

### Scientific rejection -> regressed

The evidence states that one mandatory deterministic gate remains failed across repeats. The model predicts `regressed` instead of `scientific_rejection`. The deterministic boundary again forces `reject_candidate`; the state error is contained.

### Certification ready -> equivalent

All frozen gates pass, evidence completeness is 1.0, provenance is valid, the certification recheck is satisfied, and repeats are consistent. The model predicts `equivalent` instead of `certification_ready`. Because the current boundary is deliberately conservative and does not positively promote a non-certification state into `certify_candidate`, the error propagates to `continue_lineage_best`. This is the only residual effective-system-action miss and is a genuine model-state readiness issue, not a deterministic-boundary implementation bug.

## Primary-variable failures

Four rows use the wrong `primary_variable`:

- scientific rejection -> `evaluation_integrity` instead of `hard_gate_status`;
- scientific rejection -> `candidate_regression` instead of `hard_gate_status`;
- incomplete evidence -> `evaluation_integrity` instead of `runtime_evidence` while the decision itself is correct;
- recheck required -> `runtime_evidence` instead of `variance_risk` while the decision itself is correct.

The ontology currently makes the authoritative primary variable a deterministic function of the scientific state. These failures therefore strengthen the case for moving authoritative `primary_variable` projection into deterministic code while retaining the model-proposed value as telemetry. That would be an architecture/protocol change and is not performed by this audit.

## Evidence-reference failure

One otherwise fully correct `equivalent` case cites the four exact allowed opaque reference suffixes but drops the literal `E-` prefix from every reference. Example: allowed `E-D03D57CB6D7F`, model emits `D03D57CB6D7F`.

The model did not invent unrelated evidence and did not recall the old pre-neutralized regression identifiers. The failure is an identifier-canonicalization/contract-format error. Under the existing scorer it correctly counts as ungrounded/hallucinated because the emitted strings are not exact allowed references. A deterministic unique-reference canonicalizer could contain this without changing scientific-state inference, but that would be a protocol/runtime change and is not performed by this audit.

## Readiness conclusion

The frozen V3.1 adapter should remain the selected parent and should not be retrained merely because of the burned neutral result.

The residuals decompose into:

1. two scientific-rejection state errors that are already safely contained by the deterministic hard-gate boundary;
2. one genuine `certification_ready -> equivalent` false negative that is not contained and must be challenged explicitly in future fresh evidence;
3. four primary-variable output errors whose authoritative meaning can be projected deterministically from scientific state;
4. one evidence-reference formatting error caused solely by omission of the `E-` prefix.

Before a fresh sealed certification run, the recommended architecture decision is to consider deterministic authoritative projection of `primary_variable` and strict unique evidence-reference canonicalization, while keeping positive `certification_ready` inference model-owned and refusing to auto-promote a non-certification model decision.

Any such architecture/protocol changes require explicit approval, must be frozen before certification, and must be evaluated on fresh sealed certification and regression evidence. The burned neutral holdout may never be reused for certification.
