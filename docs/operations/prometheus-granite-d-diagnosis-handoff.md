# Prometheus — recovered D observations into Hephaestus's existing Diagnosis/Planner

**Scope:** read-only CPU preflight of authentic, previously exposed *eight-state historical D* model outputs. This is **not** a six-state measurement of the currently frozen production Granite Evaluator.

## Independent source and what is retained

Run [41609821443](https://github.com/kys928/Hephaestus/actions/runs/41609821443) read and verified the two exact original D S3 sample shards (no arbitrary listing, no writes). Raw generations were independently re-scored using the already merged strict ontology-specific scorer, reproducing **33/64 correct**, exactly **0 per-case disagreements** with the saved scores, and all original by-state counts. The aggregate-only report is stored as GitHub Actions artifact `13217803440` with verified report SHA-256 `7398eeed835422151678e4e1a79ee4c0d8ead4daf9dba61dd4333aa9acfbf953`.

The committed `docs/evidence/prometheus-granite-d/observation-v1.json` is a **non-secret, reduced transcribed observation** tied to that report SHA and its exact source shard hashes. It is *not* a replacement for the raw original and does not certify model bytes, tokenizer provenance or hidden test independence. Its strict loader deliberately refuses causal claims, unauthorized fields, altered metrics, changed ontology and protected material.

## Genuine Hephaestus service handoff

The `granite_d_diagnosis_preflight` module takes this precisely identified aggregate, produces only inert historical observations, and runs the **existing** `EvidenceBasedDiagnosisService` and `ClosedLoopExperimentPlanner`.

It does not manufacture `eval_integrity_verified` or `deterministic_scorecard` signals from a byte hash. It also does not turn the observed 0/8 equivalent and 0/8 recheck-required scores into a predetermined hypothesis about training data or optimizer behavior.

Example, without any S3 keys or paid hardware:

```bash
python -m hephaestus.development.granite_d_diagnosis_preflight \
  --aggregate docs/evidence/prometheus-granite-d/observation-v1.json \
  --output /tmp/granite-d-diagnosis-preflight.json
```

This is a **deterministic Hephaestus preflight**, not five-role specialist inference. Decision-owner metadata records `HEPHAESTUS_DETERMINISTIC_POLICY`, and the result explicitly records zero training, zero promotion, and no scientific repair. The system can legitimately request more original parent/model/scorer evidence; it cannot yet choose a data or optimizer intervention based on a causal diagnosis established by this one archived screen.

## Next scientific gate

To establish autonomous *model-backed* repair:

1. Authenticate historical model/adapter execution, the original certified six-state parent and compatible raw evaluation oracle evidence under distinct task identities.
2. Present the new verified observations and relevant frozen training/dataset lineage to Hephaestus Diagnosis, with explicit missingness.
3. Let Hephaestus's own model-backed Diagnostician/Planner select an admitted intervention, retain its verbatim inference/decision receipts, and compare counterfactual evidence states.
4. Connect a selected action to a genuinely executable governed research cycle only after the required resource authorization.

No final holdout was opened. No S3 secret is present in the active GitHub Actions workflows. No GPU or paid compute is authorized by this preflight.
