# Evidence-based diagnosis

The diagnosis subsystem turns persisted or explicitly supplied structured evidence into conservative, ranked failure hypotheses. It does not execute interventions.

## Evidence model

`EvidenceBasedDiagnosisService` consumes the shared `DiagnosisRequest`. Inline `observed_failures` records and records resolved by injected adapters are normalized deterministically. Supported adapters include an in-memory mapping adapter, a root-confined JSON/JSONL reference adapter, and a read-only existing-state adapter.

Each normalized record becomes an `EvidenceObservation`. Rules then produce `DiagnosticHypothesis` records with separate supporting evidence, contradicting evidence, required tests, intervention kinds, and bounded confidence. Report metadata labels these statement types explicitly and states that no causal claim or intervention execution occurred.

## Confidence behavior

- Confidence comes from recorded evidence quality and independent source references, not agent agreement.
- Contradicting evidence lowers hypothesis confidence and remains visible.
- Missing or unverified eval-pack and deterministic-scorecard integrity caps downstream causal-domain confidence at `0.35`.
- Direct runtime incidents may still identify a runtime failure while evaluation evidence is missing.
- Close or weak candidates produce `inconclusive`.

## Hard-evidence rules

Manifest incompleteness is missing/integrity evidence, not proof of poor data quality. Contamination, tokenizer incompatibility, wrapper mismatch, architecture mismatch, checkpoint corruption, numerical instability, and model-family limitation require explicit structured signals. Correlation is never converted into causation.

The finite failure domains and recommended intervention kinds come from `autonomous-experiment.v1`. Recommendations are labels for later planning; they are not commands.

## Optional explanation

An injected `ExplanationAdapter` may add non-authoritative prose to report metadata. The deterministic observations, hypotheses, confidence, issues, and missing-evidence fields remain authoritative. Diagnosis correctness and tests do not require an LLM or network access.
