# Integration note: evidence-based diagnosis

## Service

Instantiate `hephaestus.diagnosis.EvidenceBasedDiagnosisService`.

Constructor dependencies:

- `evidence_adapters`: zero or more implementations of `EvidenceAdapter.load(DiagnosisRequest)`;
- `policy`: optional immutable `DiagnosisPolicy` confidence thresholds;
- `explanation_adapter`: optional `ExplanationAdapter` for non-authoritative prose only.

The class conforms at runtime to the shared `hephaestus.interfaces.services.DiagnosisService` protocol. `DiagnosticianRole` is the thin role adapter and calls `service.diagnose(request)`.

## Input and output

Input is the frozen shared `DiagnosisRequest` contract. Output is the frozen shared `DiagnosisReport` contract containing normalized `EvidenceObservation` records, ranked `DiagnosticHypothesis` records, explicit missing evidence, `ContractIssue` records, and bounded confidence.

The final integration branch should construct a request after Judge entry, inject persisted evidence adapters, call the diagnosis service, persist the returned report using existing state boundaries, and pass that report to the planner. This feature branch intentionally does not modify the orchestrator or shared stores.

## Evidence adapters

- `MappingEvidenceAdapter`: fixture/application-owned reference map.
- `JsonReferenceEvidenceAdapter`: root-confined JSON and JSONL artifact references.
- `StateEvidenceAdapter`: read-only loading of matching run, runtime-event, incident, manifest, report, decision, memory, artifact-index, and lineage records from the existing filesystem state layout.

Backend-specific runtime logs, preprocessing reports, launch configs, checkpoint verification, and replay reports should be normalized by integration-owned adapters when they are not already present in those state files. Adapters must return mappings and must not mutate source state.

## Configuration

`DiagnosisPolicy` controls minimum lead confidence, ambiguity margin, contradiction penalty, maximum confidence, and the unverified-eval downstream ceiling. Defaults are conservative and require no external configuration file.

## Failure modes

- malformed inline or adapter evidence becomes a non-crashing `invalid_request` issue;
- adapter read/parse failures become retryable, non-blocking `provider_unavailable` issues;
- absent or weak eval integrity becomes a blocking `missing_evidence` issue and confidence cap;
- unresolved causes return `inconclusive`;
- optional explanation failures add only a metadata warning and never change deterministic findings.

## Test fixtures

`tests/test_evidence_based_diagnosis.py` contains a complete synthetic evidence bundle plus malformed, conflicting, missing-eval, runtime, data, tokenizer/wrapper, numerical, persistence-read, repeatability, protocol, and mutation-safety cases. Tests are filesystem-local and require no network access.

## Known limitations

- Rules consume explicit structured signals; they do not parse free-form logs semantically.
- Confidence is bounded evidence support, not a calibrated causal probability or statistical interval.
- State loading follows the current local JSON/JSONL layout and does not provide database, object-store, or remote-log adapters.
- The optional explanation adapter is deliberately prose-only.
- A model-family limitation requires an explicit controlled-comparison signal; the subsystem does not infer it from weak performance alone.

## Shared contract changes requested

None. The existing `DiagnosisRequest`, `DiagnosisReport`, `EvidenceObservation`, `DiagnosticHypothesis`, `ContractIssue`, and `DiagnosisService` protocol are sufficient.

## Exact integration wiring

1. Build `DiagnosisRequest` from Judge-entry identifiers and explicit failure observations.
2. Construct `StateEvidenceAdapter(state_root)` plus any backend-specific read-only adapters.
3. Construct `EvidenceBasedDiagnosisService(evidence_adapters=[...])`.
4. Call through `DiagnosticianRole.run(request)` or the service protocol.
5. Persist the report without modifying source evidence.
6. Pass the report to the closed-loop planner; do not execute its recommended intervention kinds directly.

## Merge-conflict risk

Low. Changes are confined to the diagnosis-owned package, diagnostician role, diagnosis policy, diagnosis-specific tests/docs, and this integration note. No shared schema, interface, orchestrator, state-store, data, backend, training, evaluation, approval, promotion, lineage, replay, or frozen-eval file is changed.
