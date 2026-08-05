# Integration note: autonomous data factory

## Branch and scope

Branch: `feature/autonomous-data-factory`

This branch implements governed dataset discovery, deterministic selection, and bounded local preprocessing. It does not modify the autonomous-experiment schemas, interfaces, approval policy, orchestrator, training, evaluation, or state stores.

## Service entry points

### Provider registry and discovery

- `hephaestus.data.DatasetProviderRegistry`
- `DatasetProviderRegistry.register(provider)` registers an implementation of the shared `DatasetDiscoveryProvider` protocol.
- `DatasetProviderRegistry.discover(request)` returns `DatasetDiscoveryResult` with normalized candidates, queried provider IDs, and provider failures represented as `ContractIssue` records.

Available providers:

- `FakeDatasetProvider`: deterministic network-free test provider.
- `LocalFixtureDatasetProvider`: discovers explicitly configured JSONL, JSON, or CSV files and records their SHA-256 revision.
- `HuggingFaceDatasetProvider`: optional metadata-only Hub adapter. Network access is disabled by default, and the adapter never imports dataset scripts or enables remote code.

The registry requires an explicit configured allowlist or `DatasetSearchRequest.provider_allowlist`; when both exist, only their intersection runs. A provider discovers candidates; it does not approve or acquire them.

### Selector

- `hephaestus.data.DeterministicDatasetSelectionService`
- `select(request, candidates) -> DatasetSelectionDecision`

Ranking is deterministic for identical inputs. The decision metadata preserves every normalized material candidate and its complete metadata-only audit, including score components, compatibility, cost estimates, missing metadata, uncertainty, preprocessing requirements, and integrity limitations. Unknown licenses, missing provenance, and low trust require explicit approval; incompatible or policy-denied candidates are rejected. No acceptable candidate produces `inconclusive`. If an otherwise acceptable candidate is held only by approval requirements, the outcome is `blocked`.

### Acquisition and preprocessing

- `hephaestus.data.DatasetAcquisitionApproval`
- `hephaestus.data.AutonomousDataPreprocessor`
- `AutonomousDataPreprocessor.process(...) -> DataFactoryResult`

Acquisition validates that the candidate is selected, the approval references the exact decision and candidate, the provider is `local_fixture`, the source is within the byte bound, and the discovered SHA-256 still matches. Selection and approval therefore remain separate from acquisition.

Processing performs bounded schema validation, malformed-row dropping, Unicode/whitespace normalization, optional record filtering, optional named-reference contamination checking, exact deduplication, optional conservative Jaccard near-deduplication, explicit prompt/target wrapper construction, tokenizer compatibility checking through a supplied checker, chunking, and deterministic serialization.

## Configuration

`DataProcessingConfig` requires an explicit `artifact_root` and supports:

- maximum input bytes;
- maximum processed rows;
- chunk size and minimum token count;
- optional near-duplicate threshold;
- maximum record count for the bounded quadratic near-duplicate check;
- explicit prompt/target template.

Optional injected hooks:

- `RecordFilter` with a stable `filter_id`;
- `ContaminationChecker` with a named `reference_set_id`;
- `TokenizerCompatibilityChecker` with stable tokenizer/checker IDs.

When hooks are absent, evidence says `not_checked` or `declared_not_verified`; it never upgrades the integrity claim.

## Artifact-root expectations

Generated data is written under:

`<artifact_root>/<dataset>/<source-revision>/<processing-policy-hash>/<content-hash>/`

The stable content directory contains `trainable.jsonl`. Run-specific records live below `runs/<run-id>/` so two runs do not overwrite each other's manifests or evidence. The processing-policy and content SHA-256 values form the versioned dataset identity.

Generated datasets and artifacts must remain outside Git and must not be committed.

## Contracts consumed

- `DatasetSearchRequest`
- `DatasetCandidate`
- `DatasetSelectionDecision`
- shared `DatasetDiscoveryProvider` and `DatasetSelectionService` protocols

## Contracts produced

- normalized `DatasetCandidate` records;
- `DatasetSelectionDecision` with `selected`, `blocked`, or `inconclusive` status;
- existing `DatasetManifest`;
- existing `PreprocessingReport`;
- existing `TrainableDataContract`;
- a data-owned `processing_evidence.json` sidecar with the detailed integrity record that the deliberately minimal shared preprocessing schema cannot carry.

## Required approvals

Every acquisition requires `DatasetAcquisitionApproval` containing:

- the exact selection decision ID;
- the exact approved candidate ID;
- at least one non-empty approval reference.

The final integration branch must construct this record only from the existing approval/governance flow. A selected candidate alone is not permission to acquire or train on it.

## Error modes

- Provider exceptions become non-secret `provider_unavailable` issues.
- Missing providers and empty discovery become explicit issues.
- Unknown license/provenance and low trust require approval.
- Incompatible formats, model/tokenizer flags, remote code requirements, size constraints, or license policy reject a candidate with reasons.
- Missing approval, source-hash drift, input limits, unsupported local formats, empty output, and failed tokenizer compatibility stop processing.
- A bounded row limit changes the audit claim from `full_scan` to `sampled_audit` and the contamination claim to `partially_checked`.

## Fixtures and tests

- `tests/test_autonomous_data_discovery.py` covers multiple fake candidates, deterministic normalization/ranking, approval blocks, incompatibility rejection, inconclusive selection, and provider failure issues.
- `tests/test_autonomous_data_factory.py` covers approved local processing, malformed rows, exact duplicates, filter and contamination hooks, tokenizer checks, stable hashes/paths, generated contracts, approval enforcement, and honest unchecked statuses.

All focused tests require no network.

## Known missing production features

- Remote acquisition is intentionally absent; the Hugging Face adapter is metadata-only.
- There is no built-in production PII detector or canonical contamination corpus. Only explicit hooks exist.
- The built-in chunker uses deterministic whitespace tokens; production tokenizer-aware packing requires an injected checker or a later data-owned adapter.
- Near-duplicate comparison is conservative and quadratic, suitable only within the configured bounded row count.
- A multi-candidate selection is processed one candidate at a time; final mixture materialization is not implemented.
- Artifact-index/state-store persistence and approval-record lookup are not wired here because those are shared/final-integration responsibilities.

## Exact final-integration wiring

1. Instantiate and allowlist providers, then register them with `DatasetProviderRegistry`.
2. Pass a planner-produced `DatasetSearchRequest` to `discover`.
3. Persist discovery candidates/issues using the integration-owned state path.
4. Pass all material candidates to `DeterministicDatasetSelectionService.select` and persist the decision unchanged.
5. Resolve every required approval through the existing approval flow; construct `DatasetAcquisitionApproval` only from approved records.
6. Instantiate `AutonomousDataPreprocessor` with an external artifact root and production-approved filter/checker adapters.
7. Process each approved selected candidate and persist/index the returned manifest, preprocessing report, trainable-data contract, and evidence references.
8. Give training only the returned `TrainableDataContract`; do not pass a free-form dataset identifier or bypass the approval record.
