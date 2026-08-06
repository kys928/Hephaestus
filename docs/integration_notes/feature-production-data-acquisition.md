# Integration note: production dataset acquisition

## Branch and baseline

- Branch: `feature/production-data-acquisition`
- Immutable branch base: `552200b95b027ed91ac76167619ac9478100e811`
- Scope: remote dataset acquisition only. Shared autonomous-experiment schemas, interfaces, policy, storage implementations, preprocessing, training, evaluation, and the orchestrator remain unchanged.

This branch extends the approved local-fixture path with a separately auditable remote path. Metadata discovery, deterministic selection, acquisition planning, policy approval, transfer, and preprocessing remain distinct operations.

## Public entry points

Import from `hephaestus.data`:

- `RemoteDatasetAcquisitionService`
- `RemoteAcquisitionLimits`
- `DatasetAcquisitionCache`
- `RemoteAcquisitionPlan`
- `AcquisitionReceipt`

Provider-side records and protocols are exported from `hephaestus.providers.datasets`:

- `RemoteDatasetAcquisitionProvider`
- `ProviderDatasetSnapshot`
- `ProviderDatasetFile`
- `HuggingFaceDatasetProvider`
- `HuggingFaceApiClient`

`RemoteDatasetAcquisitionService.plan(...)` consumes a selected `DatasetCandidate`, the exact `DatasetSelectionDecision`, explicit limits, and an optional `SecretReference`. It resolves provider evidence and produces a deterministic plan. It does not transfer files or authorize acquisition.

`RemoteDatasetAcquisitionService.acquire(...)` consumes that plan and a `DatasetAcquisitionApproval`. Transfer starts only when the approval references the exact selection decision and candidate, has at least one approval reference, and explicitly covers every policy requirement recorded by the plan.

## Provider implementation

The production adapter supports Hugging Face dataset repositories through read-only Hub metadata and immutable resolve URLs.

Provider behavior is split into:

1. `search()` for metadata discovery;
2. `resolve_revision()` for requested-reference to commit-SHA resolution;
3. `enumerate_files()` for file metadata at the resolved commit;
4. `revision_is_current()` for pre-transfer identity validation;
5. generic data-owned planning and transfer services for selection, approval, download, cache, and receipts.

The core implementation uses the Python standard library. `huggingface_hub` and `datasets` are not required. No import-time network access occurs, and network use remains disabled by default unless the provider is explicitly enabled or an injected client is supplied.

## Immutable revision behavior

Floating requests such as `main`, a tag, or another branch are retained as `requested_revision` but are never final provenance. The provider must resolve them to a full 40- or 64-character commit SHA. The resolved SHA is used for:

- file enumeration;
- source URLs;
- dataset-card reference;
- cache identity;
- partial-state identity;
- receipt provenance.

If a full immutable revision cannot be resolved, planning is blocked with `immutable_revision_unresolved`. Before online acquisition, the provider re-resolves the requested reference. A change produces `revision_changed`, blocks transfer, and removes partial state tied to the superseded plan. Offline reuse does not contact the provider but still requires a verified cache entry for the exact immutable revision.

## File and remote-code policy

Default data suffixes are JSONL, JSON, CSV, Parquet, Arrow, text, gzip, zstd, and ZIP. Callers may narrow or replace this tuple in `RemoteAcquisitionLimits`.

Provider paths are validated as non-empty POSIX-relative paths. Absolute paths, traversal components, backslashes, and malformed paths are rejected before filesystem access.

Python files, shell scripts, binaries, and dynamic-library files are never selected as dataset material. A candidate or provider snapshot marked `remote_code_required` is rejected with `unsupported_remote_code`. This branch does not import dataset repositories, execute builders, dynamically load modules, invoke `datasets.load_dataset`, or enable `trust_remote_code`.

## License, card, and provenance evidence

Hugging Face snapshots record:

- dataset ID and provider;
- requested and resolved revisions;
- immutable dataset-card reference and revision;
- license and license source;
- usage terms where exposed;
- citation and authors where exposed;
- gated/private status;
- provenance confidence;
- missing metadata.

An unknown license adds a named approval requirement to the acquisition plan. It is not interpreted as permission. `DatasetAcquisitionApproval.approved_requirements` must include every such named requirement before transfer. Gated/private plans additionally require an injected authentication reference.

## Authentication

Authentication is represented only by the existing `SecretReference` and resolved at runtime through an injected `SecretsProvider`.

Plans and receipts persist only the provider/key reference and non-secret status. Secret values are not fields of provider snapshots, plans, cache metadata, receipts, issues, or URLs. Unit tests prove that the runtime token reaches the provider/transport but does not appear in serialized evidence.

## Bounded transfer and cancellation

`RemoteAcquisitionLimits` provides:

- maximum total bytes;
- maximum file count;
- streaming chunk size;
- per-request timeout;
- disk-space reserve;
- allowed data suffixes.

Known sizes are checked during planning. Unknown or incorrect provider sizes remain bounded during chunked transfer. The transfer engine never reads the full remote response into memory. A disk-space preflight runs when provider size metadata exists. Cancellation is cooperative and checked before each file and each streamed chunk.

HTTP/authentication, gated-access, missing-file, timeout, rate-limit, provider-unavailable, connection-interruption, size-budget, disk-space, checksum, revision, remote-code, and malformed-metadata failures become structured `ContractIssue` records.

## Resume and partial state

Partial bytes and metadata are stored separately under the configured cache root. Partial metadata binds:

- provider;
- dataset ID;
- immutable revision;
- relative file path and source URL;
- provider hash;
- ETag when exposed;
- exact persisted byte count.

When range resume is supported, `Content-Range`, requested offset, immutable revision, and ETag are validated before appending. If a provider ignores a valid range request and returns a full response, the local partial file is truncated and restarted from zero. Offset, ETag, identity, or revision mismatches fail safely and invalidate the partial. A connection interruption or cancellation preserves partial state only when it remains bound to the immutable source identity. Partial files are never reported as complete.

## Content-addressed cache and offline reuse

Cache keys include provider, dataset ID, immutable revision, file path, provider object ID, and provider hash. Completed bytes are stored under local SHA-256 identity and never mutated in place.

Every cache hit recomputes the local hash and validates byte size before reuse. Corrupt content is rejected and moved to a quarantine path so failure evidence is preserved. Online mode may then reacquire it. Offline mode fails with `offline_artifact_missing` unless every required file has a verified complete cache entry for the exact plan.

## Checksum and artifact-store behavior

The receipt distinguishes:

- provider-declared hash and algorithm;
- provider-hash validation status;
- transport checksum and validation status;
- locally computed SHA-256;
- artifact-store reference and content hash;
- cache key, status, and reference.

Hugging Face LFS SHA-256 and Git blob SHA-1 identities are validated when present. The local SHA-256 is always computed. Supported transport SHA-256 headers are verified when exposed; absence is recorded as missing evidence rather than silently upgraded.

An injected `ArtifactStore` receives completed verified files through its existing `put_file` protocol. The returned object is verified through `ArtifactStore.verify()` before its reference enters the receipt. Storage implementations are unchanged by this branch.

## Receipt and rollback behavior

`AcquisitionReceipt` is a JSON-safe data-owned record containing:

- deterministic receipt ID;
- plan, selection, approval, candidate, provider, and revision identities;
- acquired file evidence and byte totals;
- cache status;
- dataset-card/license evidence;
- transfer attempts and resume evidence;
- artifact references;
- warnings, missing evidence, issues, and cleanup evidence;
- explicit `completed`, `partial`, `failed`, or `cancelled` status.

Observational start/completion timestamps are stored under `observations` and are excluded from deterministic receipt identity. Identical completed evidence therefore yields the same receipt ID.

Failed attempts remove only their owned invalid temporary files. Safe resumable partials may be preserved. Verified immutable cache objects and artifact-store objects are never deleted merely because a later attempt fails. Proven-corrupt cache objects are quarantined rather than silently removed.

## Tests

Required tests use injected providers/transports and local temporary directories. They do not contact public networks.

```bash
PYTHONPATH=src pytest -q \
  tests/test_production_data_acquisition.py \
  tests/test_huggingface_dataset_acquisition.py

PYTHONPATH=src pytest -q \
  tests/test_autonomous_data_discovery.py \
  tests/test_autonomous_data_factory.py \
  tests/test_autonomous_experiment_integration.py
```

Coverage includes immutable resolution, deterministic IDs, explicit approval, named policy approvals, secret references, bounded streaming, byte/file limits, valid resume, restart without range support, invalid resume identity, revision drift, provider and transport hashes, cache integrity, offline reuse/miss, corrupt-cache quarantine, remote-code refusal, card/license evidence, path traversal, artifact-store references, receipt round trips, and provider failure normalization.

## Known limitations

- Hugging Face is the only production provider adapter in this branch.
- The standard-library adapter does not implement provider-specific retry/backoff orchestration; retryable issues are surfaced to the caller.
- Provider file-size and hash evidence may be absent for non-LFS files. Local SHA-256 remains authoritative for acquired bytes, and missing provider evidence remains explicit.
- ZIP/gzip/zstd files are acquired as immutable source material; archive extraction and decompression remain preprocessing responsibilities and must retain their own safety bounds.
- The existing local `FileSystemArtifactStore.put_file()` buffers a file while storing it. The acquisition transfer itself is streaming; a production large-object store should provide a streaming implementation of the unchanged `ArtifactStore` boundary.
- Approval-record lookup and persistence remain final-integration responsibilities. This branch validates the supplied approval record but does not query shared state stores.
- Acquisition receipts are returned to the caller; central state/index persistence remains final-integration wiring.

## Final integration wiring

1. Instantiate and allowlist `HuggingFaceDatasetProvider(enable_network=True)` in the data-provider registry for metadata discovery.
2. Use the existing deterministic/approval-aware selector and persist its decision unchanged.
3. Construct `RemoteDatasetAcquisitionService` with explicit cache root, transport, optional `SecretsProvider`, and optional `ArtifactStore`.
4. Call `plan()` for the selected candidate and persist the deterministic plan.
5. Resolve every `plan.required_approvals` item through existing governance, then construct `DatasetAcquisitionApproval` with exact decision/candidate IDs, approval references, and `approved_requirements`.
6. Call `acquire()` and persist/index the receipt. Do not continue when its status is not `completed`.
7. Pass verified acquired cache/artifact references into the existing preprocessing boundary. Do not make remote acquisition itself preprocess or authorize training.
8. Preserve the existing coordinator/control-spine order: discovery and selection, approval, acquisition, preprocessing, training preparation.

No shared contract amendment is required for this branch. A future shared amendment may introduce a first-class acquisition-plan/receipt protocol if other subsystems need these data-owned records directly; until then, integration should persist them as explicit JSON-safe evidence without modifying `autonomous-experiment.v1`.
