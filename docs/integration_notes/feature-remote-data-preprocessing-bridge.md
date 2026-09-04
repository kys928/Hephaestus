# Integration note: verified remote acquisition to preprocessing bridge

## Scope

Branch: `feature/remote-data-preprocessing-bridge`

This change closes the data-owned handoff between the existing production remote-acquisition service and the existing autonomous preprocessor. It does not alter shared autonomous-experiment contracts, approval policy, planner behavior, training, evaluation, or the central orchestrator.

## Why this bridge exists

`RemoteDatasetAcquisitionService` already resolves immutable provider revisions, performs bounded transfer, verifies provider/transport/local hashes, records approvals, and emits a completed `AcquisitionReceipt`. The preprocessor previously accepted only `local_fixture` candidates through `acquire_approved_local_candidate()`. Final integration therefore had no honest way to pass a real remotely acquired dataset into preprocessing without relabeling its provenance.

The bridge keeps acquisition and preprocessing separate while making their handoff explicit and verifiable.

## Public behavior

`validate_remote_acquisition_for_preprocessing(...)` validates:

- the exact selected candidate and selection decision;
- the exact dataset-acquisition approval and approval references;
- completed receipt status;
- candidate, provider, and dataset identity;
- requested and immutable resolved revisions;
- receipt-selected file identity;
- local cache-file existence and byte count;
- recomputed SHA-256 against the acquisition receipt;
- artifact-store SHA-256 agreement when present;
- an allowlisted preprocessing record format.

It returns the existing `LocalAcquisition` transport record only after all checks pass. This does not change provider provenance: the original `DatasetCandidate` continues through preprocessing and into `DatasetManifest` metadata.

`AutonomousDataPreprocessor.process_remote_acquisition(...)` accepts the completed receipt as a separate argument, invokes the verifier, then enters the same normalization/filtering/deduplication/chunking/contract-building path used by approved local data.

Processing evidence records the acquisition receipt ID, plan ID, provider, requested/resolved revision, cache status, and immutable artifact references.

## Parquet support

The preprocessor now accepts Parquet as a bounded input format. `pyarrow` remains optional and is imported only when a Parquet file is actually processed. Absence produces an explicit capability error rather than silently changing formats.

Parquet loading uses metadata plus bounded batches and stops at `DataProcessingConfig.max_rows`. A truncated bounded audit remains explicitly marked as sampled rather than complete.

## Manifest provenance

When a discovered candidate did not yet have an `artifact_ref`, the dataset manifest now uses the verified acquisition `source_content_hash` as the immutable source reference. The manifest still records the original provider ID and immutable revision; it never rewrites a Hugging Face candidate as `local_fixture`.

## Validation

New tests cover:

- successful JSONL remote-receipt preprocessing with provider provenance preserved;
- post-acquisition byte/hash drift rejection;
- candidate/revision drift rejection;
- bounded Parquet preprocessing when the optional dependency is installed.

Existing local-fixture processing remains available through `AutonomousDataPreprocessor.process(...)` unchanged at its public boundary.

## Next integration step

The first-scientific-bootstrap integration may now:

1. discover and select a real dataset;
2. plan and explicitly approve immutable remote acquisition;
3. acquire and stage the selected bytes in content-addressed storage;
4. pass the completed receipt into `process_remote_acquisition(...)`;
5. stage the resulting `DatasetManifest`, `TrainableDataContract`, processed JSONL, and processing evidence;
6. keep those data artifacts fixed while model selection becomes the first experiment's primary variable.

No real training is authorized by this bridge.
