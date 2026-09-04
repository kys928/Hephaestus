# Integration note: first selected dataset acquisition and preparation

## Scope

Branch: `integration/first-selected-dataset-acquisition`

This integration crosses the first approved post-failure data boundary:

`explicit approval -> immutable acquisition -> bounded preprocessing -> durable DatasetManifest -> TrainableDataContract -> independent Network Volume readback`

It does not launch training, mutate a model, execute the planned experiment, or promote anything.

## Upstream decision

The frozen autonomous discovery/planner evidence selected:

- decision: `dataset-selection-fd8699f8cbd8b4957ca2`
- candidate: `dataset-fb91684d87fe5f28`
- dataset: `sail/symbolic-instruction-tuning`
- license: `mit`
- immutable revision: `c0b1111933a7b87bef0e5b3221d8e5f76b5ac27c`
- planned experiment: `experiment-d0e911d6bd1fb7ae`
- planned run: `planned-run-b8e558e54effac85`
- primary variable: `dataset_mixture`

The operator instruction was recorded as `approval://operator/chat-2026-09-04-selected-dataset-preparation` and covers `dataset_selection_approval` for the selected candidate.

## Immutable acquisition

The live driver re-resolves the provider revision and deterministically chooses the smallest positive-size train data file inside the 512 MiB bound. For this revision that is:

- `train/sql.json`
- `300298688` bytes
- SHA-256 `6312f82b31237031557a621cbf1728159048e1830908088727eb05c407e2cd26`

The provider-declared SHA-256, local transfer SHA-256, artifact-store reference, and full Network Volume readback agree.

RunPod's S3-compatible API did not preserve custom `sha256` user metadata on these objects. Integration verification therefore treats full `GetObject` SHA-256 plus exact byte size as authoritative; metadata is supplemental only.

## Source format evidence

Although the provider path ends in `.json`, the immutable bytes contain newline-delimited JSON objects. The integration adapter handles only the JSON decoder `Extra data` condition and records:

- declared format: `json`
- effective format: `jsonl`
- detection: `json_decoder_extra_data`
- source bytes mutated: `false`

No source bytes are rewritten to make the dataset fit the preprocessor.

## Bounded preprocessing

The preparation intentionally processes at most 100,000 source rows. It records `sampled_audit` and `full_corpus_processed=false` rather than implying full-corpus coverage.

Observed result:

- rows seen: `100000`
- valid rows: `100000`
- malformed rows: `0`
- exact duplicates removed: `5311`
- deduplicated source records: `94689`
- output records after 256-token chunking: `171295`
- processed bytes: `157151627`
- processed SHA-256: `bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`

PII filtering and contamination checking are explicitly `not_checked` in this boundary. Approximate deduplication was not run.

## Wrapper and tokenizer binding

Instruction pairs are normalized into the existing prompt/target contract and wrapped as:

`<|prompt|>\n{prompt}\n<|target|>\n{target}`

The processing evidence binds the exact frozen tokenizer:

`sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`

The exact tokenizer checker measured:

- records: `171295`
- tokens: `62833833`
- unknown tokens: `16427`
- unknown-token fraction: `0.0002614355867801348`
- compatibility: `true`

The independent verifier streamed the durable processed JSONL and observed `171295` trainable records, including `35774` records containing both explicit prompt and target markers.

## DatasetManifest

The durable manifest is:

- ID: `manifest-planned-run-b8e558e54effac85`
- SHA-256: `0495018a0cc7c70494d5a00bc51a471568e850d8e3fa11cb0696c9674c71cc76`
- runtime key: `hephaestus/scientific/v1/runtime_bindings/planned-run-b8e558e54effac85/dataset/dataset_manifest.json`

It binds the selected immutable revision, MIT license, raw source hash, processed hash, tokenizer identity, wrapper policy, chunking policy, sampled-audit scope, and run identity.

## TrainableDataContract

The new contract is:

- ID: `trainable-data-planned-run-b8e558e54effac85`
- run ID: `planned-run-b8e558e54effac85`
- manifest ID: `manifest-planned-run-b8e558e54effac85`
- schema: `trainable-data.v1`
- SHA-256: `ef273fe913f582289ffad2cd05a431e9d541091a51db97b0a649eb47579f2a5a`
- processed dataset ref: `s3://cviwpryzao/hephaestus/scientific/v1/objects/sha256/ba/bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`

The verifier re-read the persisted contract and manifest independently and rejected any disagreement between the contract manifest ID, run ID, processed dataset ref, durable processed object, tokenizer evidence, or proof payload.

## Independent readback

GitHub Actions run `33881924136` completed successfully. The separate verifier full-read and SHA-256 checked:

- raw source shard
- processed trainable JSONL
- DatasetManifest
- TrainableDataContract
- preprocessing report
- processing evidence
- acquisition receipt
- approval record

The verifier ended with `status=verified`.

Frozen repo proof is under `docs/evidence/first-selected-dataset-preparation-001-33881924136/`.

## Control boundary

The successful evidence explicitly records:

- `training_launched=false`
- `model_mutated=false`
- `experiment_executed=false`

The next subsystem may consume this TrainableDataContract, but this integration does not authorize training by itself.
