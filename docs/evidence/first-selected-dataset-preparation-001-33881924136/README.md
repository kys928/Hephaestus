# First selected dataset preparation proof

Exact proof from GitHub Actions run `33881924136` at commit `dcb0fe729f281737403e1efcfd07653372f78ec3`, artifact `9940513064`.

## Boundary proven

`explicit approval -> immutable acquisition -> bounded preprocessing -> durable DatasetManifest -> TrainableDataContract -> independent Network Volume readback`

No training or model mutation occurred.

## Selected immutable source

- selection decision: `dataset-selection-fd8699f8cbd8b4957ca2`
- candidate: `dataset-fb91684d87fe5f28`
- dataset: `sail/symbolic-instruction-tuning`
- license: `mit`
- immutable revision: `c0b1111933a7b87bef0e5b3221d8e5f76b5ac27c`
- bounded shard: `train/sql.json`
- shard bytes: `300298688`
- raw SHA-256: `sha256:6312f82b31237031557a621cbf1728159048e1830908088727eb05c407e2cd26`
- approval ref: `approval://operator/chat-2026-09-04-selected-dataset-preparation`

The provider filename ends in `.json`, but the immutable bytes contain newline-delimited JSON objects. The preparation records this explicitly as `declared_record_format=json`, `effective_record_format=jsonl`, `detection=json_decoder_extra_data`, and `source_bytes_mutated=false`.

## Bounded preprocessing result

- rows inspected: `100000`
- valid rows: `100000`
- malformed rows: `0`
- exact duplicates removed: `5311`
- source records after deduplication: `94689`
- output trainable records after 256-token chunking: `171295`
- processed bytes: `157151627`
- processed SHA-256: `sha256:bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`
- prompt/target-wrapped records independently observed: `35774`

The audit is intentionally `sampled_audit`: preprocessing was bounded to 100,000 source rows and does **not** claim the entire upstream corpus was processed. PII filtering and contamination checking were not run in this boundary.

## Tokenizer and wrapper binding

- tokenizer identity: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`
- tokenizer check: compatible
- records checked: `171295`
- tokens checked: `62833833`
- unknown-token fraction: `0.0002614355867801348`
- wrapper: `<|prompt|>\n{prompt}\n<|target|>\n{target}`
- prompt marker: `<|prompt|>`
- target marker: `<|target|>`

## Durable manifest and contract

DatasetManifest:
- manifest ID: `manifest-planned-run-b8e558e54effac85`
- durable SHA-256: `sha256:0495018a0cc7c70494d5a00bc51a471568e850d8e3fa11cb0696c9674c71cc76`
- runtime key: `hephaestus/scientific/v1/runtime_bindings/planned-run-b8e558e54effac85/dataset/dataset_manifest.json`

TrainableDataContract:
- contract ID: `trainable-data-planned-run-b8e558e54effac85`
- run ID: `planned-run-b8e558e54effac85`
- manifest ID: `manifest-planned-run-b8e558e54effac85`
- schema version: `trainable-data.v1`
- durable SHA-256: `sha256:ef273fe913f582289ffad2cd05a431e9d541091a51db97b0a649eb47579f2a5a`
- processed dataset ref: `s3://cviwpryzao/hephaestus/scientific/v1/objects/sha256/ba/bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`
- runtime key: `hephaestus/scientific/v1/runtime_bindings/planned-run-b8e558e54effac85/dataset/trainable_data_contract.json`

## Independent readback

A separate verifier re-read the raw shard, processed dataset, DatasetManifest, TrainableDataContract, preprocessing report, processing evidence, acquisition receipt, and approval from Network Volume `cviwpryzao`. Every accepted artifact was verified by full `GetObject` SHA-256 plus exact byte size; verification does not depend on optional S3 custom metadata.

Final verifier status: `verified`.

Control boundary remained intact:
- `training_launched=false`
- `model_mutated=false`
- `experiment_executed=false`
