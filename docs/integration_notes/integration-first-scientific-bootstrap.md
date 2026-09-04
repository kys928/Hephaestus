# Integration note: first scientific bootstrap

## Scope

This integration stages the first non-fixture scientific dataset, tokenizer, random-initialized model, and typed experiment evidence onto the RunPod Network Volume `cviwpryzao`. It does not launch a RunPod Pod and does not start training.

## Completed live bootstrap

GitHub Actions run `33853743323` completed successfully against `EU-CZ-1`.

Verified staged identities:

- dataset: `Salesforce/wikitext`
- immutable dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- raw train shard SHA-256: `sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`
- processed dataset SHA-256: `sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`
- tokenizer directory identity: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`
- model directory identity: `sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39`
- random-init model parameters: `1,874,688`
- bootstrap bundle artifact: `sha256:6774e92d2b595353a18211ffa772fb82b362d462ee2e3c26144f705d26525436`

The model forward smoke test passed on CPU with finite logits and shape `[1, 16, 8192]`.

## Typed chain

The staged and parsed typed chain is:

1. `DatasetManifest`: `manifest-first-scientific-bootstrap-001`
2. `TrainableDataContract`: `trainable-data-first-scientific-bootstrap-001`
3. `ModelSelectionDecision`: `model-selection-model-search-b6744896f7e3a16c`
4. `ExperimentProposal`: `experiment-60bff7cb4f478f91`

The experiment remains pending and records `model_selection_approval` as a required approval. The bootstrap launch boundary records `launch_authorized=false`, `runpod_pod_created=false`, and `training_launched=false`.

## Independent paid-volume verification

A separate read-only workflow, run `33854282172`, independently re-read the staged state from `cviwpryzao` and completed successfully.

It verified:

- the bootstrap bundle by full GET and SHA-256;
- raw WikiText bytes: 6,357,543 bytes and the expected SHA-256;
- processed dataset bytes: 15,920,505 bytes and the expected SHA-256;
- 19 referenced evidence objects by content-addressed path, exact size, and full SHA-256 readback;
- both tokenizer components in content-addressed and materialized locations, then rebuilt the tokenizer directory identity;
- all three model components in content-addressed and materialized locations, including the 7,503,792-byte `model.safetensors`, then rebuilt the model directory identity;
- successful parsing and cross-binding of all four typed contracts.

The independent verification artifact is `first-scientific-volume-verification`, GitHub Actions artifact ID `9929543698`.

## RunPod compatibility note

RunPod's Network Volume S3 layer does not preserve the custom SHA-256 object metadata assumption used by generic AWS-oriented storage code. The bootstrap therefore treats the content-addressed key, exact byte count, and a full GET/recomputed SHA-256 as the authoritative verification boundary.

## Next boundary

The first scientific inputs and proposal now exist and are independently verified. A subsequent execution step may bind these staged local identities into the real Transformers lifecycle and launch the bounded experiment according to the existing training/action-boundary policy. This integration itself does not launch paid compute.
