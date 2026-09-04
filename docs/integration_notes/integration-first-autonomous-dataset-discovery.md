# Integration note: first autonomous dataset discovery

## Scope

This integration closes the `dataset_discovery_and_selection` boundary produced by the first targeted post-failure diagnosis. It consumes the exact frozen `DatasetSearchRequest`, discovers real external candidates, evaluates immutable provenance, license evidence, bounded task coverage, and compatibility with the exact frozen tokenizer, runs the existing deterministic selector, and only after a `selected` decision allows Planner to create the next one-primary-variable `ExperimentProposal`.

It does not acquire dataset bytes for training, apply approvals, mutate the Network Volume, or launch training.

## Upstream evidence

- current branch base: `2d285eb8db75a6a310a9ea2a14a5ee534a8bfce0`
- diagnosis: `diag-b6b21afc926ffa31`
- diagnosis domain: `data_coverage`
- diagnosis confidence: `0.825`
- causation claimed: false
- DatasetSearchRequest: `dataset-search-4c7158b167aff959`
- intervention: `intervention-0b05ae5169cd5943`
- primary variable: `dataset_mixture`
- tokenizer directory identity: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`
- exact tokenizer file used for compatibility checks: `sha256:3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25`

## Discovery implementation

`scripts/run_first_autonomous_dataset_discovery.py` uses:

- `DatasetProviderRegistry`
- `HuggingFaceDatasetProvider(enable_network=True)`
- `DeterministicDatasetSelectionService`
- `ResolvedEvidenceExperimentPlanner`

The search vocabulary is deterministically derived from the frozen request's requested instruction/structured/continuation capabilities. Discovery is bounded to 30 unique provider candidates and enrichment to 14 candidates.

For each enrichment candidate the driver:

1. resolves the provider revision to an immutable commit SHA;
2. obtains immutable dataset-card/license/provenance evidence;
3. enumerates data files without executing dataset code or enabling remote code;
4. refuses gated/private/remote-code candidates;
5. performs a bounded public Dataset Server preview;
6. measures instruction/response, structured-response, explicit length-constraint, termination, repetition, and continuation signals;
7. reads the exact training tokenizer from the RunPod Network Volume;
8. tokenizes the preview and measures unknown-token and over-context rates;
9. refuses candidates whose preview unknown-token fraction exceeds 1%; and
10. passes only fully audited candidates to the existing deterministic selector.

The preview is explicitly `bounded_public_preview_heuristic_not_full_dataset_proof`; it is evidence for ranking and compatibility, not a full-data quality claim.

## Live run

GitHub Actions run: `33876486327`

Artifact:

- ID: `9938053158`
- ZIP SHA-256: `60a2adc1aa8ec08e341c645c5e0eda1415e3f1d8a5993f9331479a50065650e6`

The discovery produced 30 unique metadata candidates. Fourteen were subjected to immutable-provenance/content-preview enrichment. Three survived the tokenizer and preview gate and entered deterministic selection. Eleven were rejected before selection because their measured preview unknown-token fraction exceeded the conservative 1% tokenizer threshold.

## Fully audited selection candidates

### `sail/symbolic-instruction-tuning`

- candidate: `dataset-fb91684d87fe5f28`
- license: MIT
- immutable revision: `c0b1111933a7b87bef0e5b3221d8e5f76b5ac27c`
- estimated material bytes: `1,162,784,387`
- source format: JSON
- selector score: `0.60715556`
- preview rows: 100
- instruction-pair coverage: 100%
- bounded-length constraint signal: 8%
- termination signal: 48%
- non-repetition signal: 98%
- structured JSON response signal: 0%
- causal-LM continuation signal: 0%
- tokenizer preview: 17,720 tokens, 0 unknown tokens
- over-context preview fraction: 7.5%

### `FinLang/investopedia-instruction-tuning-dataset`

- candidate: `dataset-f6878fc185019db7`
- license: CC-BY-NC-4.0
- immutable revision: `bbd0e0fc02185b3e8cfd6fba6df04129a4fb3e17`
- estimated material bytes: `307,524,500`
- source format: CSV
- selector score: `0.54248889`
- instruction-pair coverage: 100%
- termination signal: 95%
- non-repetition signal: 98%
- tokenizer preview unknown-token fraction: `0.85572%`

This candidate remained policy-compatible but ranked below the selected candidate.

### `HuggingFaceH4/instruction-pilot-outputs-filtered`

- candidate: `dataset-3f2ba58760c7a779`
- license: Apache-2.0
- immutable revision: `1cd13f88f02f1197783cdfbb6d81da9c3ca56b3a`
- estimated material bytes: `1,872,629`
- selector score: `0.41026667`
- result: rejected below selector threshold `0.45`

## Selection result

- decision: `dataset-selection-fd8699f8cbd8b4957ca2`
- status: `selected`
- selected candidate: `dataset-fb91684d87fe5f28`
- selected dataset: `sail/symbolic-instruction-tuning`
- selection confidence: `0.49357699`
- selector issues: none

The moderate confidence is intentional: provenance/license/tokenizer evidence is strong, but candidate coverage is based on a bounded preview and some provider metadata such as total row count was unavailable.

The selected dataset strongly covers the diagnosed instruction-response gap and is exceptionally compatible with the current tokenizer, but the bounded preview did not demonstrate structured-JSON or continuation examples. Selection therefore tests the `data_coverage` hypothesis; it does not claim to solve every failing semantic dimension.

## Planner handoff

The integration refuses to call Planner unless:

- `DatasetSelectionDecision.status == selected`;
- every selected candidate completed the bounded preview audit; and
- tokenizer compatibility is positively measured.

Only after those conditions were satisfied did Planner create:

- ExperimentProposal: `experiment-d0e911d6bd1fb7ae`
- primary variable: `dataset_mixture`
- dataset selection: `dataset-selection-fd8699f8cbd8b4957ca2`
- status: `pending`
- required approval: `dataset_selection_approval`

The proposal keeps the recorded model, tokenizer, seed, evaluation reference, learning rate, scheduler, warmup, batch size, context length, and step budget controlled while changing the dataset mixture as the single primary variable.

No dataset acquisition, approval application, or training launch occurred.

## Frozen proof

Exact outputs are committed under:

`docs/evidence/first-autonomous-dataset-discovery-001-33876486327/`

Hashes:

- discovery: `5beff6b9f7582a86ad5bcda1f6dbcec3b0fb9a92e0c254cc0c4e89c614353424`
- candidates: `ea9edae30ff261df28e9b3e98a9c8154c699295f9527b99d21c79c6adcfb8aec`
- selection: `f9b379a36106927d8b1d395baf475df072a837b9babbc4acdcd15b101a8d300b`
- experiment: `530032d5bf2ab1b443592fbd8d3bb61616ca11e293a6ca30d327cc020ff73ffc`

A one-shot workflow downloaded the original successful artifact, verified the ZIP and all four exact JSON hashes, committed those exact bytes, and was removed before review.

## Next boundary

The next governed boundary is approval + immutable remote acquisition/preprocessing for the selected dataset. Training must not start until the exact dataset selection is approved, acquisition completes with verified immutable provenance, preprocessing produces a new `TrainableDataContract`, and the pending ExperimentProposal is bound to those data artifacts.
