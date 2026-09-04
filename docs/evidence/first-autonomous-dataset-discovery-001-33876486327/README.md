# First autonomous dataset discovery proof

Exact frozen outputs from GitHub Actions run `33876486327`, artifact `9938053158`.

Upstream:
- diagnosis: `diag-b6b21afc926ffa31` (`data_coverage`, confidence `0.825`)
- DatasetSearchRequest: `dataset-search-4c7158b167aff959`
- intervention: `intervention-0b05ae5169cd5943`
- one primary variable: `dataset_mixture`
- tokenizer: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`

Result:
- Hugging Face metadata candidates discovered: `30`
- fully audited candidates surviving immutable provenance and tokenizer gates: `3`
- selected dataset: `sail/symbolic-instruction-tuning`
- selected immutable revision: `c0b1111933a7b87bef0e5b3221d8e5f76b5ac27c`
- selected license: `mit`
- selected candidate score: `0.60715556`
- selection decision: `dataset-selection-fd8699f8cbd8b4957ca2`
- selection confidence: `0.49357699`
- selected tokenizer preview: 17,720 tokens, 0 unknown tokens
- next ExperimentProposal: `experiment-d0e911d6bd1fb7ae`
- experiment primary variable: `dataset_mixture`
- experiment status: `pending`
- required approval: `dataset_selection_approval`

Planner was not called until the selection status was `selected` and the selected candidate had completed immutable-provenance, bounded coverage, and tokenizer compatibility checks.

No dataset was acquired, no approval was applied, and no training was launched.
