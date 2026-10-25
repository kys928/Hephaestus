# Governed Hugging Face staged handoff

## Problem and implementation

The staged pipeline could discover Hugging Face metadata but its data factory
accepted only `local_fixture`. Separate preview reports could therefore never
become verified remote acquisition and preprocessing operations.

`with_discovery_services(dataset_review_policy=..., candidate_review_refs=...)`
now persists source review bytes and their content-addressed references in the
same discovery operation as the candidates. Selection reopens these immutable
records, enforces scientific preview requirements, and incorporates the exact
review digest into the selection identity and required approvals. Replacing a
revision, file scope or review invalidates the selection and prior approvals.
An inconclusive selection retains explicit rejection reasons in its persisted
decision. No policy threshold or original tokenizer identity was relaxed.

`with_huggingface_data_factory` connects the existing
`RemoteDatasetAcquisitionService` to the staged factory. It retains provider
provenance, streams within byte/file limits, checks immutable revisions and
audited hashes, persists plan/receipt/review references, and uses the existing
`process_remote_acquisition` preprocessing entry point. Later operations verify
cached and processed bytes again. This first adapter deliberately supports
**one explicitly named JSON/JSONL/CSV/Parquet training file**. Multi-shard
aggregation is unsupported and must not silently process only the first shard.

Explicit field mapping is available before generic normalization. For the JSON
lead, the instruction, source `data`, and `schema` must all be model inputs;
`output` is the target. The earlier preview mapping omitted `data` and `schema`;
its tokenizer metrics do not establish compatibility of the actual complete
input. SAF must map `question` to `reference_answer`, not student answers or
feedback. Raw text mapping preserves `text` without inventing an EOS boundary.
The configured mapper and processing settings are included in the accepted
preprocessing-profile hash.

## Two independent permissions

1. **Quarantined screening access**: a `DatasetScreeningAccessScope` lists the
   candidate identity, exact training file, byte ceiling, audit purposes and
   protected reference manifests. It has no training permission. A requested
   scope is not an authorized scope. The independent operator must separately
   accept the scope digest; a candidate cannot grant access to itself.
2. **Staged selection/acquisition approval**: after full-source review, the
   native gate obtains a decision for the exact selection identity and every
   requirement, including `dataset_review:<sha256 ref>`. It cannot substitute
   for screening permission, scientific clearance or independent custody.

The remote factory accepts only independently configured review digests and
screening-scope digests. The full-source review must bind the exact file hashes,
frozen tokenizer identity and bytes, normalization and preprocessing profile,
rights evidence, and full-corpus PII, exact/near-duplicate, semantic and frozen
evaluation leakage reports. Reports identify all four training, development,
diagnostic and holdout manifests. Artifact bytes are independently hash checked.
The review gate consumes external scientific attestations; it does **not**
prove their correctness by itself or certify semantic absence of leakage.

Operation 12 performs a delegated, network-free scientific readiness check
before consulting operator approval. Missing clearance stays pending even if an
approval implementation would otherwise return an approved decision. Remote
acquisition is independently disabled by default. The original local route and
phase order remain unchanged.

## Live validation scope and limitations

`run_governed_hf_staged_handoff.py` reads the original SHA-frozen Planner and
diagnosis, verifies the original tokenizer bytes from read-only object storage,
revalidates three shortlisted public sources, and persists actual staged
discovery/selection operation records. It does **not** claim a new broad search.
Imported Planner context is labeled imported; stages 1–9 are not marked complete.
This isolated handoff cannot count as an uninterrupted 45-operation campaign.

Dataset Server preview responses are hashed and their server revision, when
available, is compared with the reviewed repository SHA. Missing or mismatched
revision evidence fails eligibility; a pinned README alone is insufficient.
Only bounded public previews are read. No raw preview rows are persisted.
No full corpus, protected holdout text, training, GPU spending, production
promotion, scientific certification, merge or frozen milestone mutation is
authorized by this integration.

Tests use explicitly synthetic scientific attestations and mocked remote
transport to exercise refusal and actual cache/preprocessing behavior. They do
not establish that any real dataset is scientifically cleared. The live run
must be assessed separately. The existing genuine CPU campaign remains **11/45**
until a new uninterrupted campaign is actually executed.

## Outstanding source decisions

- `vara201/ecommerce-orders-structured-output`: source card declares MIT but
  leaves example origin undocumented and warns about possible customer/payment
  information. Review the complete instruction/data/schema input with the
  original tokenizer. Do not infer source rights or synthetic provenance.
- `Short-Answer-Feedback/saf_communication_networks_english`: the card declares
  CC BY 4.0 and identifies its original ACL 2022 study. Question/reference pairs
  repeat across student-answer records; source row count is not a count of unique
  supervision tasks. Prior preview unknown-token rate exceeded 1%, and explicit
  response-length evidence was sparse. Do not include student answers/feedback
  or fetch validation/test splits under a train-only scope.
- `Salesforce/wikitext`, `wikitext-2-raw-v1` train only: prior filtered preview
  met the tokenizer check; the provider's CC BY-SA 3.0/GFDL metadata and card's
  CC BY-SA 4.0 prose require source-level reconciliation and attribution review.
  No right or frozen holdout access is created by a provider license label.

The live evidence includes concrete **requested**, nonauthorized file/byte
scopes. Protected reference identities and custodial read permissions remain
missing and must be established before any full-corpus screening. Dataset
approval must follow those checks. No candidate should be acquired merely to
advance the operation counter.
