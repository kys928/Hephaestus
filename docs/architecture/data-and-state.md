# Data, state, lineage and evidence custody

## Data flow and admission

A diagnosed capability problem may lead to dataset/model discovery. Discovery returns **candidates**, not approval. Governed selection considers rights, provenance, compatibility, source identity, contamination risk, completeness and missing metadata before a data source can enter an experiment.

An admitted run binds its dataset versions/hashes, row counts, mixture, filters, preprocessing/chunking, tokenizer and prompt/target wrapper into a manifest and trainable-data contract. The [data manifest policy](../data_manifest_policy.md) and [autonomous experiment contract](../autonomous_experiment_loop.md) define the expected recorded boundaries.

The held-out evaluation pack stays frozen and segregated. A successful tokenizer match or small preview never proves dataset license clearance or absence of semantic leakage.

## Persistent truth vs. large artifacts

- [`src/hephaestus/state/`](../../src/hephaestus/state/) holds compact run, lineage, decision and governance records; append-only history is preferred.
- [`src/hephaestus/schemas/`](../../src/hephaestus/schemas/) defines interoperable typed JSON-compatible records.
- [`src/hephaestus/storage/`](../../src/hephaestus/storage/) and artifact indexes hold bounded path/hash references; model weights, training corpora, full raw logs and private holdouts stay in explicitly approved durable storage.
- [Retrieval memory](../retrieval_memory_policy.md) derives auditable context from records and is advisory; it is not an authority to rewrite recorded history.

The [lineage policy](../lineage_policy.md) distinguishes `best`, `last_stable` and `certified_stable` checkpoints. A branch never inherits certification automatically. A suspected poisoned lineage must not be rescued just because it consumed compute.

## Evidence levels and recovery

A path or URL is not evidence that the referenced bytes exist. Content hash, immutable revision, independent readback and custody records determine what can be reproduced. [Replay verification](../replay_verification_policy.md) distinguishes `missing`, `insufficient`, `partial` and `reproducible`; reference-only replay must not be called byte-identical.

Actual recovery depends on a bounded, registered action and external authorization, not just a model suggestion. See [recovery](../autonomous_recovery.md), [durable model storage](../durable_model_storage_and_repository_evidence.md) and [safe operations](../operations/run-safety.md).

## Historical source

The original R10 execution and archive hashes are documented in [the frozen evidence index](../evidence/original-cuda45-r10/README.md). The later [integration provenance](../integration_notes/151-r10-full-history-and-runtime.md) records what was ported to `main`; integrating source code is not rewriting the execution record.
