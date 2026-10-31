# State layout — stable reference path

[State architecture](architecture/data-and-state.md) describes the separation between compact typed run/decision/lineage state and heavy model/data artifacts referenced by immutable path/hash identity.

- `src/hephaestus/state/`: persisted control records and append-only decisions.
- `src/hephaestus/schemas/`: versioned cross-component payloads.
- `src/hephaestus/storage/`: evidence and artifact custody interfaces.

Use [lineage policy](lineage_policy.md), [retrieval memory](retrieval_memory_policy.md) and [replay verification](replay_verification_policy.md) for specific persisted trust guarantees. Do not commit generated SQLite, checkpoints or private training data.
