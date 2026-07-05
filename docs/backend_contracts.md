# Backend Contracts

TODO: document concrete decisions and invariants for this subsystem.

## Execution evidence contract

Backends should report heavy artifacts by path reference and may attach lightweight evidence summaries containing existence, byte size, hash type, and content hash. Evidence summaries must not inline model weights, checkpoints, datasets, or logs.
