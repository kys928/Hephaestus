# Architecture

TODO: document concrete decisions and invariants for this subsystem.

## Draft capability integration notes

The integration branch represents the draft capability set as additive modules rather than a monolithic orchestrator rewrite. Data acquisition and preprocessing roles delegate normalization to `src/hephaestus/data/` helpers, safety decisions are represented as explicit guard records, LLM access remains behind deterministic boundary abstractions, backend execution evidence is summarized as path/hash metadata, and scoring utilities remain deterministic and JSON-serializable.

Operator-originated mutation requests are deliberately separated from code-edit execution attempts: console actions are append-only operator-action records, while code-edit execution attempts remain bounded dry-run records tied to approved proposals.
