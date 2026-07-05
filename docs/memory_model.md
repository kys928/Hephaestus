# Memory Model

Hephaestus memory is a durable, queryable summary layer built from completed runs, decisions, reports, manifests, and lineage state. It is not a vector store and it is not a hidden scratchpad. Records are explicit JSON objects stored in `memory_records.jsonl`.

## Memory record schema

A `MemoryRecord` contains:

- `memory_id`: deterministic identifier.
- `memory_type`: category such as `promotion_block`, `known_dead_end`, `runtime_issue`, `eval_issue`, `data_issue`, `successful_intervention`, `rollback_event`, `branch_event`, or `lineage_status_change`.
- `source_kind` and `source_id`: source record identity.
- optional `lineage_id`, `run_id`, `stage_name`, and `created_at`.
- `severity`: default `warning`.
- `summary`: short operator-readable statement.
- `tags`: machine-queryable labels.
- `evidence_refs`: path references to supporting artifacts.
- `related_ids`: linked decisions, runs, checkpoints, or interventions.
- `confidence`: numeric confidence.
- `metadata`: JSON-serializable auxiliary fields.

## Deterministic identity

`memory_builder` derives ids by hashing memory type, source kind, source id, lineage id, run id, stage name, summary, and sorted tags. `MemoryStore.append()` ignores duplicate `memory_id` values. This makes memory construction idempotent for the same source facts.

## Memory construction

After judge exit, the coordinator builds memory records from the run record, lineage state, decisions, reports, and manifests. Implemented memory extraction includes promotion blocks, successful promotions/interventions, data issues, eval issues, runtime issues, repeated failures, rollback/branch/status changes, and known dead-end signals when available in source records.

Memory records summarize decision-critical lessons. They must not inline full datasets, full reports, logs, or model outputs. Evidence belongs in artifact files and is referenced by path.

## Query usage

The entry judge receives selected relevant memories before making an entry decision. The coordinator currently supplies recent dead ends, prior promotion blocks, data issues, and eval issues for the lineage, capped to a small set. The query layer also exposes similar failure patterns, intervention history, stable lineages, and suspect/poisoned lineages.

## Invariants

- Memory is append-oriented and de-duplicated by deterministic id.
- Memory records are JSON-serializable summaries.
- Heavy evidence is referenced, not embedded.
- Memory can influence judge context but does not directly override policy gates.
- A memory record must identify its source so operators can audit where it came from.
