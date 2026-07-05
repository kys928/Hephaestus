# State Layout

Hephaestus persists decision-critical state through `src/hephaestus/state/`. The current implementation uses deterministic JSON and JSONL files under a caller-provided state root.

## Store files

| Store | File | Shape | Purpose |
| --- | --- | --- | --- |
| `RunStore` | `run_records.jsonl` | append-only JSONL | Per-run lifecycle records. |
| `DecisionStore` | `decision_records.jsonl` | append-only JSONL | Judge and policy decisions. |
| `DecisionStore` | `approval_requests.jsonl` | append-only JSONL | Operator approval requests. |
| `DecisionStore` | `approval_decisions.jsonl` | append-only JSONL | Operator approval decisions. |
| `ManifestStore` | `manifests.jsonl` | append-only JSONL | Normalized dataset manifests. |
| `ReportStore` | `reports.jsonl` | append-only JSONL | Plans, profiles, reports, incidents, eval reports. |
| `ArtifactIndex` | `artifact_index.jsonl` | append-only JSONL | Lightweight index of artifact path refs. |
| `MemoryStore` | `memory_records.jsonl` | append-only JSONL with de-duplication by `memory_id` | Lineage memory records. |
| `CodeEditProposalStore` | `code_edit_proposals.jsonl` | append-only JSONL with latest-by-id listing | Code-edit proposals and resolutions. |
| `EvalPackStore` | `eval_packs.jsonl` | append-only JSONL | Frozen persisted eval-pack definitions. |
| `LineageStore` | `lineage_states.json` | single JSON object keyed by lineage id | Current state for all lineages. |
| `LineageStore` | `lineage_state.json` | single JSON object | Legacy/current-lineage compatibility document. |

## JSON serialization rules

`JsonStore.append()` writes one sorted-key JSON object per line. `JsonStore.all()` ignores blank lines and returns records in file order. `JsonStore.get_latest(key, value)` scans from the end and returns the newest matching record.

`JsonSingleDocument.write()` writes sorted, indented JSON. Lineage state uses this format to keep the latest complete lineage map in one document.

## Lineage state

`LineageStore` normalizes records with `LineageState`. `set_current()` writes both the all-lineages document and the legacy single-lineage document. `add_child()` updates parent child ids and keeps the legacy document in sync if it points at the updated parent.

Lineage records are the durable source for best checkpoint, stable checkpoint, certified checkpoint, trust level, status, parent/child relationships, and intervention history.

## Artifact references

The state layer stores heavy evidence by reference only. Examples include dataset manifests, processed datasets, metrics files, probe files, deterministic-check files, checkpoints, and intermediate eval artifacts. The artifact index stores records like `{run_id, kind, ref}`.

## Query layer

`Query` composes stores to answer governance questions without re-running phases. Implemented queries include latest run, recent failures, runs in stage, recent decisions, checkpoint relationships, repeatability summaries, pending approvals, approval history, memory categories, intervention history, stable lineages, and suspect or poisoned lineages.

## Persistence invariants

- Persisted records must be JSON-serializable.
- Decision-critical records must flow through state stores rather than ad hoc files.
- Heavy evidence must be stored by path reference.
- Append-only stores preserve audit history; callers that need current state should use latest-by-id helpers.
- Eval packs persisted through `EvalPackStore` are forced frozen if not already frozen.
