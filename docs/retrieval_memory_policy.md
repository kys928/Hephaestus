# Retrieval memory policy

Hephaestus retrieval memory is a deterministic, auditable index derived from persisted state.

## Core guarantees

- Memory records are derived from persisted evidence (`run_records`, `decision_records`, manifests, reports, lineage state), not freeform model belief.
- Each memory record includes explicit source linkage (`source_kind`, `source_id`) so operators can audit provenance.
- Memory generation is deterministic: identical source evidence yields identical `memory_id` values.
- Retrieval memory is advisory context for Judge/Planner inputs; it does not autonomously decide actions.

## Scope

The retrieval layer answers questions such as:

- Which similar failures happened in this lineage before?
- Which gate blocks repeated?
- Which lineages are suspect/poisoned/stable?
- Which data or eval integrity issues are recurring?
- Which rollback/branch interventions are historically relevant?

## Non-goals

- No vector search or embeddings.
- No autonomous planner rewriting.
- No mutation of prior persisted records.
- No broad self-healing or agentic code editing behavior.

## Auditability

Every memory record carries:

- canonical memory type/severity,
- source linkage,
- evidence references,
- explicit confidence about evidence support.

Confidence describes evidence support strength, not model quality.
