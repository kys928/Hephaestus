# REPO_STRUCTURE_PLAN.md

## Documentation organization (current)

- `docs/README.md`: canonical documentation index.
- `docs/architecture/`: implementation map, phase/role authority, data/state/evidence boundaries.
- `docs/operations/`: safe local setup, approval/incident procedures, integration and rollback.
- `docs/scientific-reports/`: readable claims, experiment outcomes, limitations and source links.
- `docs/evidence/`: original published scientific artifacts and immutable historical receipt indexes. **Do not rename or rewrite evidence just to reorganize docs.**
- `docs/integration_notes/` and `docs/reviews/`: historic development/source-review records retained at original paths.
- Old `docs/*.md` entrypoints are maintained as compatibility/reference pages for existing PR links.

These directories are curated reading paths, **not** a claim that every historical note has been moved. Moving evidence or old linked records for aesthetic reasons would break provenance.

## Top-level structure
- `configs/`: human-authored system and policy configuration.
- `docs/`: architectural and operational documentation.
- `prompts/`: role-specific prompt templates.
- `state/`: persisted machine-readable control memory.
- `artifacts/`: heavy outputs (logs, checkpoints, eval dumps).
- `scripts/`: explicit operator scripts.
- `examples/`: sample inputs/plans/manifests.
- `tests/`: unit and integration tests.
- `src/hephaestus/`: application packages.

## Package structure (`src/hephaestus/`)
- `control/`: spine coordination, transitions, rollback/restart/promotion control.
- `schemas/`: typed JSON-serializable boundary objects.
- `state/`: append-only/version-aware stores and queries.
- `safety/`: safety gate checks.
- `roles/`: bounded specialist role entry points.
- `llm/`: model client and structured output adapters.
- `scoring/`: deterministic scoring and aggregation.
- `evaluation/`: eval-pack execution helpers.
- `data/`: acquisition, preprocessing, manifest + contract builders.
- `runtime/`: launch/session/event/incident utilities.
- `backends/`: backend contracts and adapters.
- `policy/`: judge/runtime/stage/restart/promotion policies.
- `utils/`: small shared helpers.

## Ownership boundaries
- Control layer orchestrates order only; it does not absorb role internals.
- Roles recommend/transform; stores persist; policy decides constraints.
- Schemas define all cross-boundary payloads.

## State vs artifact separation
- `state/` contains compact, indexable records.
- `artifacts/` contains large, immutable evidence files.
- State records hold artifact paths and hashes, not embedded heavy content.
