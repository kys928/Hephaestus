# Governed Operator Console

Hephaestus includes a small standard-library HTTP console at `hephaestus.app.console` for local operator inspection.

The console preserves read-only inspection routes for persisted run, lineage, replay, artifact, memory, and code-edit proposal records. It also exposes a narrow governed operator-action endpoint that appends policy-evaluated action records and can route code-edit approval/rejection decisions through the code-edit workflow. It does not expose file-edit, patch-apply, training-launch, or eval-pack mutation actions.

```bash
python -m hephaestus.app.console --state-root state --host 127.0.0.1 --port 8765
```

HTML routes:

- `GET /` — home and run summary.
- `GET /runs` — run table.
- `GET /runs/<run_id>` — run detail with lineage, manifest, evaluation, judge gates, replay verification, artifacts, memory, and warnings.
- `GET /lineages` — lineage table.
- `GET /code-edits` — code-edit proposal table.
- `GET /operator-actions` — append-only operator action table.
- `GET /healthz` — read-only health payload.

JSON routes:

- `GET /api/runs` — run list.
- `GET /api/runs/<run_id>` — run inspection payload.
- `GET /api/runs/<run_id>/replay` — replay verification report.
- `GET /api/code-edits` — code-edit proposal and execution-attempt lists.
- `GET /api/operator-actions` — append-only operator action list.
- `GET /api/run?run_id=<id>` — backward-compatible run inspection route.
- `GET /run?run_id=<id>` — backward-compatible HTML run detail route.

Mutation route:

- `POST /api/operator-actions` — append a policy-gated operator action. Supported action types are `approve_code_edit`, `reject_code_edit`, and `note`. Code-edit approval/rejection actions target `code_edit_proposal` records and are delegated to the governed code-edit workflow; the route never applies patches or launches training.
