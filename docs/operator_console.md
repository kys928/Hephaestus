# Read-only Operator Console

Hephaestus includes a small standard-library HTTP console at `hephaestus.app.console` for local operator inspection.

The console is intentionally read-only. It shows persisted run, lineage, replay, artifact, memory, and code-edit proposal records, and rejects mutation-oriented HTTP methods with `405 Method Not Allowed`. It does not expose approval, rejection, execution, or training-launch actions.

```bash
python -m hephaestus.app.console --state-root state --host 127.0.0.1 --port 8765
```

HTML routes:

- `GET /` — home and run summary.
- `GET /runs` — run table.
- `GET /runs/<run_id>` — run detail with lineage, manifest, evaluation, judge gates, replay verification, artifacts, memory, and warnings.
- `GET /lineages` — lineage table.
- `GET /code-edits` — code-edit proposal table.
- `GET /healthz` — read-only health payload.

JSON routes:

- `GET /api/runs` — run list.
- `GET /api/runs/<run_id>` — run inspection payload.
- `GET /api/runs/<run_id>/replay` — replay verification report.
- `GET /api/code-edits` — code-edit proposal list.
- `GET /api/run?run_id=<id>` — backward-compatible run inspection route.
- `GET /run?run_id=<id>` — backward-compatible HTML run detail route.
