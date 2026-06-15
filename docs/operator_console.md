# Read-only Operator Console

Hephaestus includes a small standard-library HTTP console at `hephaestus.app.console` for local operator inspection.

The console is intentionally read-only. It lists persisted run records, shows run inspection payloads, and rejects mutation-oriented HTTP methods with `405 Method Not Allowed`. It does not expose approval, rejection, execution, or training-launch actions.

```bash
python -m hephaestus.app.console --state-root state --host 127.0.0.1 --port 8765
```

Useful endpoints:

- `/` renders a minimal HTML run list.
- `/api/runs` returns run records as JSON.
- `/api/run?run_id=<id>` returns the inspection payload for a run.
- `/healthz` returns a read-only health payload.
