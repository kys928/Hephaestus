# Local app quickstart

This quickstart creates deterministic, orchestrator-backed demo state for local inspection without launching real training or writing outside the provided state root. The demo uses `DryRunBackend`, so artifact and checkpoint values are references only; no real model weights, checkpoints, or training outputs are created.

```bash
python -m hephaestus.cli.doctor --format json
python -m hephaestus.cli.create_demo_state --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.cli.inspect_run --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.cli.verify_replay --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.app.console --state-root /tmp/hephaestus-demo-state
```

If `--run-id` is omitted, `create_demo_state` chooses `demo-run`, then `demo-run-2`, `demo-run-3`, and so on to avoid overwriting existing default demo runs.

`doctor` is read-only and never creates a missing state root. With an existing state root, it reports import readiness, run count, latest run ID, and latest replay status.
