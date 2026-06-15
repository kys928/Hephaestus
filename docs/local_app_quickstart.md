# Local app quickstart

This quickstart creates deterministic demo state for local inspection without launching training or writing outside the provided state root.

```bash
python -m hephaestus.cli.doctor --format json
python -m hephaestus.cli.create_demo_state --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.cli.inspect_run --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.cli.verify_replay --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.app.console --state-root /tmp/hephaestus-demo-state
```

`doctor` is read-only and never creates a missing state root. `create_demo_state` writes only under the caller-provided `--state-root`.
