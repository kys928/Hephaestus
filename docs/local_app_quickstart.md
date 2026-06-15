# Local Hephaestus quickstart

This guide creates a safe local dry-run state for operator workflow experiments. The demo command uses the dry-run backend and persists control records only under the `--state-root` path you provide. Dry-run artifact references are path references in state, not real model checkpoint files, and no real training is launched.

## 1. Create demo state

```bash
python -m hephaestus.cli.create_demo_state --state-root /tmp/hephaestus-demo-state --format json
```

The command prints the generated `run_id`, replay status, dry-run note, and follow-up commands.

## 2. Inspect the run

```bash
python -m hephaestus.cli.inspect_run --state-root /tmp/hephaestus-demo-state --run-id demo-run --format text
```

Use the `run_id` printed by `create_demo_state` if it differs from `demo-run`.

## 3. Verify replay evidence

```bash
python -m hephaestus.cli.verify_replay --state-root /tmp/hephaestus-demo-state --run-id demo-run --format text
```

Replay verification reads persisted evidence and reports whether the run is reproducible, partial, insufficient, or missing.

## 4. Optional: launch the operator console

If the operator-console app lane is present in your checkout, launch it against the same state root:

```bash
python -m hephaestus.app.console --state-root /tmp/hephaestus-demo-state --host 127.0.0.1 --port 8765
```

This document is intentionally independent from the operator-console documentation so the CLI demo/bootstrap flow remains usable even when the app lane is not merged.

## Doctor check

To check a checkout or state root without writing state:

```bash
python -m hephaestus.cli.doctor --state-root /tmp/hephaestus-demo-state --format json
```

If you omit `--state-root`, doctor performs import checks only.
