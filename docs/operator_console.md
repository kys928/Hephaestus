# Hephaestus Operator Console

The Operator Console is a zero-dependency, local, read-only web surface for inspecting persisted Hephaestus state. It uses only the Python standard library and does not require FastAPI, Flask, Django, React, Streamlit, or any external static assets.

## Quickstart

```bash
python -m hephaestus.app.console --state-root state --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765/> in a browser.

## Routes

### HTML

- `/` — dashboard with the configured `state_root`, run count, latest runs, and navigation links.
- `/runs` — table of run records with `run_id`, `lineage_id`, `stage_name`, `status`, `judge_action`, and `checkpoint_ref`.
- `/runs/<run_id>` — detail page for one run, including run data, lineage, manifest, evaluation, judge decision/gates, replay verification, artifacts, memory, and warnings.
- `/lineages` — table of lineage states including latest run, trust level, last effective action, and best checkpoint reference.
- `/code-edits` — table of code-edit proposals. This page intentionally has no approval, rejection, or execution controls.

### JSON API

- `/api/runs` — JSON list of run records.
- `/api/runs/<run_id>` — JSON detail payload for one run, or a `404` JSON error for an unknown run.
- `/api/runs/<run_id>/replay` — JSON replay verification report.
- `/api/code-edits` — JSON list of code-edit proposals.

## Read-only guarantees

The console is inspection-only:

- It does not append, update, approve, reject, execute, or launch any state records.
- It does not launch training and does not call the orchestrator.
- It does not mutate artifacts, checkpoints, data, frozen eval packs, secrets, or model weights.
- It exposes no code-edit approval, rejection, or execution actions.
- It renders scalar values with HTML escaping and renders nested data as escaped, deterministic pretty JSON.

Existing Hephaestus store read helpers may create a missing `state_root` directory because the shared JSON store path helper creates parent directories. For strict no-file-change checks, point the console at an already-created state directory and compare snapshots of existing files before and after requests.
