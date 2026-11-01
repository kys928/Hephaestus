# Supported execution and current boundaries

Use this page to choose an active route. The action registry describes policy
vocabulary; it does not promise an executable next experiment for every action.

| Route | What actually executes | Boundary |
| --- | --- | --- |
| Local demo/console | Dry-run state and inspection; opted-in fixed fixture subprocesses for tests | No real model training proof; console does not unify the SQLite production campaign with legacy JSON stores |
| Standard staged production driver | Persistent eight-phase/45-operation orchestration with actual injected services, single action owner, bounded zero-update failed-run recovery | Missing services and approvals block; default branch/restart/rollback/stage/policy transitions lack automatic next-experiment handoffs and are rejected |
| Governed CUDA composition | Real pinned five-role inference, approved acquisition, actual training/evaluation and protected action gates | Frozen CUDA identities and genuine external authority required; original R10 completion supplies no new spending or promotion authorization |
| Public CPU autonomy benchmark | Actual tiny-model optimization, inference/NLL evaluation, development-only finite decisions, verified adapter warm starts and three sequential cycles per program | Explicit CPU-only research profile; deterministic adaptive policy; no frozen specialist models or useful-model behavior claim |
| Prospective #152 custodian service | Trusted-key verification, one final exposure session, raw token custody, paired statistics and fixed-model cold reexecution | Real service must be operated by an independent entrusted custodian; its fixture rehearsal supplies no such authority |
| Legacy Ardor backend | Loader/dataset/evaluation/runtime contracts and fixed fake-runner tests | No authenticated real Ardor repair or general automatic legacy launch; configured internal endpoint alone proves no live service |
| Code-edit workflow | Proposal and governance records | No unrestricted autonomous code-edit executor |

## Authentic CPU research walkthrough

Use Linux/POSIX host-local SQLite storage, sufficient local disk and a fresh
absolute output directory **outside synchronized workspace trees**. The accepted
campaign used an isolated `/tmp` directory. This is an operational constraint,
not a proven explanation of the earlier corruption; [#179](https://github.com/kys928/Hephaestus/issues/179)
remains open. `/tmp` is transient, so preserve quiescent evidence and a checked
SQLite backup before ending the session. From the repository checkout:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,peft-training,independent-certification]' \
  transformers==5.6.2 peft==0.21.2
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m hephaestus.development.cpu_autonomy_benchmark run /tmp/new-comparison
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m hephaestus.development.cpu_autonomy_benchmark cold /tmp/new-comparison
```

The run creates actual random model bytes and public authored task data locally,
with no model download. Nine programs match three training seeds across fixed,
heuristic and adaptive policies. Each performs twelve optimizer updates in
three native cycles. A real predeclared bootstrap fault tests bounded recovery;
its failed workflow stays terminal. A new process reexecutes the fixed
experiments, comparing token outputs, target NLL, adapter tensors,
optimizer/scheduler/RNG state and counters. Cold replay adds verification
compute beyond the matched experiment budgets. Repeat `run` only to resume
unchanged work or read cached final results; an unknown consumed final
assessment cannot be reset.

For one program, use `program ROOT --policy adaptive --seed 1117` and
`cold-program ROOT --policy adaptive --seed 1117`. The fixed manifest contains
all three-policy/three-seed commitments even when only a single program is run.
The [frozen plan](../scientific-reports/cpu-autonomy-comparison-plan-v1.md)
defines dose, selection, outcomes and weak three-seed uncertainty. The
[campaign report](../scientific-reports/remediation-campaign-20261101.md)
reports actual results. None of these examples authorizes paid work.

Restore evidence into an isolated environment retaining its original absolute
paths and pinned source/framework profile for exact cold replay. Physical-path
relocation/migration, cross-host distributed fencing and a portable imported
campaign format are not implemented. Keep raw models, data, state and outputs
outside Git. The local artifact/state owner can delete or clone files; this
is not an independent hostile-administrator custody service.

Use `SQLiteStateRepository(..., read_only=True).assert_integrity()` for existing
state, and `snapshot_to(NEW_PATH)` for a consistent checked backup. Never copy
only a live WAL database's main file or silently reconstruct lost completions.
See [campaign storage health](sqlite-campaign-health.md).

## Executable next decisions

The CPU profile supports a next admitted learning rate, same-config recovery
before optimization, verified parent adapter warm start with optimizer reset,
candidate rejection retaining baseline, and stopping at the frozen cycle cap.
Every continued checkpoint must match a retained lineage-bound producer hash.
Changing the global training recipe/data policy is not implied by choosing a
parameter within this already frozen finite experimental grid.

Standalone governed rollback/branch/restart state APIs retain their own approval
requirements. The default staged loop blocks them until a real next-run
handoff exists. `request_recheck` in the generic loop requests another complete
cycle; a pure evaluation-only automatic program is not supplied by that name.
Stage/data/curriculum changes, arbitrary new actions and automatic code edits
are excluded from this bounded research profile.

## Inactive scaffolds

The public APIs have distinct callers:

| API or asset | Current caller and status |
| --- | --- |
| `roles/planner.py::PlannerRole.run` | `control/orchestrator.py`; legacy bounded dry-run plan |
| `PlannerRole.propose_experiment` and `planning/service.py::ClosedLoopExperimentPlanner` | Typed experiment-planning API; `control/autonomous_experiment.py` and production composition use the actual planner |
| `roles/reporter.py::ReporterRole.run` | Legacy coordinator's deterministic string summary; no model-backed scientific report |
| `llm/prompt_loader.py::load_prompt` | Standalone declaration with no source/test/script callers in the reviewed tree |
| Nine top-level `prompts/*.md` TODO templates | Inactive scaffolds; neither production nor the CPU experiment loads them |
| `llm/role_contract.py::RoleMasteryContract.prompt/parse` | Active certified runtime, role probes and active Evaluator; bound to frozen `configs/models/hephaestus_role_mastery_runtime_v1.json` |

The CPU profile composes separate `cpu_autonomy_*` role owners. It uses the
standard staged production driver and actual training lifecycle, with a finite
deterministic policy instead of specialist-model inference. These distinctions
prevent a demo API or unused template from being presented as active research.

These empty modules have no active source/test/script imports in the reviewed
tree: `evaluation/probe_runner.py`, `evaluation/eval_hooks.py`,
`safety/checkpoint_guard.py`, `safety/role_boundary_guard.py`,
`safety/eval_guard.py`, `safety/state_guard.py`, `safety/policy.py`.
They remain placeholders, not operative safety controls. Active checks reside
in typed evaluation services, launch enforcement, checkpoint identity,
deterministic action/approval policy and the staged state/replay boundaries.
Package `__init__.py` scaffold labels do not imply those packages are empty.

Scientific gaps, operational performance and the remaining shortest path to
real Ardor repair and broader from-scratch development are recorded in the
campaign report. #159 tracks current documentation and the incomplete surface;
documentation cleanup does not implement these excluded capabilities.
