# AGENTS.md — Hephaestus Development Rules

## Mission

Hephaestus is a disciplined research-engineering system for training, repairing, evaluating, and stabilizing language models from scratch.

This repository is **not** a playground for freeform agent behavior.
It is a controlled system for improving model-development decisions under uncertainty.

Every code change must strengthen at least one of these properties:

- correctness
- reproducibility
- diagnosability
- rollback safety
- evidence quality
- decision quality

If a change does not clearly improve one of those, do not make it.

---

## Core operating posture

Work like a strict research lab, not an improvising hacker.

Prefer:

- explicit over implicit
- small scoped changes over broad rewrites
- deterministic behavior over cleverness
- evidence trails over hidden logic
- rollback-capable flows over risky shortcuts
- schema validation over “best effort” parsing

Never optimize for looking advanced.
Optimize for making correct decisions and leaving behind a trustworthy record.

---

## Hard rules

### 1. Do not silently redesign the system

Do not rename, move, merge, or split major architectural components unless the task explicitly requires it.

Do not introduce new architectural abstractions just because they seem elegant.

Do not convert the repo into a framework, plugin system, or generic agent platform.

Hephaestus is a specific system with a specific mission.

---

### 2. Schema-first, always

Before changing logic that reads or writes structured objects, check the relevant schema in `spec/`.

Structured objects include:

- targets
- runs
- lineages
- eval packs
- data manifests
- training plans
- diagnosis reports
- verdicts

If a code change affects any of those and the schema must change, update the schema first or in the same patch.

Do not create undocumented object shapes.

Do not “just add a field” in code without aligning the schema.

---

### 3. Preserve the control spine

The core lifecycle is:

1. define target
2. retrieve context
3. diagnose bottleneck
4. propose intervention
5. validate safety
6. freeze data plan
7. freeze training plan
8. launch run
9. evaluate candidate
10. judge outcome
11. update lineage
12. persist full trail

Do not bypass this lifecycle unless explicitly asked.

Do not collapse multiple phases into one convenience shortcut if it reduces clarity or auditability.

---

### 4. One source of truth for persisted state

Persisted machine state belongs in `state/`.

Supporting code for state access belongs in `src/hephaestus/store/`.

Do not create duplicate hidden state in random files, temp folders, caches, or ad hoc JSON dumps.

If something is needed for decisions, it should be in `state/` or referenced from `evidence/`.

---

### 5. Evidence is mandatory

Decision-relevant outputs must leave an evidence trail.

Evidence belongs in `evidence/`.

Examples:

- launch commands
- stdout/stderr logs
- eval outputs
- score summaries
- comparison outputs
- deterministic check results
- verdict rationale
- incident traces

Do not implement logic that makes important decisions without producing inspectable evidence.

---

### 6. No silent safety weakening

Do not weaken or bypass:

- checkpoint compatibility checks
- tokenizer compatibility checks
- frozen eval usage
- rollback requirements
- deterministic regression gates
- stage constraints
- dataset approval rules

If a task requires changing safety behavior, it must be explicit in the patch and easy to review.

---

### 7. No fake autonomy

Do not implement behavior that pretends to be intelligent while hiding uncertainty.

Agents/roles must separate:

- observations
- inferences
- hypotheses
- decisions
- confidence

Do not output confident recommendations when evidence is weak.

“Inconclusive” is a valid outcome.

---

### 8. No uncontrolled code editing

Do not create self-editing, repo-mutating, or open-ended code generation loops.

If a coding role is added later, it must be constrained, file-scoped, reviewable, and off by default.

V1 should not behave like an unrestricted self-modifying system.

---

### 9. Do not add broad dependencies casually

Keep dependencies minimal.

Before adding a library, ask:

- does the standard library already solve this?
- is this dependency essential for V1?
- does it reduce complexity, or just move it elsewhere?
- will it make the system harder to debug or port?

Avoid heavy orchestration frameworks unless explicitly requested.

---

### 10. No vague TODO architecture

Do not fill the repo with empty abstractions, speculative base classes, or placeholder infrastructure.

Every file added should have a concrete responsibility.

If a component is out of scope for V1, omit it rather than adding fake scaffolding.

Lightweight skeletal files are acceptable only when they stabilize the intended repo shape and are clearly marked.

---

## Allowed V1 scope

V1 is allowed to include:

- schemas in `spec/`
- persisted state handling in `state/` + `src/hephaestus/store/`
- control plane in `src/hephaestus/control/`
- bounded specialist roles in `src/hephaestus/roles/`
- runtime adapters in `src/hephaestus/runtime/`
- scoring and comparison in `src/hephaestus/scoring/`
- data manifest generation in `src/hephaestus/data/`
- diagnosis ordering and bottleneck ranking in `src/hephaestus/diagnosis/`
- compatibility checks in `src/hephaestus/contracts/`
- operator entry points in `tools/`
- tests in `tests/`

V1 should support a full disciplined loop for one experiment cycle.

---

## Out of scope for V1 unless explicitly requested

Do not add these by default:

- unrestricted code-editing agents
- autonomous repo mutation
- complex multi-agent messaging buses
- distributed workers
- web dashboards
- Postgres or external DBs
- vector databases
- queue systems
- generic plugin frameworks
- advanced reward-model orchestration
- broad contamination analysis suites
- quarantine automation beyond basic lineage status updates
- self-healing runtime daemons

If a simpler implementation works for V1, use the simpler implementation.

---

## Folder responsibilities

### `config/`
Human-authored system configuration.

Contains:

- system policy
- backend bindings
- stage definitions
- eval definitions
- decoding contracts

Do not treat config as a dumping ground for generated state.

---

### `spec/`
Formal schemas for persisted objects.

Any structured object written to `state/` should conform to a schema here.

Keep these stable, explicit, and versionable.

---

### `src/hephaestus/control/`
Control plane.

Responsibilities:

- orchestrator/controller
- state machine
- transitions
- safety validation

This layer decides workflow order.
It should not contain backend-specific shell logic.

---

### `src/hephaestus/roles/`
Bounded specialist roles.

Responsibilities:

- planner
- memory
- data curator
- training engineer
- evaluator
- judge
- lineage manager

Roles recommend or transform structured inputs into structured outputs.
They do not directly mutate persisted global state except through approved store interfaces.

---

### `src/hephaestus/store/`
Persistence and retrieval logic.

Responsibilities:

- object models
- state load/save
- filesystem indexing
- query helpers

This is the canonical access layer for `state/`.

---

### `src/hephaestus/runtime/`
Execution boundary.

Responsibilities:

- shell execution
- launch wrappers
- training backend adapter
- eval backend adapter

This is where external commands are called.

Keep it deterministic and well logged.

---

### `src/hephaestus/scoring/`
Evaluation and comparison logic.

Responsibilities:

- eval runner
- deterministic checks
- candidate comparison
- eval report construction

This layer should support strict, repeatable evaluation.

---

### `src/hephaestus/data/`
Data manifest handling.

Responsibilities:

- dataset registry
- manifest generation

Do not allow mystery datasets or undocumented mixtures.

---

### `src/hephaestus/diagnosis/`
Diagnosis logic.

Responsibilities:

- mandatory debug order
- bottleneck ranking

Do not let diagnosis jump straight to speculative deep causes without checking simpler ones first.

---

### `src/hephaestus/contracts/`
Compatibility and contract validation.

Responsibilities:

- model contract checks
- tokenizer contract checks
- checkpoint contract checks
- backend capability checks

These checks protect safety-critical assumptions.

---

### `src/hephaestus/common/`
Small shared utilities only.

Do not hide important business logic here.

If logic is domain-specific, put it in the domain module instead.

---

### `state/`
Persisted machine-readable state.

Examples:

- targets
- runs
- lineages
- eval packs
- datasets
- stage materializations

Files here must be structured, valid, and intentional.

---

### `evidence/`
Decision-support evidence.

Examples:

- logs
- raw eval outputs
- comparisons
- score summaries
- incident traces
- copied plans/manifests if useful

This folder exists to justify decisions after the fact.

---

### `tools/`
Operator entry points.

Examples:

- create target
- run cycle
- diagnose run
- launch training
- run eval
- judge candidate
- update lineage

These should be clean, practical CLI tools.

---

### `tests/`
Unit, integration, and smoke coverage.

All major control-plane behavior should be testable without real heavy model training.

Use dummy backends where possible.

---

## File creation rules

When adding a new file:

1. Put it in the narrowest correct folder.
2. Give it one clear responsibility.
3. Avoid placeholder content.
4. Add docstrings and comments only where they genuinely improve clarity.
5. Do not create files “for future flexibility” unless they stabilize an already approved architecture.

---

## Editing rules

When editing existing files:

- preserve current public behavior unless the task requires changing it
- keep diffs as small as possible
- avoid unrelated cleanups in the same patch
- avoid mixing renames with logic changes when possible
- maintain backward compatibility where reasonable in V1

If a file is messy, do not rewrite the whole thing unless necessary for correctness.

---

## Logging and observability

Prefer structured, grep-friendly logging.

Log key phase boundaries clearly:

- target loaded
- context retrieved
- diagnosis completed
- intervention proposed
- safety validated
- data plan frozen
- training plan frozen
- run launched
- eval completed
- verdict issued
- lineage updated

Do not spam logs with noisy internal chatter.

---

## Error handling

Fail loudly on:

- invalid schema
- missing required state objects
- unsafe transitions
- missing required evidence
- contract mismatch
- invalid config
- forbidden promotion conditions

Do not swallow errors and continue.

Do not silently downgrade safety-critical failures into warnings.

---

## Testing expectations

At minimum, changes should preserve or add coverage for:

- schema validation
- store roundtrips
- state transitions
- safety checks
- deterministic scoring checks
- controller cycle behavior with dummy backends

If a change affects the control spine, add or update tests.

---

## Preferred implementation order for V1

Build in this order:

1. `spec/`
2. `src/hephaestus/store/`
3. `src/hephaestus/control/`
4. `src/hephaestus/runtime/`
5. `src/hephaestus/scoring/`
6. `src/hephaestus/roles/`
7. `src/hephaestus/diagnosis/`
8. `src/hephaestus/contracts/`
9. `tools/`
10. `tests/`

Do not start with “smart agents.”
Start with the spine.

---

## What good output looks like

A successful V1 patch should move the repo toward being able to do this:

- create a target
- load prior context
- produce a diagnosis report
- propose one bounded intervention
- validate safety
- freeze a data manifest
- freeze a training plan
- launch a run through a backend wrapper
- execute frozen evals
- produce a verdict
- update lineage
- persist the full reasoning trail

If a patch does not help the system do that more reliably, reconsider it.

---

## What bad output looks like

Avoid patches that mostly add:

- abstraction layers without behavior
- speculative future architecture
- excessive framework code
- fake autonomy
- duplicate state paths
- undocumented object formats
- fancy naming without operational value
- giant monolithic files
- hidden side effects

---

## Final instruction

When in doubt, choose the option that makes the system:

- easier to inspect
- easier to test
- easier to roll back
- harder to fool
- more honest about uncertainty

That is the standard.
