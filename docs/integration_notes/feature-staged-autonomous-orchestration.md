# Integration note: staged autonomous orchestration

## Branch and immutable base

Branch: `feature/staged-autonomous-orchestration`.

The branch began at merged baseline `552200b95b027ed91ac76167619ac9478100e811`
(PR #42, whose original head was `2e155b075c5f60c4ff5537f9dae8d29f0e8e9cff`).
It does not contain production diagnosis, discovery, acquisition, preprocessing,
training, generation, evaluation, recovery, infrastructure, or judge
implementations.

## Public orchestration entry points

The opt-in orchestrator is:

```python
from hephaestus.control import (
    StagedAutonomousDependencies,
    StagedAutonomousServices,
    build_staged_autonomous_orchestrator,
)
```

Direct factory:

```python
orchestrator = build_staged_autonomous_orchestrator(
    run_id=run_id,
    lineage_id=lineage_id,
    stage_name=stage_name,
    dependencies=dependencies,
)
state = orchestrator.run()
```

The existing factory also exposes an explicit mode:

```python
orchestrator = build_orchestrator(
    state_root=state_root,
    run_id=run_id,
    lineage_id=lineage_id,
    stage_name=stage_name,
    mode="governed_autonomous",
    staged_dependencies=dependencies,
)
```

`mode="legacy"` remains the default. Existing callers receive the original
`Orchestrator` and `DefaultSpineCoordinator` behavior without providing new
dependencies.

## Mandatory top-level spine and substeps

`SPINE_ORDER` is unchanged. `PHASE_SUBSTEPS` maps the expanded workflow into the
eight existing authority boundaries:

| Top-level phase | Ordered substeps |
| --- | --- |
| Judge entry | Judge entry decision; evidence collection; evidence-based diagnosis; readiness to plan |
| Planner | Diagnosis handoff; intervention ranking; experiment proposal; discovery-request preparation; approval-requirement discovery |
| Data acquisition and audit | Dataset discovery; dataset selection; acquisition approval gate; acquisition; license/provenance evidence; audit; manifest production |
| Data preprocessor | Approved-source preprocessing; deduplication/contamination; tokenizer compatibility; trainable-data contract |
| Training engineer | Model discovery; model selection; model approval gate; training input binding; launch configuration; lifecycle launch |
| Runtime monitor | Training status poll; runtime observation; runtime evidence; bounded recovery advice; runtime-control governance |
| Evaluator | Checkpoint resolution; generation-prompt materialization; baseline generation; candidate generation; semantic comparison; deterministic regression evidence; repeatability/variance evidence; human-review references |
| Judge exit | Governed verdict; action boundary; action approval gate; promotion gate; action application; replay evidence |

Diagnosis, model discovery, generation, and recovery are substeps. They are not
new top-level agents.

## Injected dependencies

`StagedAutonomousDependencies` requires:

- `services`: `StagedAutonomousServices`, holding explicit adapters for Judge
  entry, evidence collection, diagnosis, plan readiness, planner, dataset
  discovery/selection/acquisition, preprocessing, model discovery/selection,
  training lifecycle, runtime monitor, generation, evaluator, recovery, Judge
  exit, and governed action execution;
- `state_repository`: the existing infrastructure `StateRepository` protocol;
- optional `approval_service`: matching approval decisions to persisted stable
  requests;
- optional `record_sink`: an existing integration/state sink for routing domain
  records to canonical stores;
- optional `artifact_store`: the existing `ArtifactStore` protocol for immutable
  checkpoint verification;
- optional `job_queue`: configured execution transport, retained as an injected
  boundary rather than created implicitly.

Every feature adapter implements the narrow orchestration-local
`StagedOperationService` protocol. Its request contains identities, the exact
phase/substep, stable operation ID, attempt number, input references, and prior
compact outputs. Its result contains finite status, output references,
decision-critical records, blocking issues, resumability, and compact metadata.

Missing services block explicitly. The orchestrator never instantiates a
remote provider, performs a model download, or launches real training by
default.

## Persistence and ordering

The state repository uses two collections:

- `staged_workflow_states`: append-only workflow snapshots;
- `staged_workflow_records`: ordered phase/substep records.

Each step tracks:

- top-level phase and substep;
- stable operation ID;
- attempt number;
- finite status;
- input and output references;
- blocking issues;
- approval request;
- resumability;
- completion marker;
- compact output metadata.

Every domain record has a content-derived record ID and a monotonically ordered
workflow sequence. Re-emitting the same operation/kind/payload does not append a
duplicate record. The optional record sink receives the original domain kind
and payload so a composition adapter can route Judge decisions, approvals,
manifests, reports, handles, comparisons, and actions into existing canonical
stores.

Heavy artifacts remain references. No model outputs, datasets, checkpoints, or
logs are inlined into workflow snapshots.

## Approval pause and resume

Dataset, model, and Judge-exit action gates have stable approval request IDs
derived from the operation, exact subject, and requirement set.

When approval is required, the orchestrator:

1. persists the request;
2. stops before acquisition, launch, or protected action execution;
3. returns `approval_pending` with `resumable=true`;
4. asks the injected approval service for a decision on resume;
5. verifies request ID, operation ID, subject, and requirement set;
6. rejects stale or mismatched decisions;
7. persists the matching decision before continuing.

Planner selection or metadata cannot approve its own action. A repeated resume
does not recreate the request or repeat a completed protected operation.

## Idempotency and asynchronous execution

Operation identity is stable across process interruption and resume. Completed
substeps are skipped. Interrupted and retryable substeps increment their attempt
number while retaining the same operation ID.

This prevents duplicate acquisition, training launch, generation, action
application, and immutable record creation. The training lifecycle launch is a
completed orchestration step once a handle exists, but that does not mean
training completed. `training_status_poll` pauses with `interrupted` until the
injected lifecycle adapter reports a terminal status. Resume therefore polls or
applies an explicit lifecycle control through the adapter without launching a
second run.

The injected job queue may transport work, but queue submission or launch return
is never interpreted as semantic completion.

## Failure behavior

Finite step outcomes are:

- `completed`;
- `blocked`;
- `inconclusive`;
- `retryable_failure`;
- `terminal_failure`;
- `cancelled`;
- `interrupted`;
- `approval_pending`.

Subsystem exceptions become persisted `subsystem_failure` evidence with the
exception type and a terminal-failure state. They are never converted into a
successful phase. Missing diagnosis evidence stops at Judge-entry readiness,
before the Planner phase. Missing services block rather than trigger a hidden
fallback implementation.

## Evaluation and Judge authority

Checkpoint resolution requires a concrete checkpoint reference. Immutable
`sha256:` checkpoint references are verified through the injected artifact
store when configured. Baseline and candidate generation must expose the same
non-empty generation-settings identity before semantic comparison can run.

Training status or loss cannot substitute for semantic comparison. Evaluator
recommendations remain advisory: only `governed_verdict` supplies the action
examined by the existing action registry. Unknown and forbidden actions block
before any executor call. Approval-required actions are re-evaluated with their
matching approval at action application. A `promote_checkpoint` action also
requires the injected Judge/promotion-gate adapter to report
`promotion_allowed=true`.

Recovery output is advisory evidence. It reaches Judge/runtime control
governance but is not executed blindly.

## Replay evidence

The final replay substep records:

- exact top-level phase order and stable operation IDs;
- workflow, run, lineage, and stage identities;
- dataset and model revisions;
- configuration and seed identities;
- concrete checkpoint reference;
- generation-settings and eval-pack identities;
- approval references;
- accumulated evidence references.

A workflow cannot receive its final completion marker when those required replay
identities are missing. Replay verification policy is not weakened.

## Legacy compatibility

The legacy coordinator, roles, dry-run backend path, lineage transitions,
promotion gates, approval records, replay verifier, and existing factory
arguments are unchanged. `build_orchestrator(...)` uses the legacy path unless
the caller explicitly selects `mode="governed_autonomous"` and supplies staged
dependencies.

## Unresolved continuation dependencies

All five continuation branches were still at the immutable baseline and had no
open PRs when this branch began:

- `feature/production-execution-infrastructure`: provide the production state,
  artifact, job, lock, secret, and event adapters;
- `feature/production-data-acquisition`: bind remote governed acquisition,
  license/provenance evidence, audit, manifest, and preprocessing adapters;
- `feature/real-hf-training`: bind model discovery/selection and asynchronous
  production training lifecycle/status/control adapters;
- `feature/evaluation-generation-bridge`: bind frozen prompt materialization,
  baseline/candidate generation, generation parity, and semantic evaluator
  handles;
- `feature/autonomous-recovery`: bind runtime incident evidence and bounded
  recovery advice/control requests without bypassing Judge authority.

The current branch uses protocols and deterministic fakes until those APIs are
reviewed. It does not copy their unfinished feature code.

## Exact future integration merge order

Use a new integration branch from the reviewed common baseline. Do not rebase
one continuation feature branch onto another.

1. Merge `feature/production-execution-infrastructure` to establish composition
   adapters without domain decisions.
2. Merge `feature/production-data-acquisition` and bind dataset-side adapters.
3. Merge `feature/real-hf-training` and bind model/training lifecycle adapters.
4. Merge `feature/evaluation-generation-bridge` after the concrete checkpoint
   and lifecycle handles are stable.
5. Merge `feature/autonomous-recovery` after runtime event/control contracts are
   stable.
6. Merge `feature/staged-autonomous-orchestration` last, then add only narrow
   composition adapters for the reviewed concrete services.
7. Run the full staged matrix, legacy suite, replay verification, approval,
   promotion, lineage, and generated-artifact checks before proposing the final
   integration PR to `main`.

No continuation service should modify `SPINE_ORDER`, the central action or
approval policy, promotion gates, lineage transitions, replay policy, or frozen
eval packs during that composition pass.
