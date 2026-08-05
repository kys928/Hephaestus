# Parallel component ownership

## Purpose

This file is the merge-conflict boundary for the autonomous experiment milestone. Feature branches must treat shared contracts and other teams' directories as read-only.

## Shared contract owner

Branch: `integration/autonomous-experiment-contracts`

Owned paths:

- `src/hephaestus/schemas/contract_common.py`
- `src/hephaestus/schemas/diagnosis_contract.py`
- `src/hephaestus/schemas/discovery_contract.py`
- `src/hephaestus/schemas/experiment_contract.py`
- `src/hephaestus/schemas/lifecycle_contract.py`
- `src/hephaestus/interfaces/`
- `docs/autonomous_experiment_loop.md`
- `docs/component_ownership.md`
- contract tests and its integration note

After this branch merges, those files are shared read-only contracts. Amendments require a separate small PR.

## Feature ownership

### Evidence-based diagnosis

Branch: `feature/evidence-based-diagnosis`

Owned paths:

- `src/hephaestus/diagnosis/`
- `src/hephaestus/roles/diagnostician.py`
- `src/hephaestus/policy/diagnosis_policy.py`
- diagnosis tests and documentation

Consumes `DiagnosisRequest`; produces `DiagnosisReport`.

### Autonomous data factory

Branch: `feature/autonomous-data-factory`

Owned paths:

- `src/hephaestus/data/`
- `src/hephaestus/providers/datasets/`
- data tests and documentation

Consumes `DatasetSearchRequest`; produces candidate and selection records, then existing manifest/preprocessing/trainable-data contracts.

### Semantic evaluation

Branch: `feature/semantic-evaluation`

Owned paths:

- `src/hephaestus/evaluation/`
- `src/hephaestus/evaluators/`
- `src/hephaestus/scoring/`
- new versioned eval packs
- evaluation tests and documentation

Consumes `ExperimentProposal` and training evidence; produces `ExperimentComparison` and existing eval reports/scorecards.

### Real training lifecycle and governed model discovery

Branch: `feature/real-training-lifecycle`

Owned paths:

- `src/hephaestus/backends/`
- `src/hephaestus/training/`
- `src/hephaestus/runtime/`
- `src/hephaestus/providers/models/`
- training/backend/model-provider tests and documentation

Consumes `ModelSearchRequest` and `ExperimentProposal`; produces model candidates, a governed `ModelSelectionDecision`, and updates `TrainingRunHandle` through the lifecycle protocol. Model providers discover metadata and compatibility; they do not decide experiment strategy or bypass approval policy.

### Closed-loop planner

Branch: `feature/closed-loop-planner`

Owned paths:

- `src/hephaestus/planning/`
- `src/hephaestus/roles/planner.py`
- `src/hephaestus/policy/experiment_policy.py`
- planning tests and documentation

Consumes diagnosis and selection contracts; produces intervention and experiment proposals.

### Execution infrastructure

Branch: `feature/execution-infrastructure`

Owned paths:

- `src/hephaestus/infrastructure/`
- `src/hephaestus/jobs/`
- `src/hephaestus/storage/`
- `deploy/`
- `docker/`
- infrastructure tests and documentation

This branch provides adapters only. It must not change domain decisions or migrate every store in one change.

### Final integration

Branch: `integration/real-autonomous-loop`

Exclusive responsibility:

- wire services into `src/hephaestus/control/` and factories;
- modify the orchestrator only as narrowly required;
- add end-to-end integration tests;
- resolve contract-consistent cross-subsystem wiring.

No feature branch may modify `src/hephaestus/control/orchestrator.py`.

## Shared read-only paths

Unless explicitly assigned, all workers treat these as read-only:

- shared contract files listed above;
- `PROJECT_MISSION.md` and `AGENTS.md`;
- existing lineage, approval, replay, and action-boundary policy;
- existing frozen evaluation packs;
- central orchestrator and cross-subsystem factories.

## Change protocol

When a worker finds a missing shared field or method:

1. document the exact requirement in its integration note;
2. avoid changing the shared contract in the feature branch;
3. open a minimal contract-amendment PR;
4. merge that amendment first;
5. rebase affected branches onto the amended contract commit.

## Pull-request limits

Each PR should stay under roughly 15–25 changed files, address one subsystem, avoid unrelated cleanup, avoid global formatting, and include an integration note at `docs/integration_notes/<branch-name>.md`.
