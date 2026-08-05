# Autonomous experiment loop contract

## Scope

This document defines the shared contracts for the first real, evidence-driven Hephaestus experiment loop. It does not implement diagnosis, dataset/model discovery, preprocessing, training, evaluation, orchestration, or infrastructure. Those remain separate subsystem responsibilities.

Contract version: `autonomous-experiment.v1`.

## Design rule

Agents may recommend and select datasets or models only through governed discovery and selection contracts. A free-form model or dataset identifier inside a prompt is not an approved training input.

Every selection must preserve:

- the diagnosed problem and evidence references;
- all material candidates considered;
- compatibility, provenance, license, risk, cost, and missing metadata;
- ranking and rejection reasons;
- required approvals;
- confidence and an explicit `inconclusive` outcome when evidence is weak.

## Control-spine placement

The mandatory control spine remains unchanged:

1. Judge entry
2. Planner
3. Data acquisition and audit
4. Data preprocessor
5. Training engineer
6. Runtime monitor
7. Evaluator
8. Judge exit

The new contracts extend what these phases exchange. They do not add a hidden ninth phase or collapse roles into a monolithic agent.

## Contract flow

1. Judge entry or the diagnosis adapter creates a `DiagnosisRequest` from persisted run evidence.
2. A diagnosis service returns a `DiagnosisReport` containing observations, ranked hypotheses, missing evidence, and confidence.
3. The planner emits one or more `InterventionProposal` records. Each proposal names one primary variable and preserves controlled variables.
4. When an intervention needs data or a model, the planner emits a `DatasetSearchRequest` and/or `ModelSearchRequest`.
5. Approved providers return normalized candidate records. Providers discover; they do not approve.
6. Selection services return a `DatasetSelectionDecision` or `ModelSelectionDecision`. `selected`, `inconclusive`, and `blocked` are distinct outcomes.
7. Existing data roles may turn an approved dataset selection into `DatasetManifest`, `PreprocessingReport`, and `TrainableDataContract` records.
8. The planner emits an `ExperimentProposal` linking diagnosis, intervention, selection decisions, baseline, budget, success criteria, failure criteria, and rollback plan.
9. A training lifecycle service returns a `TrainingRunHandle` and accepts explicit `TrainingControlRequest` records for interrupt/resume/cancel behavior.
10. Evaluation returns an `ExperimentComparison` referencing persisted evaluation evidence.
11. Judge exit applies existing gates and lineage policy. The comparison is evidence, not an automatic promotion command.

## Lifecycle states

The autonomous experiment lifecycle is represented separately from lineage status. Allowed states and edges live in `schemas/lifecycle_contract.py`.

Terminal states are `completed`, `blocked`, `failed`, and `cancelled`. Terminal records are immutable; a retry or revised strategy creates a new experiment identity or an explicit branch.

No component may jump directly from diagnosis to training without recorded intervention, selection/approval where required, and readiness evidence.

## Error vocabulary

Subsystems exchange `ContractIssue` records. The finite categories include invalid requests, unsupported capability, policy blocks, missing evidence, provider failure, candidate incompatibility, unknown license/provenance, contamination risk, artifact-integrity failure, runtime failure, inconclusive evaluation, budget exhaustion, approval requirements, and internal contract violations.

An issue records whether it is retryable and blocking. Human-readable messages are not policy decisions.

## Versioning

`autonomous-experiment.v1` is additive. Later branches consume these fields without changing their meaning. A worker that needs a breaking change must propose a separate contract amendment before modifying its feature branch.

Adding optional metadata is backward-compatible. Renaming fields, changing status semantics, changing lifecycle edges, or changing protocol method signatures is a contract change.

## Non-goals

This contract layer does not:

- choose a real dataset or model;
- download data or execute remote code;
- launch or resume training;
- score model quality;
- diagnose failures;
- alter promotion or approval policy;
- rewrite the orchestrator;
- provide queues, databases, cloud scheduling, authentication, or deployment.
