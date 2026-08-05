# Integration note: autonomous experiment contracts

## Branch

`integration/autonomous-experiment-contracts`

## Purpose

Provide additive, versioned schemas and Python protocols so diagnosis, data, model discovery, planning, training, and evaluation branches can develop independently without editing the orchestrator or inventing incompatible payloads.

## Services to instantiate later

The final integration branch will need implementations of:

- `DiagnosisService`
- `DatasetDiscoveryProvider` and `DatasetSelectionService`
- `ModelDiscoveryProvider` and `ModelSelectionService`
- `ExperimentPlanningService`
- `TrainingLifecycleService`
- `ExperimentEvaluationService`

This branch provides only protocols.

## Contracts consumed

- existing `JsonSchema` serialization behavior;
- existing mandatory control spine;
- existing dataset manifest, preprocessing, trainable-data, eval, decision, lineage, approval, and replay contracts by reference.

## Contracts produced

- `ContractIssue` and finite vocabularies;
- diagnosis request/report contracts;
- dataset/model discovery candidates and selection decisions;
- intervention, experiment, training-control, run-handle, and comparison contracts;
- explicit lifecycle states and allowed transitions;
- runtime-checkable service/provider protocols.

## Configuration required

None. Implementations may later add provider registries and configuration, but those are not part of this PR.

## Failure modes

- invalid or unknown vocabulary is normalized conservatively;
- unknown selection status becomes `inconclusive`;
- confidence values are clamped to `[0, 1]`;
- invalid lifecycle jumps are detectable through `LifecycleTransition.is_allowed()`;
- issues remain explicit rather than being converted into exceptions at the contract boundary.

## Test fixtures

The contract tests provide fake dataset discovery, diagnosis, and training services. Future subsystem tests should reuse the protocol shapes but keep external network access optional.

## Known missing wiring

- no service registry;
- no state stores for the new records;
- no orchestrator integration;
- no real provider, planner, trainer, evaluator, or diagnostician;
- no approval-policy additions for discovery or model selection;
- no migration/version negotiation beyond the fixed v1 field.

## Merge-conflict risk

Low. The change is additive and does not modify the orchestrator, existing roles, existing stores, existing frozen eval packs, or existing schema exports.

## Exact next action

Merge this branch first. Start every feature branch from the resulting `main` commit and treat the contract files as read-only. The first recommended feature branch is `feature/evidence-based-diagnosis`.
