# Integration note: closed-loop experiment planner

## Branch and scope

Branch: `feature/closed-loop-planner`.

This branch implements planning only. It does not modify or wire the orchestrator and does not execute discovery, data preparation, training, evaluation, approval, rollback, branch, restart, or stop actions.

## Planner service class

`hephaestus.planning.ClosedLoopExperimentPlanner`

It structurally implements the frozen `ExperimentPlanningService` protocol:

- `propose_interventions(diagnosis) -> Sequence[InterventionProposal]`
- `propose_experiment(diagnosis, intervention, dataset_selection, model_selection) -> ExperimentProposal`

It also exposes:

- `create_discovery_requests(diagnosis, intervention) -> (DatasetSearchRequest | None, ModelSearchRequest | None)`

`PlannerRole` delegates these contract methods while preserving its existing dry-run `run(...)` adapter until final integration.

## Policy class

`hephaestus.policy.experiment_policy.ExperimentPolicy`

The policy validates the one-primary-variable rule, controlled variables, evaluable criteria, and rollback plan. It performs deterministic heuristic ranking and delegates high-impact lineage approval classification to the existing `ApprovalPolicy`. It does not grant approval.

## Constructor dependencies

`ClosedLoopExperimentPlanner(policy=ExperimentPolicy(), memory_query=None, memory_limit=20)`

- `policy`: ranking, validation, cost/risk/reversibility, and approval-requirement policy.
- `memory_query`: optional implementation of local `PlanningMemoryQuery`; the existing filesystem-backed `state.Query` conforms structurally.
- `memory_limit`: bounded retrieval size for each memory category.

## Memory and query requirements

The planner reads only:

- `dead_ends_for_lineage(lineage_id, limit)`
- `similar_failure_patterns(lineage_id, tags, limit)`
- `intervention_history(lineage_id, limit)`

Memory is advisory evidence with source linkage. The planner does not mutate memory or infer that repeated agent agreement is evidence. A dead-end intervention is rejected unless its record contains an evidence basis and the current diagnosis carries genuinely new evidence references.

## Planning stages

1. Deterministically order the leading and alternative diagnostic hypotheses.
2. Generate contract-valid candidate intervention kinds.
3. Assign one primary variable and explicit controls.
4. Retrieve prior attempts and known dead ends.
5. Estimate bounded benefit, information gain, five cost dimensions, risk, reversibility, stage suitability, baseline quality, evidence completeness, and approval burden.
6. Reject unsafe or unsupported candidates and preserve reasons.
7. Rank accepted candidates deterministically.
8. Emit governed dataset/model search requests only when the selected intervention needs them.
9. Validate selection decisions, baseline, criteria, evidence, budget, approvals, and rollback before emitting an experiment proposal.

Selection decisions must match the deterministic request ID produced for the diagnosis/intervention pair. A nominally selected decision with blocking issues is rejected.

## Contracts consumed and produced

Consumed:

- `DiagnosisReport`, including observations, leading/alternative hypotheses, missing evidence, and metadata references;
- `DatasetSelectionDecision` and `ModelSelectionDecision` when required;
- deterministic `MemoryRecord` dictionaries through the existing query surface;
- existing `ApprovalPolicy` decisions.

Produced:

- ranked `InterventionProposal` records;
- `DatasetSearchRequest` only for `replace_or_mix_dataset`;
- `ModelSearchRequest` only for `change_model`;
- `ExperimentProposal` after all applicable selection and safety checks pass.

No shared contracts were changed.

## Ranking behavior

Ranking is deterministic for identical inputs. The score is a transparent bounded heuristic, not a calibrated probability. Proposal metadata preserves score components, estimate confidence, prior-attempt count, dead-end matches, approval requirements, rank, and all considered/rejected alternatives.

Known dead ends without new evidence are rejected. Missing evidence blocks training-like candidates and promotes a diagnostic evidence-collection intervention. Tie-breaking uses stable intervention kind, primary variable, and content-derived identifier ordering.

## Configuration

No new config file is required. Conservative cost, benefit, risk, reversibility, stage-preference, and primary-variable tables live in the planner-owned policy/service modules. Existing `configs/policies/approval_policy.yaml` remains authoritative for rollback, branch, and restart approval classification.

Diagnosis adapters should populate these optional metadata references when available:

- `baseline_ref` or explicit `baseline_justification`;
- `baseline_quality`;
- `lineage_trust_level`;
- frozen `eval_pack_ref`;
- tokenizer, architecture, data-manifest, training-recipe, and decoding references;
- budget and discovery constraints;
- `new_evidence_refs` when reconsidering a recorded dead end.

## Failure modes

- Invalid or multiple primary variables: proposal rejected.
- Missing controls, success/failure criteria, or rollback: proposal rejected.
- Material missing evidence: training-like interventions rejected; diagnostic proposal returned.
- Inconclusive/blocked diagnosis or a blocking diagnosis issue: no discovery or training experiment.
- Known dead end without new evidence: candidate rejected and reason preserved.
- Poisoned/deprecated/archived lineage: continuation-like candidates rejected even when memory retrieval is unavailable.
- Missing baseline without explicit justification: no experiment; `ExperimentPlanningError`.
- Required dataset/model decision absent, inconclusive, blocked, or empty: no experiment.
- Selection decision belongs to a different search request or carries a blocking issue: no experiment.
- Unneeded selection decision supplied: no experiment, preventing accidental governed-input substitution.
- `stop` recommendation: no experiment.

## Unresolved downstream dependencies

At branch creation, PR #35 was merged and no feature PRs were open. Diagnosis, autonomous data, semantic evaluation, and real-training implementations therefore had no merged stable service entry points or integration notes. This implementation targets only the frozen shared protocols and contract metadata extension points.

Final integration must verify the concrete diagnosis adapter supplies baseline/evidence metadata, route discovery requests to governed providers/selectors, persist planner outputs, and pass proposals to training/evaluation only after approval and readiness gates. No fallback provider or replacement contract is included here.

## Exact integration wiring

The final integration branch should:

1. Instantiate `Query(state_root)` and `ExperimentPolicy(approval_policy=existing_approval_policy)`.
2. Instantiate `ClosedLoopExperimentPlanner(policy=experiment_policy, memory_query=query)`.
3. Register that instance as the `ExperimentPlanningService`.
4. Pass the persisted `DiagnosisReport` into `propose_interventions` and persist every ranked proposal plus rejected-alternative reasons.
5. Select an intervention through the existing judge/approval boundary; do not treat planner rank as authorization.
6. Call `create_discovery_requests` only for the selected intervention and route any request through the governed discovery and selection services.
7. Call `propose_experiment` with the resulting selection decisions, or `None` where discovery is not required.
8. Persist the complete `ExperimentProposal`.
9. Run existing approval and run-readiness gates before handing the proposal to `TrainingLifecycleService` or `ExperimentEvaluationService`.
10. Preserve the mandatory control-spine order and leave judge exit authoritative for promotion, rollback, branch, restart, or stop execution.
