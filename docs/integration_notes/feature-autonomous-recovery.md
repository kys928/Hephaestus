# Integration note: bounded autonomous recovery

## Branch and immutable base

- Branch: `feature/autonomous-recovery`
- Required starting commit: `552200b95b027ed91ac76167619ac9478100e811`
- The branch was verified at that commit before editing; the commit was also verified as an ancestor of `main`.

This feature adds recovery decision and execution boundaries only. It does not modify shared autonomous-experiment contracts, `src/hephaestus/interfaces/`, the orchestrator, autonomous coordinator, existing policy, data, evaluation, training, backend, job, storage, lineage, replay, or frozen-eval implementation.

## Service classes

- `hephaestus.recovery.BoundedRecoveryService`
  - constructor: injected `RecoveryAttemptStore`, optional `RecoveryPolicy`, optional existing `StagePolicy`;
  - method: `decide(RecoveryRequest) -> RecoveryDecision`;
  - has no execution side effect.
- `hephaestus.roles.incident_responder.IncidentResponderRole`
  - `assess(request)` delegates to the decision service;
  - `execute_approved(decision)` is a separate explicit call and requires an injected controller.
- `hephaestus.recovery.RecoveryController`
  - constructor: injected attempt store, action-kind-to-executor map, and semantic allowlist;
  - `execute(decision)` rechecks eligibility, the existing action registry, approval references, handler availability, and attempt idempotency.
- `InMemoryRecoveryAttemptStore` and `FakeRecoveryActionExecutor`
  - deterministic offline fixtures, not production persistence or execution.

## Consumed evidence

`RecoveryRequest.evidence` accepts normalized adapters or structured records for:

- `DiagnosisReport` and explicit diagnosis evidence;
- runtime incidents and events;
- `TrainingRunHandle`, prepared-job, checkpoint, and resume-token evidence;
- `ExperimentComparison` and evaluation reports;
- provider and dataset-acquisition failures;
- job records, leases, heartbeats, worker ownership, late completion, and stale results;
- storage integrity and state persistence failures;
- replay verification;
- lineage state and memory-derived prior attempts.

Every input adapter must provide explicit structured fields for facts such as corruption, incompatibility, contamination, hash verification, token validity, and worker ownership. Summaries are not semantically guessed.

## Produced decisions

`RecoveryDecision` contains:

- a `FailureClassification` with support, contradictions, confidence, retryability, domain, evidence fingerprint, and stable failure signature;
- a `RecoveryRecommendation` with semantic action, existing registry action, parameters, reversibility, approval requirement, and evidence refs;
- `RetryBudgetDecision` across operation/signature/run/experiment/lineage/global-window scopes;
- a deterministic `BackoffDecision`;
- strict `CheckpointRecoveryDecision` when resume is requested or recommended;
- stable attempt ID, status (`eligible`, `approval_required`, `blocked`, or `inconclusive`), and `ContractIssue` records.

Decision metadata explicitly records that no action was executed, no Judge verdict was changed, and no checkpoint was promoted.

## Failure taxonomy

The implementation supports every requested category and additionally distinguishes storage-integrity and state-persistence failure. `unknown_inconclusive` remains mandatory when the evidence is missing, weak, conflicting, or cannot establish exclusive worker ownership.

## Retry, backoff, and infinite-loop policy

Default budgets:

- per operation: 2;
- per failure signature: 3;
- per run: 5;
- per experiment: 8;
- per lineage: 12;
- per caller-supplied deterministic global window: 20;
- identical unchanged evidence without progress: 1;
- repeatedly failed same action: 2;
- cumulative lineage recovery cost: 100 policy units.

All values are constructor-configurable. Counts and exhaustion reasons are returned in the decision. An A/B/A/B failed-action pattern blocks oscillation. Backoff uses bounded exponential delays and stable hash-derived jitter; the subsystem never sleeps.

## Checkpoint recovery requirements

An adapter must expose:

- checkpoint existence, `content_hash`, and explicit hash verification;
- resume-token existence and `valid=true`;
- exact `model_revision`, `tokenizer_ref`, `architecture_family`, `training_recipe_ref`, `data_contract_ref`, `data_contract_hash`, and `backend_id` on both checkpoint and token evidence;
- replay status `reproducible`;
- current lineage status that permits continuation.

Missing or mismatched evidence yields blocking issues. Final integration should adapt the lifecycle service's `checkpoint_record.json`, `resume_token.json`, replay report, and lineage state into these fields, then have the approved resume handler create the existing `TrainingControlRequest(action="resume")`.

## Worker-loss behavior

- Expired leases or missing workers may recommend `request_replacement_worker`, externally mapped to registered `rerun_same_config` behavior.
- Replacement parameters require a new exclusive lease and stale-result rejection.
- Duplicate ownership, live unattached processes, late completion, or stale result evidence changes the recommendation to evidence collection and explicitly rejects stale completion.
- The recovery package does not mutate queues or implement lease persistence. A final handler must use the injected `JobQueue` boundary and preserve its owner checks and stable job identity.

## Approval and action boundaries

- Every executable recommendation maps to an existing action-registry entry.
- Stage policy, lineage trust, registry category, forbidden state, and configured approval policy are evaluated before eligibility.
- Rollback, branch, and quarantine require matching existing approval evidence for the same registered action, run, and lineage.
- The controller rechecks registry permission and approval reference before invoking a handler.
- Promotion is unsupported by recovery; Judge verdict and checkpoint promotion are never controller outputs.

## Idempotency and persistence expectations

The controller records `executing` before calling a handler and updates the same stable attempt to `succeeded` or `failed`. Repeated calls with that attempt ID return the recorded state without re-execution.

Production integration must inject a durable compare-and-set attempt store. The included memory store is process-local and intended only for fixtures. External retry/resume/worker/artifact handlers must themselves honor the same attempt ID as their idempotency key.

## Exact final-orchestrator wiring

Recovery must not become a hidden ninth control-spine authority.

1. Runtime monitor, evaluator, or Judge entry persists the original failure evidence.
2. An integration adapter builds `RecoveryRequest` from references plus current lineage, replay, job/lease, budget, and prior-attempt evidence.
3. `IncidentResponderRole.assess()` produces and persists a decision without executing it.
4. Evidence-collection or bounded low-level control recommendations feed the existing phase owner; experiment-changing recommendations feed Planner/Judge exit.
5. Existing approval request/decision stores authorize high-impact registered actions.
6. Only after authorization may a composition-root-injected `RecoveryController` call an idempotent handler.
7. Resume handlers use `TrainingLifecycleService.control(TrainingControlRequest(...))`; worker handlers use `JobQueue`; rollback/branch/quarantine handlers use the existing governed transition path after Judge exit.
8. Result refs and attempt status return to append-only state and become evidence for the next normal control-spine pass.

## Known limitations

- Recovery records are subsystem-local because shared contracts are frozen.
- Free-form log interpretation is intentionally absent; adapters must emit explicit signals.
- Confidence is bounded evidence support, not a causal probability.
- The fake attempt store has no cross-process durability, locking, authentication, or authorization.
- There is no built-in persistent global time window; callers supply a deterministic `budget_window_id`.
- A successful handler call is not treated as genuine recovery progress until later evidence explicitly records progress.
- Stage profiles do not name every low-level recovery control. Safety stop, evidence recheck, checkpoint resume, and quarantine are treated as recovery-safe stage exceptions while all experiment-changing actions remain stage-governed.

## Missing shared contract requirement

No shared change is required for this feature PR. Final cross-process integration would benefit from a future small contract amendment adding a first-class recovery-service protocol and persisted recovery decision/attempt envelopes. Until then, integration should use the subsystem-local JSON-safe records and adapters documented here.
