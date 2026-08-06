# Bounded autonomous recovery

The recovery subsystem converts explicit failure evidence into deterministic classifications, retry-budget decisions, backoff decisions, and bounded recommendations. It does not replace diagnosis, approve its own actions, mutate Judge verdicts, promote checkpoints, or execute work during assessment.

## Authority separation

1. `BoundedRecoveryService.decide()` normalizes evidence, classifies the failure, evaluates budgets, and recommends an action.
2. Existing action-boundary, stage, lineage, replay, checkpoint, and approval policy determine eligibility.
3. Matching approval evidence authorizes registered high-impact actions.
4. `RecoveryController.execute()` may call only an injected, semantic-action allowlisted handler after rechecking the existing action registry.

`IncidentResponderRole.assess()` calls only the decision layer. Execution requires a separate `execute_approved()` call and an explicitly injected controller.

## Failure evidence

`RecoveryRequest` accepts structured records from diagnosis, runtime incidents/events, training handles, checkpoint and resume records, evaluation comparisons, provider/data acquisition failures, job/lease records, storage/state failures, replay reports, lineage state, and prior recovery attempts. Normalization is JSON-safe, deterministic, and preserves every source reference.

Corruption, incompatibility, contamination, instability, and exclusive worker ownership require explicit signals. Free-form summaries are retained for inspection but are not parsed into hard recovery facts.

## Classification and uncertainty

The finite taxonomy covers provider/network interruptions, worker loss, crashes and interruptions, memory/resource exhaustion, data-loader and data-integrity failures, tokenizer/model/checkpoint incompatibility, checkpoint and resume integrity, missing/incomplete evaluation, deterministic regression, variance, replay, policy/approval, configuration, unsupported capability, lineage status, storage/state persistence, and an explicit `unknown_inconclusive` outcome.

Each classification records support, contradictions, alternatives, confidence, retryability, failure domain, approval need, and whether automatic handling is safe. Conflicting or weak evidence produces `unknown_inconclusive`; agent agreement is never evidence.

## Recovery actions and registry mapping

Semantic recovery detail stays in the owned recovery record while external mutation remains bound to the existing registry.

| Recovery recommendation | Existing registered action |
| --- | --- |
| retry operation / retry after backoff / restart bounded job | `rerun_same_config` |
| replace lost worker | `rerun_same_config` with exclusive-lease requirements |
| reacquire partial artifact | `rerun_same_config` with artifact parameters |
| resume verified checkpoint | `continue_from_checkpoint` |
| rerun evaluation / collect evidence | `request_recheck` |
| rollback verified checkpoint | `rollback_to_checkpoint` |
| branch experiment | `branch_new_experiment` |
| quarantine lineage | `quarantine_lineage` |
| stop | `abort_run` |
| escalate | advisory only; no executable action |

This mapping does not add aliases to the shared action registry or bypass it. Unknown and forbidden registered actions remain blocked.

## Retry and backoff policy

Budgets are explicit per operation, failure signature, run, experiment, lineage, and caller-supplied deterministic global window. Identical evidence without observed progress, repeated action failures, A/B action oscillation, and cumulative cost can independently stop retries. A genuine progress record resets operation-local retry and backoff history; broader run/lineage costs remain cumulative.

The decision engine returns a backoff duration and never sleeps. Delay is bounded exponential backoff with optional deterministic hash-derived jitter. Provider, worker, storage, and training failures have separate bases.

## Checkpoint and worker safety

Resume requires explicit checkpoint existence, content hash, hash verification, valid resume token, exact model/tokenizer/architecture/recipe/data/backend compatibility, reproducible replay status, and a non-poisoned lineage. Missing or partial evidence refuses resume; there is no loose-load fallback.

Expired leases may recommend bounded replacement. Duplicate ownership, live unattached processes, late completion, and stale result evidence block replacement and mark stale completion rejected until exclusive ownership is established. Queue and lease persistence remain infrastructure responsibilities.

## Persistence and idempotency

Recovery attempts have stable identifiers derived from the request, operation, failure signature, evidence fingerprint, and recommendation. The controller records an attempt before invoking a handler and will not invoke the handler again for the same attempt ID.

`InMemoryRecoveryAttemptStore` and `FakeRecoveryActionExecutor` are deterministic test implementations only. Multi-process deployment requires a durable, compare-and-set attempt store and idempotent external handlers.
