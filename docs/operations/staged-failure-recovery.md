# Terminal training failure and a fresh bounded attempt

`StagedProductionCycleDriver` now routes a narrow class of actual terminal
training failures to the existing deterministic `BoundedRecoveryService`
before returning a blocked program. The default ceiling is two training
attempts per scientific cycle, frozen across restarts. A successful retry must
still complete every original phase and all 45 substeps.

Eligibility requires the original sealed failed handle to match the actual
persistent lifecycle, only `non_zero_exit` issues, no checkpoint/resume token,
and an independently retained hash of `hf-optimizer-progress.v1`. That receipt
must explicitly show zero optimizer steps and no update ever started. Missing,
altered or ambiguous evidence blocks. The worker fences its progress record
before each optimizer update, so a crash during an update cannot masquerade as
a safe pre-update failure. Historical incident step-zero fields alone are not
proof that optimization never happened.

The recovery service must independently return an eligible approval-free
`restart_bounded_job` recommendation under the configured stage policy and
retry budgets. The driver retains the failure and decision, waits the declared
backoff, and gives the retry a new `...-attempt-N` run identity. The original
failed workflow remains terminal and incomplete. The production loop returns
`waiting` while the retry/backoff is pending; normal resume polling executes
the next attempt. A retry does not count as a completed scientific cycle.

The typed progress fence flushes the receipt file and its parent directory
before an update can begin. This closes the window where an unflushed
`update_in_flight` rename could be lost after optimization started.

Before relaunch, a wrapper checks the finalized scientific proposal against
the failed attempt: model, tokenizer, actual processed-data hash, parent
adapter, seed, recipe, optimizer, dose and hyperparameters must agree. New
run-scoped contract/processing wrapper paths and an explicitly declared CPU
fixture fault are operational fields. Unknown constraint keys remain included
in the fingerprint. The wrapper retains provider reconciliation when the
original delegate supports it; unsupported ambiguous outcomes stay blocked.

OOM, cancellation, post-update failures, failed evaluation, checkpoint
corruption, missing evidence and approval failures have no automatic retry
through this bridge. They need their existing governed recovery/approval path
or operator investigation. No scientific parameter adjustment, code edit,
rollback, branch or promotion is authorized here.

The default staged action-intent service explicitly blocks automatic branch,
restart, rollback, stage changes and recipe/data/evaluation policy mutation.
The standalone governed action executor retains its independently approved
state-transition APIs; those APIs do not by themselves implement the next
scientific experiment. A future automatic transition must provide a typed,
verified handoff to its child/reset/rollback run before this exclusion changes.

The `bounded_cpu_research` stage is an opt-in local research profile without
certification or promotion eligibility. Its predeclared bootstrap fault runs
after real model/tokenizer preparation and before optimizer construction.
`fixture_bootstrap_failure=True` is rejected outside this CPU stage. It exists
to test genuine worker failure, diagnosis, restart and evidence preservation;
it must be reported as a fault injection, not an unexpected production event.

Fresh verified staged checkpoints can now supply lineage membership through
their lineage-bound producer commitments. The action executor still checks
the current manifest against that retained identity; another lineage's
commitment grants no continuation authority.

`test_staged_failed_run_recovery.py` checks the failure boundaries, immutable
attempt ceiling and parameter-drift denial, then runs an actual failed CPU
worker, reconstructs production composition and completes two real optimizer
updates under a new attempt. Other stage workers in that isolated integration
test are explicit fixtures; the separate research benchmark supplies genuine
domain services across complete scientific cycles.
