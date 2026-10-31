# Durable multi-cycle Evaluator-repair control-flow integration — CPU evidence

## Parent and scope

This change is stacked directly on draft PR #90
(`0d0b632a130b450ba00fd53e00a35b8bc1e3f25d`).

Hephaestus **already has** a single top-level durable
`ProductionLoopRunner` and an eight-phase, 45-substep
`StagedProductionCycleDriver`. We reused both. Adding a competing campaign
scheduler would split idempotency and action ownership.

This PR hardens the existing campaign restart contract and adds two explicit
control-flow integration acceptance scenarios. All expensive/model operations
in these tests are **identified deterministic CPU test doubles**; the real
orchestrator, production composition, artifact store, SQLite state, memory
retrieval and Judge-action executor are exercised.

## Production changes

`production/loop.py` now rejects:

- blank campaign identity or nonpositive cycle budget;
- reuse of an existing program ID with `resume=False`, preventing state reset;
- attempts to resume a program as a different lineage or stage;
- contradictory completed-cycle histories and duplicate cycle identities;
- changes to the frozen cycle cap after campaign creation.

It continues to persist and resume the existing 45-substep workflow under
the same operation IDs and single Judge-action owner.

## Test A: three whole staged cycles across process reconstruction

`tests/test_durable_multicycle_campaign.py` runs every stage through the
actual `StagedProductionCycleDriver`, with explicit deterministic scientific
workers for otherwise unavailable live services.

1. **First simulated intervention** produces a controlled failed comparison
   and a governed `reject_candidate` action. The experiment proposal,
   intervention and comparison are content-addressed and indexed by the
   real `ExperimentMemoryQuery`.
2. **Second simulated intervention** consults that *verified* historical
   failure and chooses a different variable. A training poll reports an
   externally pending operation; the loop enters `waiting` without applying
   an action or incrementing the scientific cycle counter.
3. A **new production runtime and new worker** resume cycle two from the same
   SQLite state without repeating the staged `lifecycle_launch`. A third
   simulated cycle performs evaluation and terminates under a governed
   `abort_run`.

It also verifies the exact three applied action receipts, unique operation
identities, three immutable comparison records, and terminal-run idempotency.

## Test B: actual five-role model inference integration into the loop

A separate test uses the existing `CertifiedRoleReasoner`,
`ArtifactModelEvidenceBinder`, actual five-role frozen JSON contract and
SHA-verified adapter materialization, with a **declared fake generation worker**
instead of actual 4B–8B model weights.

It verifies that model-originated diagnosis and Planner selection reach the
real production loop's `experiment_proposal` and that the Judge chooses an
action allowed under the configured scientific stage policy. The frozen
`smoke_test` stage allows `continue_lineage_best`; we do not weaken this
policy to manufacture a successful candidate rejection.

This proves production-path *model-output admission*, not real GPU reasoning.

## Unfinished and explicitly blocked for a real autonomous campaign

`StagedAutonomousServices` still needs **complete production adapters** for
all 45 steps. The current `ProductionCompositionRoot` exposes real domain
services but does **not** supply a comprehensive staged dependency factory.

The remaining work is to bind, without placeholders:

- approved dataset/model discovery and actual acquisition;
- genuine preprocessing/training input creation and validation;
- real PEFT launch/status with durable externally reconciled GPU resources;
- actual baseline/candidate generation and restored sample artifacts;
- independent experiment comparison and sealed holdout certification;
- Judge decisions derived from those real outputs and protected activation
  with explicit operator authorization.

The CPU test's outcome sequence is scripted as a **simulation**, and no
scientific model improvement, live training, genuine certification or
production role activation is claimed.

The PR is a complete incremental control-flow/restart correctness slice,
not a claim that the full self-repair campaign is implemented or ready for GPU.

No cloud resources were created, no S3 objects mutated, no PR merged and no
production routing activated.
