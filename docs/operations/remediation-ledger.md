# Implementation campaign ledger

Audit baseline: `bc79bf021c4f357a709594d4083befa18f39a82e` (2026-10-31).
All eleven original probes reproduced before repairs. The retained
`campaign-baseline-probes.json` records the original unsafe outcomes. Every
finding below is resolved: its failing regression was corrected, integrated
by a normal two-parent merge, and verified in the combined mainline tree.
Dated audit reports and original R10 evidence have not been rewritten.

## Findings and permanent regression evidence

| Finding | Priority; original source | Original reproduction; repaired behavior | Regression |
| --- | --- | --- | --- |
| F01 | P1; `production/actions.py`, `training/hf_lifecycle.py` | Rehashed replacement weights accepted at an old path; lineage-bound producer commitments now reject substituted bytes and missing legacy trust | [Checkpoint custody tests](../../tests/test_checkpoint_action_recovery_integrity.py) |
| F02 | P1; `production/actions.py` | Prepared continuation applied after checkpoint corruption; normal and unapplied recovery now share preconditions; committed exact after-state only finishes its journal | Same checkpoint custody module |
| F03 | P1; `control/staged_autonomous.py`, `production/loop.py` | Two controllers invoked one operation twice; host-local exclusive ownership plus durable invocation intent/returned receipts fence replay | [Ownership and process-death tests](../../tests/test_staged_execution_ownership.py) |
| F04 | P1; `evaluation/cpu_prospective.py`, `evaluation/tier_a_cuda_target.py` | Both research paths accepted duplicate keys; all future paths use strict typed recursive JSON scoring with an explicit scorer version | [Research scorer tests](../../tests/test_research_answer_scoring.py) |
| F05 | P2; `evaluation/paired_holdout.py` | `1e400` decoded to infinity and passed validity; recursive decoded finite-value checks reject overflow, NaN and infinity | Same scorer module and paired-holdout tests |
| F06 | P2; `training/hf_worker.py` | Step-three off-cadence final checkpoint reset cumulative examples/tokens to zero; all checkpoint saves and resumes retain real counters | [Actual CPU accounting tests](../../tests/test_hf_training_dose_accounting.py) |
| F07 | P2; `training/hf_worker.py` | Three batches/accumulation two consumed [0,1,2,0] in one declared epoch; finite epoch windows consume [0,1],[2], normalize the actual partial window, retain the cursor and reject unsupported shuffle | Same actual CPU accounting module and [schedule tests](../../tests/test_training_batch_schedule.py) |
| F08 | P1; `data/chunking.py`, `data/preprocessing.py`, `training/hf_worker.py` | Long prompt split lost its boundary and supervised prompt tokens; structured pairs stay indivisible, masking derives from the template, unsafe bounds reject | [Supervision integrity tests](../../tests/test_structured_supervision_integrity.py) |
| F09 | P2; `production/composition.py` | Absolute supplied configs failed from unrelated CWD; composed planner, approval, recovery, stage and evaluator share the resolved root | [Configuration-scope tests](../../tests/test_production_config_scope.py) |
| F10 | P1; `production/recovery.py`, `production/loop.py` | Sixth healthy pending poll asserted after five observations; only actual failures consume a persistent finite budget and exhaustion is typed | [Polling and retry tests](../../tests/test_production_polling_budget.py) |
| F11 | P1; legacy `control/orchestrator.py`, `roles/runtime_monitor.py` | Denied guard still called launch; coordinator and monitor now enforce dry-run or explicitly opted-in fixed fixture workers, including the actual prepared spec | [Legacy launch-boundary tests](../../tests/test_legacy_launch_enforcement.py) |

## Accepted branches, ordinary merges and full CPU CI

The original root causes, separately from their observed symptoms, were:

- F01: continuation trusted a checkpoint's current self-consistent manifest
  without an independently retained producer identity.
- F02: unapplied prepared-action recovery omitted the normal checkpoint
  precondition check.
- F03: persisted operation identity did not supply exclusive side-effect ownership.
- F04: research evaluators used ordinary JSON decoding rather than the strict
  duplicate-key and typed-answer contract.
- F05: syntactically valid numeric overflow was not checked for decoded finiteness.
- F06: the off-cadence final save omitted the true cumulative accounting values.
- F07: accumulation filled its window by wrapping the batch cursor beyond the
  declared epoch instead of ending and normalizing a partial window.
- F08: text chunking discarded structured prompt/target identity and masking boundaries.
- F09: composed services fell back to their own relative default configuration roots.
- F10: successful pending observations were counted as failed infrastructure attempts.
- F11: the legacy guard's denied result was recorded without enforcing a launch stop.

Every listed PR passed all maintained workflows emitted for its exact accepted
head, including pinned Torch CPU regression and ancestry/activation checks.
The links identify complete CI runs; individual focused test counts are not a
substitute for these checks.

| Findings | Branch; PR | Exact merge commit | Full pinned CPU run |
| --- | --- | --- | --- |
| F01/F02 | `fix/audit-checkpoint-custody-recovery`; [#164](https://github.com/kys928/Hephaestus/pull/164) | `1efaf7810125644a9e6df7b11f3c035d1770014b` | [41228985029](https://github.com/kys928/Hephaestus/actions/runs/41228985029) |
| F10 | `fix/audit-healthy-polling-budget`; [#165](https://github.com/kys928/Hephaestus/pull/165) | `52a920b6cc9e399406c67777b12ec09f5a652b2f` | [41229127837](https://github.com/kys928/Hephaestus/actions/runs/41229127837) |
| F03 | `fix/audit-staged-execution-ownership`; [#166](https://github.com/kys928/Hephaestus/pull/166) | `62a709b969eab34d749a936210cb4b27a3757e4d` | [41229809648](https://github.com/kys928/Hephaestus/actions/runs/41229809648) |
| F11 | `fix/audit-legacy-launch-enforcement`; [#167](https://github.com/kys928/Hephaestus/pull/167) | `de243a95d13c9b66cf2d0f7ec53161392cb4ac9d` | [41229884862](https://github.com/kys928/Hephaestus/actions/runs/41229884862) |
| F04/F05 | `fix/audit-strict-answer-scoring`; [#168](https://github.com/kys928/Hephaestus/pull/168) | `72f780d94978624fd81727a4911b6fd5f41ff206` | [41230239711](https://github.com/kys928/Hephaestus/actions/runs/41230239711) |
| F08 | `fix/audit-structured-supervision`; [#169](https://github.com/kys928/Hephaestus/pull/169) | `2f09ebd12a7a3b86977ccefe64dc3e72882403b1` | [41230931251](https://github.com/kys928/Hephaestus/actions/runs/41230931251) |
| F09 | `fix/audit-production-configuration`; [#170](https://github.com/kys928/Hephaestus/pull/170) | `359a68b23c30e70e167fc01b52bdcca321625ea0` | [41230542244](https://github.com/kys928/Hephaestus/actions/runs/41230542244) |
| F06/F07 | `fix/audit-training-dose-accounting`; [#171](https://github.com/kys928/Hephaestus/pull/171) | `4b6798b7ef08f68c5753c9e9f1c53eda5e1744c4` | [41231434218](https://github.com/kys928/Hephaestus/actions/runs/41231434218) |

The combined M2 checkout matched main tree
`000381739af367e9f4ab087c37b05c1aed96becf` exactly. Local applicable validation
returned **1,133 passed, 21 skipped, two optional downloaded-model tests
deselected** because the environment lacked SOCKS support. Those two tests ran
in the successful remote full suite. After custody and staged recovery, local
validation returned **1,159 passed, 21 skipped, the same two deselected**;
main's merged #174 tree matched `3141a619af94d632ff5b902064dc2f1fc05f9506`.

The accepted #177 head `44f3cdad201de192b4c69c290e7b222ebfcbc76a`
passed **1,177 tests, 21 skipped, two pre-existing warnings** in the
[full pinned CPU run 41272930183](https://github.com/kys928/Hephaestus/actions/runs/41272930183).
Both maintained production/autonomous integration workflows also passed.
The ordinary #177 merge is `6a8aac2b75f9c9da674cccae5ce94ea2643dbfb1`,
tree `ed4442cd697bc6810f5e412f5ad1ca2f6216dfb1`, containing all eleven fixes.

## Additional accepted implementation work

| Scope; branch; PR | Ordinary merge commit | Maintained validation |
| --- | --- | --- |
| Prospective distribution/power/source preparation; `science/152-prospective-design-preparation`; [#172](https://github.com/kys928/Hephaestus/pull/172) | `b43f79fb1cb094cae49785d3c2ec98f1399fbbb5` | [Full CPU 41232954367](https://github.com/kys928/Hephaestus/actions/runs/41232954367) |
| One-use custody and actual CPU rehearsal; `science/152-custody-cpu-rehearsal`; [#173](https://github.com/kys928/Hephaestus/pull/173) | `f2a43693fe31e834911841801250ab00f3e08bb7` | [Full CPU 41268735148](https://github.com/kys928/Hephaestus/actions/runs/41268735148), 1,150 passed/21 skipped |
| Actual zero-update staged failure recovery; `fix/staged-failed-run-recovery`; [#174](https://github.com/kys928/Hephaestus/pull/174) | `c4b82647407365532382d8359f96c1d6c54da6b3` | [Full CPU 41269225629](https://github.com/kys928/Hephaestus/actions/runs/41269225629) |
| Durable optimizer fence and honest transition boundary; `fix/recovery-fence-transition-contract`; [#175](https://github.com/kys928/Hephaestus/pull/175) | `ad1fc2da55d23fcf0bbdc64b79f6aeb5125eccee` | [Full CPU 41270200854](https://github.com/kys928/Hephaestus/actions/runs/41270200854), 1,169 passed/21 skipped |
| Exact original public model/tokenizer recovery; `science/recover-original-public-source`; [#176](https://github.com/kys928/Hephaestus/pull/176) | `6e3ce80ea7707069d9ae00797dc46a1dd1571eb3` | [Final production 41270892410](https://github.com/kys928/Hephaestus/actions/runs/41270892410); subsequent combined full CPU passed |
| Closed SQLite readers, integrity and checked snapshots; `fix/sqlite-campaign-health`; [#178](https://github.com/kys928/Hephaestus/pull/178) | `d59173750263e4e991bad4b3a964d9e86fecae4e` | [Full CPU 41272396968](https://github.com/kys928/Hephaestus/actions/runs/41272396968) |
| Actual three-policy/three-seed multi-cycle research; `research/authentic-cpu-autonomy-comparison`; [#177](https://github.com/kys928/Hephaestus/pull/177) | `6a8aac2b75f9c9da674cccae5ce94ea2643dbfb1` | [Full CPU 41272930183](https://github.com/kys928/Hephaestus/actions/runs/41272930183), 1,177 passed/21 skipped; complete isolated cohort and fresh-process replay |

## Adversarial review and discovered follow-ups

The same engineering agent performed adversarial review; no independent human
review or external scientific certification is claimed. Probes covered
coherently rehashed replacement, prepared-action crash windows, cross-lineage
continuation, simultaneous controllers, killed owners, ambiguous external
side effects, long pending polls, genuine failure exhaustion, prepared-spec
substitution, nested duplicate keys/nonfinite floats/typed equality, template
drift, partial gradients, exact sample order and restart counters.

The first #169 full CI exposed a second text-only producer in
`development/existing_peft_bridge.py`. It was repaired before acceptance and
full CI was rerun. No failed CI result was treated as a pass.

Further integration findings are tracked separately from the eleven audit defects:

- Newly trained staged checkpoints lacked RunStore-based membership despite a
  retained producer commitment. #174 admits only matching lineage-bound
  commitments, still checking the retained digest before continuation.
- Terminal staged failure previously stopped before recovery advice. #174 now
  preserves the failed workflow and creates a bounded new attempt only with
  verified zero-update evidence and the existing eligible recovery decision.
- An atomic rename alone did not durably fence optimization, and state-only
  branch/restart actions lacked a next-experiment handoff. The follow-up flushes
  file/directory evidence and explicitly excludes those automatic transitions.
  See [staged recovery](staged-failure-recovery.md) for the supported scope.
- Shared-filesystem/cross-host fencing, unsupported shuffle and arbitrary
  automatic transition execution remain explicit unsupported capabilities.
  Missing evidence is not migrated through trust-on-first-use.
- Actual research uncovered leaked SQLite readers and redundant repeated
  evidence carry-forward. #178 closes readers and checks storage integrity
  before mutation, with consistent snapshots. Two workspace-resident cohorts
  failed persistence validation, including one after the closure fix. Their
  corruption root cause remains unassigned under [#179](https://github.com/kys928/Hephaestus/issues/179).
  The complete isolated cohort passed cold validation; this does not close
  the storage investigation or establish a general cure.

## Larger capabilities and external boundaries

Milestones 1 and 2 are implemented and integrated. #172 versions the prospective
distribution and power sensitivity and records authentic original source
recovery. #173 implements one-use custody, actual CPU generation and cold
readback: its full remote suite passed **1,150 tests, 21 skipped**. Its genuine
three-seed fixture rehearsal performed twelve updates; baseline and selected
candidate both scored 0/8 with eight candidate truncations. This is a null
engineering rehearsal, not confirmatory science.

#152 remains open. #176 recovered the exact original public base/tokenizer and
chat-template bytes; the original CPU adapter and 96/24 training/development
examples are also recovered and hash-verified. The separate private R10
archive remains unavailable, independent entrusted custody has not acted,
and the blinded power/EOS pilot and final design acceptance are outstanding.
No original-model confirmatory study was executed in this campaign.

Milestone 4's bounded native recovery is integrated in #174/#175. #177's
complete isolated cohort contains nine authentic programs, 27 sequential
native cycles and 108 actual updates. Fresh-process verification reexecuted
another 108 updates and reproduced raw inference and all optimizer state;
the original database was unchanged. All policies scored 0/16 exact answers.
Adaptive mean target NLL was lower than fixed by .029398 and heuristic by
.025000, but both exact seed sign tests gave p=.25 (Holm .50). The
[frozen plan](../scientific-reports/cpu-autonomy-comparison-plan-v1.md) and
[full campaign report](../scientific-reports/remediation-campaign-20261101.md)
record the controls, negative/inconclusive result and failed earlier attempts.
No useful behavioral gain or generally superior autonomous research is claimed.

#102 stays open for its genuine external historical certification obligations.
#151 remains closed for its accepted scoped integration. Current runbooks and
capability limitations are maintained under #159, preserving dated evidence.
