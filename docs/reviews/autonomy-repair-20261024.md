# Autonomous recovery repairs — evidence ledger

Baseline inspected: PR #81 `796785b87797c6019499a03d1ec80fd2bc31f87e`,
PR #82 `f1a8470e5b0f6e5b296eca1c2b7d705878d98bfd`, main
`304875f0c7d65650553aafcce608b5d345d93d9d`. Both PRs were draft/unmerged.
PR #82 CI runs 40082119343/40082119348 passed. Independent local baseline:
485 passed, 6 skipped. No weights, cloud jobs, training or production activation.

## Governance repair

- P0-02 **confirmed**: nonempty caller approval strings and a promotion boolean
  authorized sensitive actions. Seven initial adversarial cases reproduced this
  and contract/vocabulary defects before changes.
- Executor now resolves an actual `DecisionStore` approval event, requires its
  latest decision/request, operator identity, correct outcome/type, expiry,
  run/lineage/action/checkpoint/child scope and current lineage digest. Approval
  requests can be prepared using `build_action_approval_request`; this never
  grants approval. Legacy unscoped approvals intentionally fail closed.
- Promotion re-reads hash-addressed evaluation/manifest bytes, verifies the
  candidate's persisted run/lineage and recalculates promotion and certification
  policy. Caller booleans, status, confidence and certification claims cannot
  grant promotion. This is lineage metadata mutation, never active-role routing.
- P1-07 **confirmed**: Controller `noop` was rejected as an unknown action. It is
  now explicitly non-executable, with no expansion of the action registry.
- P1-11 **confirmed, single-host scope**: all lineage writes and action admission,
  preparation, recovery and application share a kernel-owned process lock.
  No lease expiry can permit a paused old writer to continue past takeover.
  Startup reconciles prepared intents. Before/after divergence blocks recovery.
  SQLite and JSON remain separate stores; no distributed atomicity is claimed.
- P1-12 **confirmed**: entire runtime contract is independently hash-pinned;
  returned profiles cannot mutate live rules. Base identities/revisions are
  pinned to original training config. `scripts/verify_frozen_role_contract.py`
  independently compares five profiles and byte-for-byte prompt rendering with
  immutable training commit c831bbfc0e2054d54fccd1ef374fd84722612c79.
  Adapter configuration compatibility remains in the subsequent cache repair.

Validation: full CPU regression **503 passed, 6 skipped**. Dedicated tests cover
forged, stale, expired, superseded and mismatched grants, unknown operators,
changed targets, valid retries, cross-process writer exclusion, startup healing,
existing mid-branch and post-write crashes, mutable profiles and contract edits.

Deployment limits: local-filesystem flock semantics are required. Multi-host
controllers remain unsupported. The approval store is a trusted operator control
plane; access to append decisions must be restricted by deployment. No adversary
with arbitrary filesystem/source write access is modeled as an untrusted caller.
Real inference, PEFT continuation and independent scientific certification remain
unverified. The remaining findings are tracked in subsequent repair commits.

## Evidence repair

- P0-03 **confirmed**: only a protocol and synthetic test binder existed. A real
  `ArtifactModelEvidenceBinder` now resolves hash-addressed source envelopes,
  checks source run/lineage/stage/kind, derives bounded context from actual JSON
  bytes and retains source identities across cycles. It never promotes model
  prose or JSON claims into verified approval/checkpoint facts. Missing,
  corrupted, superseded and identity-mismatched sources fail closed.
- Workflow records can be sealed into immutable artifacts by the staged
  orchestrator. The production driver supplies its real artifact store, and the
  composition root supplies the binder/evidence store. Cross-cycle bindings are
  explicit; full provenance stays durable while model context is bounded.
- P0-05 **confirmed**: restored metadata previously admitted edited samples.
  Generation now emits immutable report/sample receipts. Restoration verifies
  source bytes, frozen pack/settings/seed identity, checkpoint/loading handoff,
  exact task/seed coverage and output/prompt hashes. Comparison restoration
  re-reads an immutable receipt and recomputes deterministic scoring from those
  verified samples, without repeated generation. Legacy metadata without receipts
  fails closed; it is not silently re-certified.
- P1-10 **confirmed**: role inference now requires an artifact store and typed
  generation result. Receipts preserve the exact prompt/raw output, parsed answer,
  model/adapter/tokenizer identities, profile, seed, finish reason and execution
  provenance. The actual GPU generator also records rendered input tokens/text,
  output token IDs, effective decoding, package/CUDA/device metadata and timing.
  CPU test-double receipts explicitly identify themselves. Invalid raw outputs
  are preserved as rejected receipts. No actual GPU validation is claimed.
- Additional confirmed defect: the operator GPU probe verified file hashes but
  substituted a separate caller-written fact string. Context now comes directly
  from the verified source JSON; misleading prose is ignored.

Validation: **519 passed, 6 skipped** locally. Dedicated evidence tests exercise
actual immutable filesystem bytes, source tampering (including recomputed local
output hashes), missing samples, forged persisted metadata, changed checkpoints
and seeds, comparison edits, immutable-store and local-path restoration, source
supersession, cross-cycle provenance and raw-output/decision replay. Existing
service doubles were updated to implement the artifact/result interfaces; no
production fallback or scientific assertion was weakened.

## Scientific selection and experiment memory

- P0-01 **confirmed**: Planner inference was advisory and ran after delegation.
  Model decisions now pass an explicit frozen-vocabulary translation and the
  existing `ClosedLoopExperimentPlanner` admission/ranking policy. Admitted
  selections change the real experiment proposal and discovery requests. Sources,
  model receipt, controls, budget and required approvals remain attached.
  Unsupported, out-of-scope, ungrounded or budgetless training proposals block.
- P1-06 **confirmed, role-specific fix**: Controller/Planner inference and
  admission happen before delegated execution. An invalid Controller output
  produces zero delegate calls. Diagnosis/Evaluator/Judge interpret deterministic
  observations; action mutation remains separate and approval-gated. Diagnosis
  may select a supported hypothesis or declare inconclusive; it cannot change
  observations, erase missing evidence or invent a verified cause.
- P1-07 **additional confirmed mismatch**: domain `resume_training` incorrectly
  named checkpoint position as the primary changed variable. Continuation now
  changes training duration, matching the frozen role contract while preserving
  baseline/checkpoint controls. Every frozen Planner decision has a deliberate
  action/variable translation; evaluator recommendations remain non-executable.
- P1-08 **confirmed**: production now injects state and verified bound-evidence
  diagnosis adapters plus a durable experiment-memory query. The query combines
  existing legacy memory with verified experiment/comparison records; it does
  not create another memory database. A resumed runtime recalls an observed
  failed controlled intervention and refuses repetition unless new evidence or
  changed controls justify it. Failure records do not assert causality.
- Integration records now receive immutable source receipts so experiment memory
  can check actual content; unsealed historical rows are not failure proof.

Validation: **531 passed, 6 skipped** locally. Tests execute the real staged model
composition, production binder, admission and proposal code with explicit CPU
model/service doubles. Changing only the model choice changes the experiment ID
and primary variable. Other tests cover invalid action pairs, all frozen Planner
translations, bounded continuation, missing budget, campaign scope, persistent
failed-history retrieval and new-evidence retries.

Boundary: designs requiring new model/dataset discovery are explicitly pending
until actual validated selections are supplied to the existing planner/coordinator
finalization interface. A forged `approved` status cannot launch an unresolved
design through the guarded training service. This repair does not establish a
live, fully composed autonomous training campaign or real model reasoning quality.

## External allocation, adapter publication and final boundary review

- P0-04 **confirmed and reproduced**: provider acceptance followed by process
  death left no persisted external ID. The existing RunPod adapter now writes a
  durable intent first, attaches stable operation/request markers, discovers and
  validates existing resources after uncertainty, and refuses another POST.
  Missing/duplicate/mismatched resources receive bounded reconciliation, then
  require operator review. Neither a timeout nor an empty listing proves that
  allocation failed. Intent records contain hashes/identities, never API keys or
  environment values. Provider receipt write failures follow the same path.
- Hard-stop testing found an **additional actual defect**: a persisted `running`
  workflow could inherit `resumable=false` from the previous completed step and
  remain permanently stuck. In-flight steps are now restartable. Provider
  uncertainty has an explicit interrupted/resumable control result, with the
  same stable staged operation identity; exhaustion blocks further progress.
- P1-09 **confirmed**: adapter replacement could delete a directory during model
  loading. Materialization now has a process-safe lock, immutable generation
  paths and atomic pointer publication. Previously returned paths are retained,
  including corrupt generations. No background garbage collection may remove
  them while loaders run. Checksums in a mutable cache receipt are independently
  checked against the SHA-pinned archive; editing weights AND the receipt fails.
- P1-12 **remaining compatibility repair implemented**: PEFT configuration must
  match the certified base model and revision. Both repository identifiers and
  original HF snapshot paths are supported, including legitimate null revision
  fields when the pinned snapshot/archive establishes identity. Substitution or
  wrong revisions block before inference. All five synthetic CPU archives now
  declare their actual role-specific base identities.
- Final review strengthened P0-02: approval decisions pin the exact reviewed
  request digest. Promotion rehashes the existing finalized checkpoint manifest,
  reads the active stage and independently frozen eval pack, rechecks scorecard
  identities/metrics/thresholds, recomputes deterministic gates and applies frozen
  repeatability/certification requirements. Caller flags cannot override them.
  Prepared but unapplied actions recheck authorization on restart; already-applied
  transitions only finish journal/mirror reconciliation.
- Final review strengthened P0-01/P0-03: using a selected plan re-derives its domain
  choice from the preserved raw model output and rechecks cited source bindings.
  Evidence supersession is durable; a newly bound replacement is available for
  future context while the retired source cannot authorize the old plan.

Final local validation: **563 passed, 6 skipped**, plus `git diff --check`, Python
compilation and the independent frozen-source contract verifier. Cross-process
cache tests publish one complete generation with one download. A real staged
orchestrator is reconstructed after simulated remote acceptance and sends only
one provider POST. These are real control/persistence/artifact implementations
against deterministic provider/model doubles, not live GPU or scientific trials.

## Per-finding adjudication and proof

All twelve findings were independently checked against source and callers. The
original frozen role rules were correct; the defect was missing integrity guards,
not an observed preexisting change to their wording.

| Finding | Verdict / root cause | Repair and principal proof | Remaining validation boundary |
| --- | --- | --- | --- |
| P0-01 Planner advisory only | Confirmed; model output did not enter selection | Existing Planner admits translated proposals; `test_changing_model_selection_changes_the_downstream_experiment` | Actual-weight reasoning and full campaign composition |
| P0-02 Caller authorization claims | Confirmed; raw strings/boolean granted transitions | Authoritative scoped/digest-bound grants and recomputed gates; `test_fabricated_approval_reference_cannot_authorize_branch`, adverse certification tests | Operator store/access controls remain trusted deployment authority |
| P0-03 No production binder | Confirmed; tested binder was synthetic | Actual source-byte binding and lifecycle; `test_production_binder_derives_context_from_verified_source_and_retains_run_provenance`, supersession tests | Live S3 campaign ingestion not exercised |
| P0-04 Duplicate remote allocation | Confirmed; post-acceptance/pre-receipt window | Durable intent and reconciliation; `test_staged_controller_recovers_acceptance_window_with_same_operation_identity` | Live provider behavior and durable mounted state deployment |
| P0-05 Metadata-only restoration | Confirmed; sample/report bytes not re-read | Immutable generation/comparison receipts and rescoring; `test_restored_generation_requires_authentic_complete_artifact_bytes` | Real generated scientific corpus not certified |
| P1-06 Inference after side effects | Confirmed for proposal stages | Proposal/admission precede execution; `test_invalid_controller_model_cannot_execute_its_delegate` | Observation delegates must keep their non-mutation contract |
| P1-07 Vocabulary mismatch | Confirmed; noop missing; continuation variable differed | Explicit non-execution/translation; `test_controller_noop_is_a_valid_non_executable_outcome`, complete Planner vocabulary test | No model vocabulary or action registry expansion |
| P1-08 Missing scientific dependencies | Confirmed; empty adapters/query | Actual durable composition; resumed experiment-memory and bound-diagnosis tests | Old unsealed rows cannot become scientific failure proof |
| P1-09 Cache race | Confirmed under concurrent materializers | Immutable publication/locking; `test_concurrent_processes_publish_one_complete_adapter_generation` and interrupted extraction | Old generations require controlled offline cleanup |
| P1-10 Missing inference replay | Confirmed; hashes lacked preserved inputs/outputs | Immutable typed receipts; `test_role_inference_receipt_replays_exact_prompt_raw_output_and_decision` | Actual GPU environment/decoding compatibility remains unmeasured |
| P1-11 Incomplete writer recovery | Confirmed; separate stores and no ownership fence | Kernel-owned single writer, startup reconciliation, expiry recheck; cross-process and crash tests | No multi-host transactional claim |
| P1-12 Unpinned contract/base compatibility | Confirmed guard gap; baseline text matched training | Canonical pin, independent frozen-source verifier, adapter/base/revision tests | HF base-weight byte authenticity and actual PEFT loading remain unproven |

## Reviewable changes and deployment blockers

The repair series is stacked on the existing draft #82/#81, without merging either:
#83 governance, #84 evidence, #85 model selection/memory, then the external
execution/cache/final-verification repair. Every stage has CPU regression evidence.
No production registry flags, credentials, frozen packs or configuration policies
were changed. Positive promotion tests use explicitly synthetic checkpoint bytes
and an isolated temporary approved profile; they never modify repository policy.

The stricter boundary intentionally refuses production certification through an
unhashed legacy eval pack. A real campaign needs an explicitly approved, correctly
hashed certification profile and genuinely unseen holdouts. That approval is not
manufactured by this repair.

Additional verified deployment issue: the existing provider contract/configuration
uses REST v1. RunPod's [official overview](https://docs.runpod.io/api-reference/overview)
announces its retirement on **15 November 2026**. The allocation repair preserves
that existing API contract; a separately reviewed v2 migration is required before
production deployment. Reconciliation fields were checked against RunPod's
[official list-Pods contract](https://docs.runpod.io/api-reference/pods/GET/pods).

Outstanding system milestones remain real five-model GPU inference validation,
governed PEFT continuation, complete durable campaign composition, independent
scientific certification, transactional role activation/rollback and an actual
self-repair demonstration. Nothing in this series proves those milestones.
