# Integration note: first targeted post-failure diagnostics

## Purpose

This integration closes the first `collect_more_evidence` continuation after the
rejected bounded scientific checkpoint. It gathers targeted measurements from
the exact failed run, reruns the deterministic diagnosis subsystem, and hands
the resolved evidence into the closed-loop Planner.

It does not train, mutate a model, mutate a dataset, promote a checkpoint, apply
a Judge action, or execute a Planner intervention.

## Immutable upstream chain

- training run: `first-bounded-scientific-training-001-33866198758`
- semantic evaluation: `first-semantic-evaluation-001-33869352751`
- first post-failure diagnosis: `diag-49904c2b7fa6cd1a`
- first diagnosis status: `inconclusive`
- first diagnosis recommendation: `collect_more_evidence`
- checkpoint manifest: `sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3`
- model identity: `sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39`
- tokenizer identity: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`
- processed dataset identity: `sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`
- frozen eval pack: `semantic_behavior_v1@1.0.0`
- eval-pack content hash: `ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad`

## Targeted probes

`PostFailureDiagnosticProbe` adds two measured, reusable evidence producers.

### Training dynamics

The probe read the full persisted `metrics.jsonl` plus the exact normalized
training configuration from the RunPod Network Volume.

Observed:

- metric points: `20`
- first logged step: `5`
- last logged step: `100`
- first logged loss: `8.864677429199219`
- final loss: `7.302044868469238`
- total loss drop: `1.5626325607299805`
- total loss drop fraction: `0.1762763025739493`
- final epoch fraction: `0.014577790735813988`
- final six-point loss slope: `+0.0019888196672712055` loss/step
- minimum gradient norm: `0.7821573615074158`
- maximum gradient norm: `2.80792236328125`
- configured base learning rate: `0.0005`
- final learning rate: `0.0`
- scheduler: `linear`
- warmup steps: `10`
- scheduler trace conformed to the recorded warmup/linear-decay contract: `true`

The bounded run had consumed only about 1.46% of an epoch, and loss was materially
lower overall. However, the measured tail was no longer improving under the
probe's conservative rule. Therefore this evidence did **not** emit
`undertraining_detected` or `training_budget_exhausted`.

The learning-rate trace, finite metrics, and gradient evidence also did not
support an optimizer/scheduler pathology. The probe emitted `optimizer_stable`
and `numerically_stable`, and did not emit `scheduler_misconfigured`.

This does not prove that a longer run could not improve. It means this evidence
is insufficient to diagnose undertraining as the current leading failure domain.

### Training-data vs evaluation-task coverage

The probe scanned the exact processed JSONL object byte-verified as
`sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`.

Observed:

- rows scanned: `54,878`
- plain text rows: `54,878`
- prompt/target supervised rows: `0`
- structured-target rows: `0`
- exact frozen-eval prompt hits: `0`
- rows containing instruction-like cue text: `29`

The frozen `semantic_behavior_v1` pack explicitly evaluates instruction
adherence, JSON structure, bounded response length, termination, repetition, and
continuation behavior. The training data contains no explicit prompt/target or
structured supervision for those task forms.

The probe therefore emitted `data_coverage_gap`. This is structural coverage
evidence. It is not a claim that every future successful model must train on
these exact prompts, nor is it proof that the coverage gap caused the regression.

## Resolved diagnosis

After adding the targeted probe records to the exact prior diagnosis evidence,
`EvidenceBasedDiagnosisService` returned:

- status: `completed`
- leading domain: `data_coverage`
- confidence: `0.825`
- causation claimed: `false`

The correct interpretation is: **data coverage is the leading supported failure
hypothesis under the currently measured evidence.** Undertraining and
optimizer/scheduler pathology were not supported by these probes.

Still unresolved by this run:

- model-family limitation, which requires a controlled model-family comparison;
- overfitting, which requires held-out train/eval-gap evidence.

## Planner continuation

The first live integration exposed a same-cycle loop: once
`collect_more_evidence` had been fulfilled and diagnosis had become actionable,
the base Planner's heuristic score still ranked another `collect_more_evidence`
proposal above the actionable data intervention.

`ResolvedEvidenceExperimentPlanner` adds a narrow continuation rule. A diagnosis
producer can mark an intervention kind as fulfilled in
`metadata.resolved_intervention_kinds`. Only when diagnosis is `completed`, has
no missing evidence, and has no blocking issues does the wrapper remove that
already-fulfilled intervention from the same evidence cycle. It preserves the
base Planner's ordering among all remaining proposals.

With that boundary applied, Planner selected:

- intervention: `replace_or_mix_dataset`
- single primary variable: `dataset_mixture`

The Planner then emitted the governed `DatasetSearchRequest` and stopped at:

`dataset_discovery_and_selection`

No `ExperimentProposal` was fabricated because `replace_or_mix_dataset` requires
a real selected dataset decision first. No training was launched.

## Execution evidence

Final targeted workflow:

- GitHub Actions run: `33874832481`
- exact run commit: `38c19d3a5a6b791d29627c08a8a08b58fabc0a31`
- artifact ID: `9937410950`
- artifact ZIP SHA-256: `15566ca3c6adb9818ffba361ebc4ddbe064deec67eb191ed63674f161888a206`

Exact output hashes:

- diagnostics: `5d14b98087783fcb6b87264ec66ad8c60e0f6db6bd3e8443bfffbb6b3337cf00`
- probe: `c943334f2f6e7162380be74748506d26dd13542fece21d42672d35f3abbfc2e9`
- diagnosis: `11cb67960df8c9f60a861e57b4f74145ad64cd25f5b52bb3835c754cde74f086`
- planner: `52e6db0f5c4f9297e763ed11581f98a6e7c92af45e45ca5e6922504e8f9b3325`

The one-shot freezer independently reproduced those exact bytes before committing
them under:

`docs/evidence/first-targeted-post-failure-diagnostics-001-33874832481/`

The temporary freezer workflow was deleted before integration review.

## Proven continuation

The first executed failure chain now reaches:

`training -> semantic evaluation -> Judge reject -> conservative diagnosis -> collect_more_evidence -> targeted probes -> actionable diagnosis -> one-variable Planner decision -> dataset discovery boundary`

The next correct boundary is governed dataset discovery and selection for the
`dataset_mixture` intervention. Only after an evidence-backed dataset selection
exists may Planner construct the final one-primary-variable `ExperimentProposal`.
