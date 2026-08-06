# Integration note: production autonomy continuation wave

## Branch and source work

Integration branch: `integration/production-autonomy`.

The branch began from merged PR #42 commit `552200b95b027ed91ac76167619ac9478100e811` and composes:

- PR #47 — durable production execution adapters;
- PR #46 — governed production dataset acquisition;
- PR #43 — optional real Transformers causal-LM training;
- PR #45 — bounded autonomous recovery;
- PR #44 — opt-in staged autonomous orchestration.

The intended generation continuation branch had no commits and no pull request. Review therefore treated the training-to-evaluation generation bridge as missing functionality and implemented it on this integration branch rather than claiming that six completed branches existed.

## Review findings

The five published branches respected their exclusive ownership boundaries and did not overlap in implementation paths. Their individual tests were useful but did not prove composed behavior.

Two material gaps were found during composition:

1. training produced strict checkpoint loading instructions, while no service consumed them to create the `semantic_evaluation` evidence required by `ExperimentEvaluationService`;
2. the optional real training implementation had no package extra, making intentional clean installation of its dependencies undocumented and non-reproducible.

The integration branch adds the missing bridge and `training`, `generation`, and combined `ml` optional dependency groups.

## Generation bridge

`hephaestus.generation.EvaluationGenerationService`:

- loads the frozen evaluation pack through the existing verified loader;
- requires a verified pack content hash;
- materializes every task and explicit seed without changing prompts or decoding settings;
- creates stable generation-settings, seed-set, task/seed, report, and sample identities;
- requires a completed run, concrete checkpoint, and explicit generation handoff;
- consumes the real training backend's `loading_instructions.json`;
- persists sample-level JSON evidence atomically;
- verifies cached sample identity and output hashes before reuse;
- quarantines corrupt cached samples and regenerates only missing evidence;
- attaches the exact `TrainingRunHandle.metadata["semantic_evaluation"]` structure consumed by `ExperimentEvaluationService`;
- records generation evidence but does not score, promote, approve, or mutate Judge decisions.

`DeterministicFakeGenerationBackend` provides bounded offline tests. `TransformersCausalLMGenerationBackend` lazily imports optional dependencies, loads only local finalized model/tokenizer directories, disables remote code, and performs no implicit network acquisition.

## Staged integration

`StagedGenerationAdapter` supplies:

- frozen prompt materialization;
- baseline generation;
- candidate generation;
- matching `generation_settings_id` and `seed_identity` evidence;
- required `generation_report` records.

`StagedExperimentEvaluationAdapter` supplies:

- concrete candidate checkpoint resolution;
- direct baseline/candidate handoff to `ExperimentEvaluationService`;
- `experiment_comparison`;
- deterministic regression evidence;
- repeatability/variance evidence;
- human-review references.

These are injected operation adapters. They do not alter `SPINE_ORDER`, add a ninth role, or replace Judge exit.

## Integrity and parity

The bridge uses the frozen `semantic_behavior_v1` pack exactly as loaded by the evaluator. Baseline and candidate generation share a settings identity derived only from the generation protocol, pack identity/version/hash, and frozen decoding configuration. Checkpoint identity is intentionally excluded from the settings identity and retained separately in each sample.

Every required task/seed pair must exist before a report is complete. Backend failures or missing samples produce partial evidence and blocking issues rather than a false completion.

## Optional dependencies

Core imports remain dependency-free. Real model training or inference may be installed through:

```bash
python -m pip install '.[training]'
python -m pip install '.[generation]'
# or
python -m pip install '.[ml]'
```

All three currently install pinned minimum versions of PyTorch, Transformers, and Tokenizers. Required CI does not download a model and does not require those packages; optional real-model tests remain separately gated.

## Honest limitations

- The integration suite proves the generation/evaluation contract with a deterministic fake backend. Real optional Transformers generation must be exercised in an environment containing the optional ML dependencies and a finalized local checkpoint.
- The required CI does not establish large-model quality, throughput, GPU behavior, or distributed training.
- Production PostgreSQL, object storage, secret managers, and telemetry backends remain optional and were not exercised against live external services by the feature branches.
- The staged orchestrator is opt-in. A deployment composition root must still provide concrete diagnosis, provider, selection, acquisition, preprocessing, model selection, lifecycle, runtime, recovery, Judge, approval, and action adapters.
- The SQLite execution baseline is one-host, multi-process and at-least-once; it is not exactly-once or cross-host consensus.

## Merge and cleanup guidance

The final pull request should remain unmerged until its composed GitHub Actions suite passes and reviewers confirm that no generated data, checkpoints, weights, state, caches, logs, or secrets are present.

Feature branches merged into this integration branch and the unused generation feature branch may be deleted after the final integration branch and pull request are safely published. The integration branch must remain until the final pull request is resolved.
