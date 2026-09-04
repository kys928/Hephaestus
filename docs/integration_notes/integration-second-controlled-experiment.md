# Integration note — second controlled experiment

## Purpose

Execute the Planner-approved one-primary-variable follow-up experiment after the diagnosed data-coverage failure.

The scientific comparison is:

- baseline training run: `first-bounded-scientific-training-001-33866198758`;
- candidate training run: `planned-run-b8e558e54effac85`;
- primary variable: `dataset_mixture`;
- baseline dataset: the original bounded Wikitext preparation;
- candidate dataset: the approved, immutable, prepared `sail/symbolic-instruction-tuning` intervention;
- model initialization, architecture, tokenizer, seed, optimizer, scheduler, LR, warmup, batch size, context length, precision, checkpoint cadence, shuffle policy, step budget, decoding settings, and frozen eval pack remain controlled.

## Fixed scientific identities

- experiment: `experiment-d0e911d6bd1fb7ae`
- lineage: `lineage-first-scientific`
- stage: `smoke_test`
- model/random-init identity: `sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39`
- tokenizer identity: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`
- previous trained baseline checkpoint manifest: `sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3`
- candidate processed data: `sha256:bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`
- candidate DatasetManifest: `sha256:0495018a0cc7c70494d5a00bc51a471568e850d8e3fa11cb0696c9674c71cc76`
- candidate TrainableDataContract: `sha256:ef273fe913f582289ffad2cd05a431e9d541091a51db97b0a649eb47579f2a5a`
- frozen semantic pack content hash: `ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad`

## Controlled training recipe

The candidate uses the same bounded v2 recipe as the first run:

- 100 optimizer steps
- batch size 8
- context length 256
- AdamW
- linear scheduler
- learning rate 0.0005
- warmup 10 steps
- weight decay 0.01
- gradient clipping 1.0
- float32 CUDA
- seed 1729
- deterministic sequential loading (`shuffle=false`)
- checkpoint at step 100

The larger lifecycle byte/row guards only accommodate the already-verified prepared artifact; they are execution-capacity bounds, not scientific recipe variables.

## Execution recovery boundaries

The first real candidate-training Pod reached a healthy RTX 4090/CUDA runtime but failed before optimizer step 1 because the generic worker tokenization guard was `max_total_tokens=20_000_000`, while the immutable prepared dataset declares 32,325,317 tokens. No checkpoint or non-zero optimizer-step evidence existed. The failed run directory is therefore eligible for audited archival before retry, and the admission-only token ceiling is raised to 40,000,000. This does not alter the fixed first 100 sequential batches, model initialization, optimizer, or any other scientific variable.

The Network Volume execution sentinel is also attempt-aware. Before retry, stale `executions/<run-id>/driver_result.json` and `pod_runtime.log` are copied to a content-addressed failed-attempt archive and round-trip verified before the canonical stale keys are removed. During monitoring, a terminal record is accepted only when its `repo_sha` equals the exact current workflow commit. This prevents a prior failed attempt from being mistaken for the current launch.

## Evaluation contract

After successful training and independent checkpoint verification, a separate GPU evaluation Pod generates the same 18 frozen semantic task/seed outputs for each of:

1. the previous Wikitext-trained checkpoint (baseline), and
2. the new instruction-data-trained checkpoint (candidate).

`ExperimentEvaluationService` compares those two trained runs under `semantic_behavior_v1`, and `SemanticComparisonJudgeAdapter` records the governed Judge exit. The Judge action is not applied by this workflow.

## Runtime-only representation adapters

Two narrow adapters preserve source evidence while satisfying concrete runtime schemas:

- the selected DatasetManifest stores its processed SHA on its single dataset entry, so the training runtime projects that value as a read-only convenience property for invariant checking;
- the frozen Planner proposal records baseline provenance as `run://<run-id>`, while `ExperimentEvaluationService` expects the concrete `TrainingRunHandle.run_id`, so evaluation verifies the frozen URI then projects it to the exact baseline run ID.

Neither adapter mutates the frozen source evidence, eval pack, model, tokenizer, or scientific variables.

## Approval and action boundary

The user explicitly requested this real controlled training and semantic evaluation on 2026-09-04. The workflow records:

`approval://operator/chat-2026-09-04-second-controlled-experiment`

The workflow may launch training and evaluation compute, but it must not automatically apply the Judge action, promote a checkpoint, delete a rejected checkpoint, or launch another follow-up experiment.
