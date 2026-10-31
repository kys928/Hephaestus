# Pinned-baseline -> actual generation/evaluation staged handoff (CPU)

Parent: draft PR #92, SHA `77988696ddbe46d0789831f97e74b0fa32a87562`.

The single existing production loop remains the workflow owner. No new
orchestrator, Judge, scoring algorithm or inference backend is introduced.

## Real operational integration

`StagedVerifiedEvaluation` binds a **SHA-256 content-addressed**
`verified-baseline-run.v1` artifact to the finalized candidate experiment
proposal from the real immutable staged record reader. The training lifecycle
must report `completed` and match both the expected experiment identity and
persisted completed-poll checkpoint. A finalized checkpoint manifest is
verified against **actual component bytes** before any generation is admitted.

Production composition injects the existing `EvaluationGenerationService`,
`StagedGenerationAdapter`, `StagedExperimentEvaluationAdapter`, and
`ExperimentEvaluationService`. The generation backend is an **explicit
dependency**. The service is **disabled by default** and cannot infer unless
the operator enables generation.

The existing generation/evaluation implementations provide:
- Frozen evaluation-pack prompts and identical baseline/candidate settings.
- Persistent, independently hashed generation receipts and task samples.
- Exact baseline vs candidate comparison; deterministic and repeatability
  evidence are calculated from the generated samples.
- Fresh-controller restart reconstruction from persisted state, without
  requiring in-memory generation objects or repeating completed inference.

### Verification

GitHub Actions CPU run #40175778575: **3 focused passed; 607 full passed,
9 skipped**.

The tests exercise the real composition root, immutable artifacts, generation
service, evaluation service, checkpoint verification and SQLite state with
**explicit simulated training and deterministic sample-generation workers**.
They do not represent live base-model inference, PEFT improvement, scientific
certification, GPU readiness or active model replacement.

### Remaining boundaries and limitations

- The pinned baseline run manifest is immutable, but the *authority selecting*
  that manifest must be independently governed (frozen parent/certification
  registry). A caller-provided SHA reference by itself is insufficient to
  prove scientific baseline authenticity. The baseline's loading instructions
  and checkpoint files must also be independently authenticated by a
  production model-loading service and certificate/manifest policy.
- `StagedVerifiedEvaluation` currently rehashes the complete candidate
  checkpoint on each evaluator substep. This is conservative for integrity,
  but expensive for multi-GB optimizer state; optimize with verified immutable
  artifact receipts without weakening integrity before long GPU campaigns.
- `ProductionRuntime.with_evaluation_services` must be called **after**
  `with_training_services` to preserve the owner of checkpoint resolution.
  Production composition still must provide source acquisition,
  preprocessing, and finalized training inputs before this stage can be used.
- The frozen `semantic_behavior_v1` pack is a general semantic evaluator;
  it is **not** the independent D scientific-state holdout. The scientific
  scorer/activation gates in PRs #89–90 remain separate and unproven against
  real unseen examples.
- No automatic model dispatch, approval bypass, paid GPU, Pod creation, S3
  write, or protected production routing change was made.
