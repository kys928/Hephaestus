# Semantic evaluation integration note

## Scope

This branch adds an offline, evidence-based baseline-versus-candidate comparison service. It evaluates recorded generation artifacts and returns the shared `ExperimentComparison` contract. It does not launch generation or training, select or promote a checkpoint, mutate lineage, or persist approval/replay records.

## Evaluator service

Class: `hephaestus.evaluation.ExperimentEvaluationService`

Constructor inputs:

- `pack_name`: defaults to `semantic_behavior_v1`.
- `config_dir`: defaults to `configs`.
- `judge_adapter`: optional dependency implementing `hephaestus.evaluators.JudgeModelAdapter`.
- `minimum_practical_improvement`: optional explicit override of the pack threshold.

The public method is the shared interface shape:

```python
compare(
    proposal: ExperimentProposal,
    runs: Sequence[TrainingRunHandle],
) -> ExperimentComparison
```

`proposal.baseline_ref` must name one supplied run. Every other supplied run is treated as a candidate and must carry `proposal.experiment_id`.

## Pack identity and version

New pack:

- pack name: `semantic_behavior_v1`
- stable ID: `semantic_behavior`
- version: `1.0.0`
- integrity: `sha256-canonical-json-v1`
- frozen: `true`
- mutation policy: `new_version_required`

The pack defines generation, continuation, structured-output, repetition, length, and termination tasks. It also records task bundles, stage applicability, success/failure semantics, expected evidence, repeated-seed requirements, deterministic gate semantics, decoding settings, and practical comparison thresholds.

The loader verifies the canonical SHA-256 value before returning `content_hash_verified`. Any behavioral change requires a new pack file/version and a new hash; do not edit this frozen version in place.

## Required run artifacts

Each `TrainingRunHandle.metadata` must contain:

```json
{
  "semantic_evaluation": {
    "eval_pack_id": "semantic_behavior",
    "eval_pack_version": "1.0.0",
    "integrity_level": "content_hash_verified",
    "content_hash": "<pack hash>",
    "decoding_config": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_new_tokens": 96,
      "seeds": [11, 29, 47]
    },
    "report_ref": "reports/eval-<run>.json",
    "evidence_refs": ["artifacts/<run>/semantic-evidence.json"],
    "samples": [
      {
        "task_id": "instruction_triplet",
        "seed": 11,
        "output": "alpha beta gamma.",
        "evidence_ref": "artifacts/<run>/instruction_triplet-11.json"
      }
    ]
  }
}
```

The current adapter accepts the recorded sample payload from metadata so offline fixtures need no store or network. Production wiring should load the referenced evaluation artifact, construct the same metadata view at the evaluator boundary, and keep the persisted comparison limited to artifact references. Training loss and `metrics_ref` are not used as semantic-quality evidence.

## Comparison configuration

The frozen pack currently requires:

- three decoding seeds per task;
- exact decoding-setting equality between baseline, candidates, and pack;
- matching eval-pack ID, version, and verified content hash;
- a minimum practical improvement of `0.05`;
- minimum direction consistency of `0.67`;
- explicit moderate/high spread thresholds;
- two candidate runs for full repeatability confidence.

Aggregation reports observed count, mean, minimum, maximum, spread, and population standard deviation. These are bounded descriptive summaries. The service explicitly records `formal_significance_claimed=false`; it does not claim statistical significance from small samples.

## Output contract

The service fills the existing `ExperimentComparison` fields:

- experiment, baseline, and candidate identities;
- evaluation-report and evidence references;
- `primary_outcome`;
- per-dimension and overall effect summaries;
- deterministic gate status;
- variance risk and direction consistency;
- confidence and its ceiling;
- a non-executing recommendation;
- issues and missing evidence;
- a human-review bundle manifest referencing baseline/candidate output artifacts.

Primary outcomes are:

- `improved`
- `regressed`
- `equivalent_within_evidence`
- `mixed`
- `inconclusive`
- `invalid_comparison`

Recommendations are advisory evidence labels. Even `consider_candidate_after_human_review` is not a promotion command.

## Deterministic gates

The response checker supports:

- normalized exact match;
- required and forbidden terms;
- minimum/maximum word counts;
- n-gram repetition ceilings;
- termination and abrupt-ending checks;
- continuation prompt-echo detection;
- JSON object, required-key, type, and expected-value checks;
- bounded surface-coherence checks.

Hard candidate failures force `deterministic_gate_status=failed` and prevent an `improved` conclusion. Judge scores cannot clear a hard failure. Missing required task/seed evidence produces an incomplete, inconclusive comparison.

## Optional judge adapter

`JudgeModelAdapter.score(task, response)` may return bounded scores for `instruction_adherence`, `relevance`, and `coherence`. The adapter is dependency-injected and can be faked without network access.

Deterministic/judge disagreement is preserved under `effect_summary.judge.disagreements`. Favorable judge evidence never overrides a hard deterministic regression. Adapter failures become retryable `provider_unavailable` issues and do not silently pass.

## Failure modes

The comparison is invalid when:

- the baseline or every candidate is missing;
- run IDs are duplicated;
- a candidate belongs to a different experiment;
- semantic evaluation metadata is absent;
- pack identity/version is missing or mismatched;
- required integrity evidence/content hash is absent or mismatched;
- decoding settings differ.

The comparison is inconclusive when required samples or sample evidence references are missing, or when a positive effect is too inconsistent or variable. Weaker but honestly classified pack integrity lowers the confidence ceiling. Missing report references are surfaced without inventing one.

## Fixtures and validation

`tests/test_semantic_evaluation.py` contains offline baseline/candidate fixtures and a fake judge. Coverage includes:

- valid improvement;
- repetition and termination regressions;
- mixed dimension effects;
- variance-sensitive confidence;
- decoding/pack mismatch invalidation;
- missing-sample handling;
- frozen/versioned/hash-verified pack metadata;
- malformed structured output;
- deterministic/judge disagreement.

No test requires a network or model provider.

## Known limitations

- Surface coherence and keyword relevance are deterministic proxies, not complete semantic judgments.
- The service reports descriptive uncertainty, not formal significance.
- It evaluates pre-recorded outputs; it does not invoke a model generation backend.
- Human-review bundle persistence remains the final integration layer's responsibility.
- Ranking-set execution is structurally supported by the pack/task loader, but the v1 pack does not yet include a ranking task.
- This branch does not wire the service into the orchestrator or existing evaluator role.

## Final wiring instructions

The final integration branch should:

1. Resolve the frozen eval pack before generation.
2. Run every pack task with the exact recorded decoding configuration and seeds.
3. Persist each output and the aggregate evaluation report, retaining content/integrity metadata.
4. Populate the `semantic_evaluation` metadata view on baseline and candidate `TrainingRunHandle` records.
5. Inject an optional judge adapter only when configured; offline deterministic comparison must remain valid without it.
6. Call `ExperimentEvaluationService.compare` during the evaluator phase.
7. Persist the returned comparison and human-review bundle manifest through the integration-owned state path.
8. Pass the comparison to Judge exit as evidence only. Existing promotion, approval, lineage, and replay gates remain authoritative.
