# First post-failure diagnosis integration note

## Scope

This integration closes the first real post-Judge diagnosis boundary without changing diagnosis rules or inventing a causal label.

The input chain is the already-frozen evidence from:

- training run `first-bounded-scientific-training-001-33866198758`;
- semantic evaluation run `first-semantic-evaluation-001-33869352751`;
- source experiment `experiment-60bff7cb4f478f91`;
- lineage `lineage-first-scientific`.

## Exact upstream identities

- training verification SHA-256: `sha256:c4b1873da483fb672c146248b6a9116af11065d4fa103658fac40bc7aab4774b`
- evaluation verification SHA-256: `sha256:4a3e266413a717c31e83f7f1d894e7ccd6e0bef1d59e9b1a427b67ecf178e8c4`
- processed dataset: `sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`
- tokenizer: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`
- random-init model: `sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39`
- trained checkpoint manifest: `sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3`
- frozen semantic pack hash: `ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad`

The runner refuses to diagnose if any upstream identity or chain assertion changes.

## Diagnostic projection

`scripts/run_first_post_failure_diagnosis.py` converts only facts explicitly supported by the frozen upstream records into `DiagnosisRequest.observed_failures`.

Supported observations include:

- content-hash-verified evaluation integrity;
- the observed semantic regression and failed deterministic gate;
- identical decoding configuration across baseline/candidate;
- completed/healthy runtime status;
- verified checkpoint integrity;
- finite/numerically stable training metrics;
- verified processed-data and tokenizer identities;
- tokenizer compatibility;
- architecture compatibility established by successful strict loading/generation.

The projection deliberately does **not** emit `undertraining_detected`, `training_budget_exhausted`, `scheduler_misconfigured`, `overfitting_detected`, `data_coverage_gap`, `model_family_limitation_detected`, or any other causal-domain support signal unless such evidence is actually recorded.

## Successful diagnosis execution

GitHub Actions workflow: `33871942379`

Exact run commit: `1bd3c49ba2c52d58e46f20180d2a94586fd73e67`

All diagnosis-subsystem tests passed before the real frozen-chain diagnosis call.

Result:

- report ID: `diag-49904c2b7fa6cd1a`
- request ID: `diagnose-first-semantic-regression-001`
- status: `inconclusive`
- leading domain: `inconclusive`
- confidence: `0.0`
- issues: none
- missing baseline evidence categories: none
- causation claimed: false
- recommendation executed: false
- recommended intervention: `collect_more_evidence`
- required next test: `collect independent diagnostic evidence`

This is not a failed diagnosis execution. It is the expected conservative result when the failure is verified but the available records do not discriminate a cause.

## What the diagnosis ruled out vs. what remains unknown

The evidence positively supports evaluation integrity, decoding parity, checkpoint integrity, runtime health, numerical stability, tokenizer compatibility, and architecture compatibility. Those facts make several failure explanations less plausible, but the diagnosis service does not convert absence of a failure signal into proof of another cause.

The current evidence therefore does not distinguish among undertraining, optimizer/scheduler behavior, train/eval dynamics, data coverage/quality, model-family limitation, or other causes that require explicit targeted tests.

## Frozen proof

The successful workflow artifact ID is `9936251546` and its ZIP digest is:

`sha256:233a9e1400d2ae5f5b3f87afdded2cb1279341eb389757ca47e1577f207bc48c`

Exact output bytes are frozen under `docs/evidence/first-post-failure-diagnosis-001-33871942379/`:

- `report.json`: `sha256:57c5c0b4a31b5b7b033964ebad7178aa1fa342a9fa3fd15bf1759a87d648bece`
- `chain.json`: `sha256:b2f109293e87efd74baa7ba6667c2af11edcb8fb83708388dd286315e57e4adb`
- `input.json`: `sha256:ff4a5be479695d1c595ea5e0af92111f2452b2049891cefa4342e9fef28cf455`

No datasets, checkpoints, weights, generated semantic samples, secrets, or runtime logs are committed.

## Proven chain after this integration

`verified data/model/tokenizer → governed training → finalized checkpoint → frozen semantic generation/evaluation → Judge reject → evidence-based diagnosis`

The next correct boundary is targeted evidence collection/planning. Because diagnosis is inconclusive, the planner must not jump directly to another training intervention as though a cause were known.
