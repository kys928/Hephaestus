# Integration note: first real semantic evaluation

## Scope

This integration executes the first real baseline-versus-trained evaluation through the frozen `semantic_behavior_v1` generation/evaluation pipeline and routes the resulting `ExperimentComparison` into the existing finite Judge/promotion policy boundary.

It performs generation and evaluation only. It does not launch training, mutate model weights, apply a Judge action, promote/delete a checkpoint, or mutate lineage state.

## Source evidence and identities

The training-chain proof is already frozen in the repository under:

`docs/evidence/first-bounded-scientific-training-001-33866198758/`

The evaluated trained checkpoint is:

`/workspace/hephaestus/scientific/v1/runs/first-bounded-scientific-training-001-33866198758/checkpoint_step_100`

Its independently reproduced checkpoint-manifest identity is:

`sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3`

The baseline is the exact original random-initialized model selected for experiment `experiment-60bff7cb4f478f91`:

- model directory identity: `sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39`
- tokenizer directory identity: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`

For the generation bridge, the driver creates an evaluation-only checkpoint wrapper containing byte-for-byte copies of that model and tokenizer plus provenance/loading/manifest files. The wrapper is explicitly marked `trained=false` and validates through the existing checkpoint-manifest verifier. Its manifest identity for this evaluation is:

`sha256:97a8c65b2f475575aa57778c9ef08c6814023441648487e757c91992637eb441`

The wrapper exists only to satisfy the normal generation-handoff contract; it does not change the baseline model bytes.

## Frozen evaluation configuration

Evaluation pack:

- pack: `semantic_behavior_v1`
- stable ID: `semantic_behavior`
- version: `1.0.0`
- frozen pack hash: `ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad`
- generation settings ID: `generation-settings-2ba1dd1322d04d4925c36ac4`
- seed identity: `seed-set-8c3598742bd3db5b46521974`
- temperature: `0.0`
- top-p: `1.0`
- max new tokens: `96`
- seeds: `11`, `29`, `47`
- task/seed pairs per model: `18`
- minimum practical improvement: `0.05`
- minimum direction consistency: `0.67`

Both baseline and candidate used the same frozen task/seed plan and decoding identity.

## Judge integration boundary

The semantic evaluator intentionally returns an `ExperimentComparison` and advisory recommendation; it does not own promotion authority.

`src/hephaestus/control/semantic_judge.py` provides a narrow integration-owned adapter that keeps the existing policy boundary authoritative:

- hard deterministic failure or semantic regression becomes rejected evidence;
- only an `improved` comparison may enter the existing `PromotionPolicy`;
- equivalent, mixed, inconclusive, and invalid comparisons remain non-promotable;
- a human-review-required recommendation is never treated as fulfilled automatically;
- final finite action selection still goes through the existing `JudgePolicy`;
- the adapter returns a `JudgeExit` record but never applies the action.

The integration tests cover soft semantic regression, equivalent evidence, pending human review on an improved comparison, and hard deterministic failure.

## Live execution

Successful GitHub Actions run:

`33869352751`

Exact evaluation commit:

`5f4fe4ebf413355b8842555dcfa09c4feef404ed`

Evaluation run:

`first-semantic-evaluation-001-33869352751`

RunPod Pod:

`l9l8jdfqxy9fpq`

Preflight passed `19` focused semantic/generation/Judge tests and compiled all execution drivers before the RunPod create boundary.

A prior workflow attempt (`33869267313`) failed on the GitHub runner because the workflow referenced a nonexistent test filename. The GPU launch step was skipped, so that attempt created no Pod and incurred no GPU runtime.

## Generation evidence

Both generation reports completed successfully:

- random-init baseline samples: `18`
- trained candidate samples: `18`
- total independently re-read semantic samples: `36`

The external verifier independently re-read every persisted sample through RunPod S3 and reproduced every recorded output SHA-256.

Sample inventory identity:

`sha256:d92ac15a9131aa319c2ce0f1b2e1c0ab7adef86ed58d1061a64b20b817f4fb29`

The complete evaluation prefix contains `52` objects / `8,360,692` bytes with inventory identity:

`sha256:e620a0fc86816225bf7d7e1ce8b7d397503ec8022db660ad8132a5b2a6e5632b`

Important independently verified evidence files:

- baseline generation report: `sha256:33b5acc9d5bfafc678f7179eafb846531aa39a547c2b1a415776a58e95160bdf`
- candidate generation report: `sha256:3e882528b26bdff387a2ef1b104f2ee59bfbec1cd98b0f7d170dbdd72a0dc904`
- experiment comparison: `sha256:f59d9eabb563febf79d85425aa953bd5fd06acca4d6f4ec2b0d8c5cd58596c3e`
- human-review bundle: `sha256:0bf78d0bb5b6a9eabf71b1bf2eb9ed8e1714cc8756e5498eceda116af07fe2c6`
- Judge exit record: `sha256:008c657629f2ad7a19961255c144a056fdff43bb96c86daacd6fa355508d8eea`

## Scientific result

The trained checkpoint did **not** improve over its own random-initialized baseline under this frozen evaluation.

Comparison result:

- primary outcome: `regressed`
- deterministic gate status: `failed`
- recommendation: `reject_candidate_evidence`
- variance risk: `low`
- missing evidence: none
- blocking comparison issues: none
- confidence: `0.7071067811865476`
- confidence ceiling: `0.75` because no optional external judge-model adapter was configured
- formal statistical significance claimed: `false`

Aggregate frozen-pack score:

- random-init baseline mean: `0.19722222222222222`
- trained candidate mean: `0.03333333333333334`
- overall delta: `-0.16388888888888886`

Largest directional regressions:

- coherence: baseline `1.0` → candidate `0.0`, delta `-1.0`
- repetition: baseline `0.5` → candidate `0.0`, delta `-0.5`

Other reported dimensions were equivalent within the available deterministic evidence.

The candidate also produced hard deterministic failures across the frozen tasks, including exact/length failures on `instruction_triplet`, structure/length failures on `structured_planet_answer`, repetition/length/termination failures on `anti_repetition`, length/termination failures on `brief_termination` and `planet_fact`, and repetition/termination failures on `observatory_continuation`.

The baseline and candidate both had a zero full deterministic pass rate; therefore the conclusion is not that the random-init model was good. The directional result is narrower: under the pack's bounded deterministic/behavioral scoring, the 100-step checkpoint was measurably worse than the exact random-init state it started from.

## Judge exit

The governed Judge record is:

- verdict: `blocked`
- next action: `reject_checkpoint`
- confidence: `0.7071067811865476`
- promotion state: `rejected`
- certification state: `certification_blocked_by_regression`
- human review pending: `false`
- recheck required: `false`
- blocking comparison issues: none

This Judge action was **not applied**. `action_applied=false` is independently preserved in the launcher and terminal records. The checkpoint remains on the Network Volume as immutable evidence; it was not promoted, deleted, or used as a new training start by this integration.

## Cost and teardown

The successful evaluation Pod reported `0.5` credits/currency units per hour. Observed launcher lifetime was `125.158697489` seconds, giving an estimated cost of `0.017383152429027778`. This is an estimate, not a billing ledger.

The Pod was deleted successfully with HTTP `204` after independent verification. `funds_unavailable=false`.

## Frozen raw evaluation evidence

The raw evaluation proof is frozen under:

`docs/evidence/first-semantic-evaluation-001-33869352751/`

Exact source artifact identities:

- launcher JSON: `sha256:e8389db0f8d3d9732b784097c76474637d6cb704b8244728d7874fa6480f2624`
- verification JSON: `sha256:4a3e266413a717c31e83f7f1d894e7ccd6e0bef1d59e9b1a427b67ecf178e8c4`
- original GitHub Actions artifact ZIP: `sha256:f19f8b15db7c389beb00511522b7e799efcbe91d20df863aea1c5ff58afc7a01`
- original artifact ID: `9935299093`

The JSON files were re-downloaded from the original Actions artifact and hash-checked before being committed byte-for-byte. Generated model/checkpoint/sample objects remain on the Network Volume rather than in Git.

## Scientific boundary after this run

The evidence chain is now proven through Judge exit:

verified training proof → exact random-init baseline → frozen generation plan → baseline/candidate generation → independently hashed sample evidence → semantic/deterministic comparison → governed Judge exit → independently verified persistence → Pod teardown.

The scientific result is a rejection, not an infrastructure failure. The next closed-loop boundary is to feed this regression evidence back into evidence-based diagnosis and the planner, reject this checkpoint as a candidate for advancement, and propose the next controlled experiment without changing multiple primary variables at once.
