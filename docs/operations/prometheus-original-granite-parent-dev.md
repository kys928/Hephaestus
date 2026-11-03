# Project Prometheus — original certified Granite six-state parent development forensic

This work targets the original frozen **production certified Granite 3.3 8B Evaluator**, not the historical eight-state semantic-repair D candidate.

Source identities:

- Frozen role-mastery original training repository: `c831bbfc0e2054d54fccd1ef374fd84722612c79`
- Frozen full training pack hash: `6f4bb3b8c8312a4189136ea961e8fa32629c02fa5459e18fa960670badb203b3`
- Original certified model revision: `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`
- Original selected Evaluator adapter SHA-256: `e60497854ea64655fc18be7962dec29f13091c75ea91e576726b0fa5c6eb2325`
- Original model-selection checkpoint: optimizer step **250**
- Exact target S3 path: `hephaestus/scientific/v1/role_mastery/role-mastery-v1-35596791201/roles/evaluator/dev/step-0250.jsonl`
- Original role-level result metadata: `.../roles/evaluator/result.json`

The certified parent original source generated **1,500 train, 200 development, and 300 certification** Evaluator cases. The 200 cases were already used for selecting the winning training checkpoint. They are **exposed model-selection development evidence**. The original 300 certification case **outputs are not requested or read**. In particular, the read-only forensic script hardcodes a two-object S3 allowlist and explicitly rejects `certification.jsonl`.

The original workflow reported high scientific role-certification quality. A near-perfect original development score would establish **only** performance on the generator's original role-mastery distribution. It would not negate the later semantic-invariance research weaknesses, and the later eight-state D scores cannot be directly attributed to the production six-state adapter.

## Technical procedure

1. A temporary, PR-only, read-only GitHub Actions workflow independently checks out the original frozen September source commit.
2. It regenerates the original *development-only* reference case IDs, skill labels and expected six-state decisions, and verifies the entire pack hash from the original frozen source.
3. It runs fake-S3 tests first, without credentials.
4. A single dedicated S3 step performs GET only for **the original role result JSON** and **the selected step-250 development JSONL**. It does not issue HEAD/list/write operations or retrieve certification examples, raw certification predictions, training weights or final holdouts.
5. The audit independently parses saved raw/generated text and normalized output, recomputes strict decisions, and matches case IDs, skills, checkpoints, parent adapter and summary aggregates. Only aggregate metrics and object-hash receipts leave the ephemeral runner.

**Important provenance distinction:** Unlike the historical D shards, the original parent development JSONL currently lacks a separately retrieved pre-existing independently published SHA-256 receipt. Hashing the bytes *after recovery* identifies the object observed in this execution but does not prove prior immutability. Matching the originally archived per-checkpoint scorecard, selected adapter hash, reference pack hash and per-case saved decision scores adds consistency evidence but is not the same as independent precommitted custody.

Any discrepancy between the newly strict parser and the original historical scorer must be documented; never alter the original results or award passing decisions to malformed outputs.

## Research boundaries

The script does not load model weights, infer new tokens, train, spend paid GPU credits, change activation, or promote a checkpoint. It does not certify model behavior or scientific autonomy.

After the forensic run, **delete the temporary credentials-bearing GitHub workflow from the proposed merge tree**, retaining evidence and read-only source/tests, and rerun the maintained CPU/integration CI. Do not modify the active workflow allowlist or allow repository secrets into permanent CI.

Next, pass source-authenticated six-state development aggregates as **observations**, not causal prescriptions, to the existing Diagnosis and Planner. Any actual training intervention must be chosen by Hephaestus after sufficient evidence, using bounded governed executors and appropriate compute authorization. Keep separate source/model/ontology provenance for the eight-state historical D research.
