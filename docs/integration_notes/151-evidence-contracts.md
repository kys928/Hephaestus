# Issue #151 — integration slice A: immutable evidence custody and campaign identity

## Source and trust boundaries

- Reviewed R10 executed source commit: `d6bcec7b61bcd3b285a4bcf4cdd50036440b8edf`.
- Integration base: `acad4dddb61602dcf473774229be0b1462d935d5` (the already published public-safe R10 report in main).
- Original genuine CUDA45 result: [scientific evidence report](../evidence/original-cuda45-r10/README.md); original r10 code and archived data stay unchanged.
- This patch only **copies exact R10-reviewed existing Git blobs** into the old mainline; it does not replace `main` with the R10 branch and does not write S3, allocate Pods, activate LoRA role routing, change authority or claim scientific certification.

## Integration contents

| File | Immutable reviewed R10 Git blob |
| --- | --- |
| `src/hephaestus/storage/verified.py` | `a0b80cf2d4b64f4be248e5fd3ccaae97f44e3e70` |
| `src/hephaestus/storage/external_evidence_pointer.py` | `4168df0276e80904dc5514a631c287e002e574c0` |
| `src/hephaestus/schemas/cpu_campaign_config.py` | `4adb8da114b2c9d44e3dd0c7bb4d8be5163229d9` |
| `configs/research/cpu_chat_json_repair_v1.json` | `b988e10ff86425a3687d5b7ba2d66ae603d38d45` |
| `tests/test_external_evidence_pointer.py` | `f7322714847c1e06004829b51d6694a5aca028e4` |
| `tests/test_cpu_campaign_config.py` | `33eb8bb044f0f75bd85dc747c99b5dc1ebb65d6f` |

The `cpu-campaign-config.v2` contract uses the included reviewed JSON recipe for deterministic identity commitment; source data is *not* imported or published. The external evidence pointer is a validation-only object; no upload client is activated. `read_verified_json` hashes the **same bytes it parses**, rejects non-SHA references, extra-long payloads and nonfinite JSON constants.

## Test and rollout plan

- Run exact-source blob assertions and pinned CPU tests in the independent no-cost branch workflow.
- Merge this small contract slice after successful CI.
- Continue in **separate reviewable PRs**: trusted scientific state stores and historical replay; real services/data; training and evaluation; original CUDA five-role adapters/runtime; RunPod resource execution; independently cold-verified archive publication.
- Keep all historical r1–r10 paid YAML runners **off** `main` and leave evidence/production activation boundaries separate.
- No immediate rename/removal of old modules, workflow history or Git branches. Archive historical context only after dependency inventory and immutable links are preserved.

**Non-goals:** not enabling real GPU execution, not replacing the eight-stage control spine, not restoring raw private data, not moving the frozen Evaluator or running target optimization, and not closing #102's external certification gates.
