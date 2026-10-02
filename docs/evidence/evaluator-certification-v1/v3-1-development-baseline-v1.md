# Frozen Evaluator V3.1 development baseline

Status: **complete**

This record freezes the pre-training baseline requested for checkpoint selection on the new semantic-invariance development diagnostic. It is an inference-only development result, not a certification result.

## Scientific identity

- run ID: `evaluator-v3-1-dev-baseline-36998587411`
- GitHub Actions run: `36998587411`
- repository SHA used by the paid run: `673d5c84da0b853af412eee80d2cf89bda75b977`
- candidate ID: `evaluator-v3.1-boundary-v1`
- weights lineage: `v3.1`
- base model: `ibm-granite/granite-3.3-8b-instruct`
- base revision: `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`
- adapter SHA-256: `913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed`
- development curriculum SHA-256: `0d01fe1853742d1259a333fa9a92dec46e8789b69f2019ed34e17d815fc01a3c`
- exact 256-case development diagnostic SHA-256: `ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912`
- ordered case-ID SHA-256: `b52cbc157855da02c43bd136c972f87ee8a8a7e1ba4725c076d153360cfd5abd`
- result object: `s3://cviwpryzao/hephaestus/scientific/v4/evaluator_development_baseline/evaluator-v3-1-dev-baseline-36998587411/result.json`

The development diagnostic contained exactly 256 cases with exactly 32 cases for each scientific state:

- `certification_ready`: 32
- `incomplete_evidence`: 32
- `regressed`: 32
- `scientific_rejection`: 32
- `inconclusive`: 32
- `equivalent`: 32
- `recheck_required`: 32
- `improved`: 32

## Frozen baseline scorecard

| Metric | Frozen V3.1 baseline |
|---|---:|
| Decision exact | 39.0625% |
| Semantic exact | 39.0625% |
| Exact contract | 39.0625% |
| Semantic quality | 58.6133 / 100 |
| Schema compliance | 96.8750% |
| Evidence grounding | 99.2188% |
| Hallucination rate | 2.7344% |
| Primary-variable exact | 39.0625% |
| Primary-variable projection | 100.0000% |
| Boundary projection | 100.0000% |
| System action exact | 57.4219% |
| System semantic escalation | 0.0000% |
| Upstream-copy violation | 27.7344% |
| Initial complete-contract JSON | 99.2188% |
| Final complete-contract JSON | 100.0000% |
| Format retry attempt rate | 0.7813% |
| Format retry success rate | 100.0000% |

Wrong scientific decision count: **156 / 256**.

## Per-state decision accuracy

| Scientific state | Accuracy | Correct / 32 |
|---|---:|---:|
| `certification_ready` | 68.7500% | 22 |
| `incomplete_evidence` | 56.2500% | 18 |
| `regressed` | 21.8750% | 7 |
| `scientific_rejection` | 40.6250% | 13 |
| `inconclusive` | 59.3750% | 19 |
| `equivalent` | 0.0000% | 0 |
| `recheck_required` | 62.5000% | 20 |
| `improved` | 3.1250% | 1 |

Largest observed decision confusions include:

- `improved -> recheck_required`: 20
- `equivalent -> recheck_required`: 19
- `regressed -> recheck_required`: 13
- `equivalent -> inconclusive`: 12
- `certification_ready -> improved`: 10
- `scientific_rejection -> inconclusive`: 10

## Checkpoint-selection gate

Overall result: **FAIL**.

Checks that passed:

- evidence grounding >= 99%
- primary-variable projection = 100%
- boundary projection = 100%
- system semantic escalation = 0%

Checks that failed:

- overall decision exact >= 95%
- overall semantic exact >= 95%
- schema compliance = 100%
- hallucination <= 1%
- system action exact = 100%
- upstream-copy violations <= 1%
- every weak-state accuracy >= 90%
- every previously robust-state accuracy >= 95%

This gate failure does **not** invalidate the baseline. The purpose of this run was to establish the frozen V3.1 score on the exact development distribution before any weight update. Any repair checkpoint selected on this development diagnostic must therefore be compared against this same case set, ordering/hash identity, evaluation implementation, and frozen baseline result.

## Execution invariants

Verified by the terminal result:

- training performed: `false`
- optimizer constructed: `false`
- backward called: `false`
- adapter mutated: `false`
- adapter persisted: `false`
- certification claim performed: `false`
- production promotion performed: `false`
- automatic role dispatch performed: `false`
- burned sealed partition used: `false`
- trainable parameter count after explicit freeze: `0`

The model artifact reports 17,039,360 adapter parameters before the explicit inference freeze; after the runner freezes the loaded model, the observed trainable parameter count is zero.

## Runtime and cleanup

- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- GPU memory: 101,975,851,008 bytes
- RunPod pod: `y0ne4mqfst27cw`
- hourly rate: `$2.09`
- estimated paid compute: `$0.639235`
- preflight: passed (`8/8` input cases, 100% final complete JSON, 100% schema compliance)
- pod deletion attempts: 1
- pod deleted: `true`
- pod verified absent after teardown: `true`

## Launch-note

The first workflow attempt (`36998483768`) failed locally in the GitHub runner with a launcher recursion error before any RunPod pod was created. No model inference, diagnostic exposure, or GPU spend occurred in that attempt. The launcher binding was corrected and the single scientific baseline execution reported above then completed successfully.
