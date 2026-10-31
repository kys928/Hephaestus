# Scientific claims, verification and certification standard

The question is not merely whether a training job ran; it is **which conclusion the evidence permits**.

## Classify every result

| Level | Examples | Permitted statement |
| --- | --- | --- |
| Intent / proposal | Config, PR, planning text | Proposed; not executed |
| CPU contract/smoke | Unit tests, simulated backend, schema parser | Specified behavior tested; not evidence of real CUDA |
| Real execution | Device receipts, real optimizer steps, model generations | This scoped run executed authentic operations |
| Internal independent verification | A separate reader checks original persisted bytes, hashes and run/role identity | This internal execution and archive were independently verified |
| Behavioral comparison | Frozen baseline and candidate scored on a predeclared independent pack | Measured task difference under these conditions |
| External scientific certification | Separate required custody, contamination review, reference comparisons and signatures | Certified only for the independently established scope |
| Promotion / production activation | All deterministic and human-approval gates pass | Explicitly allowed to deploy/use under that approval |

Do not treat any row as automatically satisfying the next.

## Evidence required for improvement claims

Before an improvement experiment: pre-register a specific target, independent baseline, training/eval split and provenance, unchanged decoding controls, seeds, hard-regression gates, scoring method and uncertainty/variance plan. A comparison must include the same frozen test conditions for baseline and candidate. Track failures and non-results; do not silently revise the evaluation pack after inspecting outcomes. Prefer repeated seeds and matched tests to vanity single-score ranking. The [evaluation policy](../evaluation_policy.md), [lineage policy](../lineage_policy.md) and [promotion gates](../promotion_gate_policy.md) are binding.

For model-role scores, name the evaluation distribution, adapter revision, exact schema and denominator. A measured 100% on one finite fixture is **not** a universal 100% guarantee; a failed or missing real inference cannot be filled with synthetic success receipts.

## Custody and honest limitations

Scientific artifact pointers must include original source revision and immutable content hashes where available. A cold reader must confirm membership, completeness and matching source/run identity. An archive-presence check is weaker than re-executed training; replay trust levels differ ([policy](../replay_verification_policy.md)). Private holdouts and protected raw corpus remain outside Git.

The [original R10 index](../evidence/original-cuda45-r10/README.md) supports a bounded internal 45/5/96 execution claim. It does not prove target-model improvement or external historical certification. Those unfinished checks have explicit owners: [#152](https://github.com/kys928/Hephaestus/issues/152) for improvement and [#102](https://github.com/kys928/Hephaestus/issues/102) for independent historical custody.

**Default wording:** `verified execution`, `measured improvement under a frozen protocol`, `inconclusive`, `rejected`, or `certified for stated scope`. Never replace a missing standard with confident rhetoric.
