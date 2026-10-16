# D step-96 factor-only inference — final independent scientific assessment

**Status: inference complete; strict-format contract failed; exploratory normalized factor evidence available; no certification.**

Sources:
- [One-time approved GPU workflow #38962439250](https://github.com/kys928/Hephaestus/actions/runs/38962439250)
- [Independent raw-output structure audit #38963653286](https://github.com/kys928/Hephaestus/actions/runs/38963653286)
- [Independent strictly constrained normalized-factor audit #38964042656](https://github.com/kys928/Hephaestus/actions/runs/38964042656)

## Run and spending

- Exact saved D step-96 adapter SHA-256: `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`.
- Immutable executed source commit: `aabfc6ecada10efd2841a6c9ccdf5f566b024b33`.
- Run ID: `factor-only-D-38962439250`; precisely 64 generations persisted across eight immutable S3 sample shards.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, rate $2.49/hour.
- GPU time including verified teardown: 224.326 seconds; estimated GPU spend **$0.155159**.
- Pod `n9nw2sr5u83qoi` deleted successfully; absence verified on the first attempt.
- Initial spending ceiling remained **$0.75**. Exceptional **$1.50 contingency never used**.
- No training, certification, checkpoint selection, production promotion, new sealed pack usage or model modification.
- The once-only approval record is now marked **completed_spent_and_disabled**. No new paid execution is authorized.

## Two distinct scoring interpretations — do not combine

**Original strict six-field raw JSON requirement:** **0/64 passed.** The model failed the exact output serialization contract, despite returning the requested information in a different representation.

**Independent saved-output structural audit:** 64/64 generations were Markdown JSON code fences containing **a JSON array of exactly one object**, whose keys were exactly the six requested factors. None was truncated or hit the generation deadline. The original 0/64 result remains unchanged.

**Separately labeled, exploratory *post-hoc* wrapper normalization:** A CPU-only parser accepted exactly that complete fence-plus-singleton-array structure, changed **no scientific field values**, used **no target labels** for repair, and verified JSON types. All 64 could be decoded and scored against frozen reference evidence. This is **not retroactive strict-format compliance**, new model inference, certification or proof of hidden internal reasoning.

## Reported-factor accuracy after exact structural normalization

| Factor | Correct reports / 64 | Accuracy |
|---|---:|---:|
| `provenance_valid` | 64 | 100% |
| `recheck_satisfied` | 64 | 100% |
| `complete` | 63 | 98.44% |
| `gate_pass` | 61 | 95.31% |
| `variance_stable` | 45 | 70.31% |
| `effect` | 28 | 43.75% |
| **All six simultaneously** | **23** | **35.94%** |

The provenance-only attribution hypothesis from the previous *decision-confusion* audit is **not supported as an observed provenance-reporting problem** here: provenance was correct 64/64, while the effect category was frequently wrong.

Effect confusion was sharply asymmetric: of **48 genuinely positive-effect cases**, only **17** were reported positive, **26** negative and **5** neutral. For eight neutral cases, six were reported neutral; for eight negative cases, five were reported negative.

Variance confusion: among 56 cases with genuinely stable variance, **18** were reported unstable. Of eight cases with genuinely unstable variance, seven were reported unstable.

## Critical state-level contrasts

| Frozen target state | Relevant factor recovered | D saved final scientific decision correct |
|---|---|---|
| `equivalent` | Neutral effect 6/8; variance stable 8/8; provenance valid 8/8 | **0/8** |
| `recheck_required` | Variance unstable 7/8; provenance valid 8/8 | **0/8** |
| `inconclusive` | Provenance invalid 8/8; effect correct 0/8 | **8/8** |
| `certification_ready` | All six factors correct 8/8 | **7/8** |

These were *different prompts on the same cases*: the factor-only diagnostic and the original D decision generations. They demonstrate observable cross-prompt inconsistencies but **cannot prove what the original decision-model execution internally recognized**.

## Paired observable mismatch categories

Across 64 exact same-case pairs of factor report and prior D decision:

- **12:** Factor report correct, saved decision consistent and correct.
- **11:** Factor report correct, saved decision inconsistent with the oracle applied to those reported factors — an *observable cross-prompt precedence/decision mismatch*.
- **15:** Factor report incorrect, saved decision consistent with the oracle applied to the incorrect reported factors.
- **26:** Compound or inconsistent factor/decision mismatch.

The original saved D decision correctness remains **33/64**; factor-only prompting did not regenerate or modify those decisions.

## Interpretation and next research steps

The strongest observed failure is **effect-category reporting** (especially the tendency to read positive effects as negative), followed by overstating variance instability on stable cases. There is additional evidence of *cross-prompt decision inconsistency* in cases where the separately reported factors are correct, especially `equivalent` and `recheck_required`.

This does **not** justify E training. A productive next phase is CPU-only:
1. Inspect effect-sign evidence wording, threshold language, and whether any irrelevant negative throughput/distractor has been promoted to primary quality evidence.
2. Audit the prompt formatting and schema; the current model reliably emitted a singleton JSON array in a code fence instead of the raw object. Preserve both strict and exploratory metrics.
3. Design targeted **new, independently authored** factor-vs-decision contrasts crossing effect, stability and provenance while holding other factors fixed. Score oracle labels and validate absence of shortcuts before future inference.
4. If further inference is proposed, get *new* authorization and compare separately elicited factor and decision outputs with matched decoding and strictly controlled prompts. Do not treat self-reports as hidden reasoning.

Every artifact is diagnostic-only on development cases; no sealed certification, training, production deployment or model promotion occurred. The user-authorized inference run has finished and its spending approval is consumed.
