# Evaluator: effect-polarity and decision-precedence investigation

**Verdict: a reproducible observational shortcut candidate, not a proved causal mechanism.** No new model inference was performed; the existing development and training data remain unchanged.

## Sources and reproducibility
- Previous D factor-only inference (saved immutable 64 reports): [GPU run #38962439250](https://github.com/kys928/Hephaestus/actions/runs/38962439250).
- Original factor-only strict response requirement: 0/64. Exploratory CPU-only normalization of the exact Markdown JSON singleton-array wrapper recovers all 64 six-field reports without altering values or acceptance gates.
- Previous independent normalized-factor analysis: [#38964042656](https://github.com/kys928/Hephaestus/actions/runs/38964042656).
- Current source-level and saved-output CPU-only sign audit: see [Evaluator effect polarity and precedence CPU audit](https://github.com/kys928/Hephaestus/actions/workflows/evaluator-effect-polarity-cpu.yml). Its JSON artifact includes S3 content hashes and aggregate evidence.
- Frozen development: unchanged 256 cases; D model step-96 saved scientific decisions: unchanged 64-case balanced subset.
- There was **no** RunPod allocation, training, checkpoint selection, new inference, certification, production merge, or approval reopening.

## Finding 1: positive-to-negative errors concentrate in precedence-masked cases

| D factor-only diagnostic condition | Correct reported effect |
|---|---:|
| Effect cannot determine final state because a higher-priority condition blocks it | **5/32 (15.625%)** |
| Effect is relevant to the final state | **23/32 (71.875%)** |

**24 of 26 positive-to-negative flips** occurred within the four higher-priority blocked states: scientific rejection, incomplete evidence, inconclusive, recheck required. Those four states have genuine positive effect in the frozen development reference but a valid negative/blocked final *decision*. This is exactly the kind of supervision confound in which a learner might conflate "unfavorable verdict" with "negative quality delta".

Per-state effect accuracy among eight D screen cases each:

| Expected scientific state | True frozen effect | Correctly reported effect |
|---|---|---:|
| scientific_rejection | positive | 0/8 |
| incomplete_evidence | positive | 2/8 |
| inconclusive | positive | 0/8 |
| recheck_required | positive | 3/8 |
| improved | positive | 4/8 |
| certification_ready | positive | 8/8 |
| equivalent | neutral | 6/8 |
| regressed | negative | 5/8 |

The failed effect reports are *not* caused by negative throughput text **inside the development prompts**: all 256 frozen development cases use explicit effect wording, and **none includes the secondary throughput distractor**.

The initial observational association is strong, but prompt differences, per-state wording and other covarying factors prevent causal identification.

## Finding 2: structural supervision blind spot and train/evaluation shift

The frozen development generator uses one canonical six-factor configuration for each state. For the four blocked states, its effect is always positive. Thus, all **128/128 blocked frozen development cases** use positive effect, while the actual final state is always blocked.

The D training curriculum is more factorial, and each of eight states contributes 96 examples. Among its **384 blocked-state training examples**, the effects are approximately balanced **148 positive / 117 neutral / 119 negative**. In other words, **positive is only 38.5%** of the blocked-state training factors, but **100%** of frozen development blocked-state factors. A model that tries to guess effect from state and often neglects lower-priority evidence can fail badly here.

This is one plausible form of train-development conditional shift. The scientific-state supervision does not require the effect sign to be retained when a higher-priority issue already decides the target label. The blocked-state target rationales emphasize the blocker, and in most cases do not explicitly restate the quality-effect sign. The evidence is therefore compatible with weak factor-level supervision, but does not prove optimizer-level causality.

The D curriculum also contains **256/768 cases** with a *secondary throughput delta* explicitly opposed to primary quality delta. For **108 training examples with genuinely positive primary effect**, that secondary throughput delta is negative. The text correctly marks throughput as report-only and not authoritative for quality. This is a realistic sign-attention challenge and a possible further confound; the current no-GPU audit cannot prove it caused the reported sign errors.

## Finding 3: mapping/precedence inconsistent even when crucial factors are reported correctly

From the prior independent normalized-factor probe:

- \`equivalent\`: **6/8** report neutral effect correctly, while the original D scientific decisions were correct in **0/8**.
- \`recheck_required\`: **7/8** report unstable variance correctly, while original D scientific decisions were correct in **0/8**.
- All six factors correct in **23/64**; saved D scientific decisions correct in **33/64**.
- Separately elicited factor/decision pairs include **11 cases** where all factors were correctly reported yet the separate final decision disagreed with the frozen oracle.

These are **cross-prompt** comparisons, so no hidden chain-of-reasoning claim follows. But the mismatch makes decision precedence an independent target for controlled testing; improving factor extraction alone may not be enough.

## CPU-only counterfactual decision-mapping experiment

Implemented a separately authored panel of **60 minimal one-scientific-variable pairs** (15 defined transitions × two neutral presentation styles × with/without a secondary throughput distractor). It contains:
- Exactly **one changed scientific variable per pair**: \`gate_pass\`, \`complete\`, \`provenance_valid\`, \`effect\`, \`variance_stable\`, or \`recheck_satisfied\`.
- **48 pairs** expected to change scientific state under the frozen oracle.
- **12 precedence/negative-control pairs** expected *not* to change scientific state because a lower-priority factor was modified.
- **30 pairs** featuring a separate, report-only secondary throughput delta of opposite sign to the primary quality delta.
- **120 prospective unlabeled inputs** for a future separately authorized model decision probe; none has yet been sent to the D model.
- Full independent 96-state-factor truth-table oracle checks, including the critical:
  - Neutral stable → neutral unstable: \`equivalent\` → \`recheck_required\`.
  - Neutral stable → positive stable: \`equivalent\` → \`improved\`.
  - Neutral unstable → neutral stable: \`recheck_required\` → \`equivalent\`.
  - Positive unstable → positive stable: \`recheck_required\` → \`improved\`.
  - When variance is unstable, changing positive to negative effect **must not change** \`recheck_required\`.
  - If provenance is invalid, changing effect or variance **must not change** \`inconclusive\`.

All tests run on CPU, with no training data reuse from the original frozen development cases, and no certification claims.

**Critical limitation:** These CPU tests establish that the oracle and experimental *labels/design* are consistent. They **do not show that the Evaluator model passes these 60 contrasts**. Measuring model counterfactual consistency requires separately authorized inference on the fixed, new panel; the previous one-use inference budget is spent and disabled.

## Recommended next scientific action

Do not fund E, retrain D, or change model architecture based on these aggregates. Instead:

1. Explicitly separate "primary quality direction" from "final decision suitability" in the next factor-reporting and state-decision prompt.
2. Use the independent 60-pair panel to test *model* consistency, if later authorized, with paired exact decoding and no training. Compare quality direction with/without negative secondary throughput and under high-priority blockers.
3. Predeclare paired performance and failure taxonomy before any fresh inference. Analyze effect sign in blocked versus decisive conditions and exact state changes across each pair, rather than only overall classification accuracy.
4. If model inference confirms a blocked-state sign shortcut, consider factor-supervised multitask training with an explicit six-factor loss/objective, blocked-state effect sign balancing, and a strict deterministic state-oracle comparison. Test each change separately; no paid training until independently justified.
5. Preserve all previous sealed-data, certification, spending and production boundaries.

**Scientific interpretation:** The evidence implicates **loss of factor information under precedence masking, compounded by a training/development factor shift**, with secondary throughput distractors a separate plausible contributor. The model's true internal causal mechanism remains unknown until a controlled model experiment is run.
