# Evaluator six-factor CPU-only error anatomy

**Scientific status:** evidence-backed observational diagnosis; **extraction versus precedence remains unresolved**.

This report is based on [CPU/S3 read-only audit #38954572046](https://github.com/kys928/Hephaestus/actions/runs/38954572046). The workflow checked 31 persisted S3 objects, kept object-content hashes in the downloadable artifact, loaded no model, allocated no GPU, and wrote nothing to S3.

## Evidence provenance and denominators

| Evidence | Observed cases | Correct |
|---|---:|---:|
| Frozen parent | 256/256 | 100 |
| V1 step 16 | 256/256 | 103 |
| V1 step 96 | 256/256 | 117 |
| C step 96 | **128/256 ordered-prefix only** | 61 |
| D step 96 | **64/256 balanced-screen only** | 33 |

The C prefix contains four complete 32-case states and **no** observed equivalent, recheck-required, inconclusive or certification-ready cases. The D screen contains exactly eight cases from each state, and is a diagnostic subset, not a fresh sealed pack. C and D cannot be compared as if they observed the same cases. Independent same-case comparisons match persisted experiments: C 61/128 versus V1 63/128; D 33/64 versus V1 34/64.

All 256 original reference cases were separately decoded using the actual six template fact sentences **without looking at the expected state**. Their extracted factor values matched the frozen machine state and expected oracle labels. This validates the *synthetic reference template*, not a model's latent interpretation or real-world scientific ambiguity.

## Observed prediction bias

- **V1 step 96:** 97/256 predictions were \`inconclusive\`; only 23 were correct \`inconclusive\` predictions. \`equivalent\` correct 2/32; \`recheck_required\` correct 3/32.
- **V1 step 16:** 92/256 predictions were \`inconclusive\`; \`equivalent\` correct 0/32, \`recheck_required\` correct 15/32. The loss of recheck-required performance is visible despite a modest gain in total accuracy.
- **C:** 41/128 predictions were \`inconclusive\`, although none of its observed four target states was inconclusive.
- **D:** 32/64 predictions were \`inconclusive\`, versus only eight genuine \`inconclusive\` targets; those eight were all correct. \`equivalent\` and \`recheck_required\` each scored **0/8**.

## Six-factor counterfactual diagnosis

For every wrong saved decision, the audit searches **all 96 legal factor combinations** and identifies the fewest factor-value changes needed for the frozen oracle to output the *observed incorrect decision*. If several explanations require the same minimum number of changes, the report identifies both factors common to every minimal explanation and factors appearing in any minimal explanation.

| Checkpoint | Wrong cases | Wrong decisions whose minimal oracle explanations necessarily change provenance validity |
|---|---:|---:|
| V1 step 16 | 153 | 77 |
| V1 step 96 | 139 | 78 |
| C step 96 (observed) | 67 | 41 |
| D step 96 (observed) | 31 | 24 |

In D, 26 of 31 wrong decisions admit a **one-factor** counterfactual explanation. The remaining five require two changed scientific factors. In V1 step 96, 117 wrong decisions admit a one-factor explanation, and 22 require two.

These numbers make provenance-related distinctions the strongest targeted hypothesis, particularly for wrongly outputting \`inconclusive\` on cases with valid provenance. They are **not measurements of extracted provenance accuracy**: a model that correctly reads the input but misapplies precedence can produce the same final label as a model that misreads a factor.

The original 256 development cases are **canonical per state**, not a fully factorial held-out diagnostic. Factor values co-vary with target states, so simple factor-conditioned accuracy cannot identify independent causal effects. The audit is explanatory and prioritizing, not causal or certifying.

## What distinguishes evidence interpretation from precedence?

The current saved JSON predictions contain final \`decision\`, \`action\`, and \`rationale\` but no verified report of the independently extracted six factors. **There is no identifiable extraction-versus-precedence decomposition in those outputs.**

A future test would need independently elicited factor-only outputs (six typed fields, **no decision**), then a *separately elicited* state prediction on identical evidence and a CPU application of the frozen oracle to the reported factors. The new \`factor_only_diagnostic_prompt\` helper prepares an unlabeled prompt and the \`score_explicit_factor_probe\` helper can classify the combined records. Tests currently only exercise these helpers with synthetic responses; **no paid or free model inference has been performed**.

Potential outcomes include:

1. Reported factors match reference, but state disagrees with the oracle: observable precedence inconsistency.
2. Reported factors disagree with reference, and state agrees with the oracle applied to *reported* factors: observable factor-report inconsistency.
3. Both disagree: mixed or non-identifiable inconsistency.

Even those outcomes show **reported behavior**, not inaccessible internal reasoning; changes in output contract and prompting can change the result. Use a **fresh independent diagnostic-only corpus** for any subsequent model probe where possible. Never train on the frozen development pack, relabel its truth, reuse burned sealed cases, or call probe success certification.

## Recommendation before buying another run

**Do not fund E or repeat D yet.** The next *scientifically discriminating* experiment is to have the candidate report the six factors separately and compare against the reference fact oracle—especially validity of provenance, repeat variance and neutral effect classification—then test state assignment under a fixed, independent decision prompt.

CPU tests, artifact lineage and source-level evidence have been validated; model factor-report accuracy remains **unknown**. A paid inference budget, if required, would need its own approved scope and ceilings. No new GPU or model training is authorized by this diagnostic.
