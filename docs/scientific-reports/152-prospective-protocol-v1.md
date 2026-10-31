# Issue #152 — prospective causal model-behavior protocol v1

**Status (2026-10-31): pre-registration candidate — NO new confirmatory answers inspected, NO model improvement measured, NO external custody signed.** Freeze this decision *before* a separate custodian materializes the future holdout, baseline/candidate generations or optimizer runs. Never repurpose the 24-row previously inspected R10/CPU development or original evaluation set as "unseen".

## Frozen target

- Target: `HuggingFaceTB/SmolLM2-135M-Instruct`, pinned model revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` with content hash independently reverified from original artifact before execution.
- Four **narrow** original verifiable task grammars: arithmetic, integer-list count/sum, symbolic reversal, 5-node shortest-path; independently parsed answer keys via `generated_validation.py`. Claim scope: this grammar only; **not** generalized intelligence or from-scratch pretraining.
- Training: same admitted 96 training samples, original chat template and EOS-target supervision, LoRA rank 8, alpha 16, AdamW learning rate 5e-4, same epoch/step plan; **at least three genuine initialization/training seeds**, distinct verified optimizer state/checkpoints. No change to data, optimizer, decoding or model revision after looking at candidate outcome. Development-only seed choice before held-out unlock.
- Candidate selection: choose the single adapter with highest **development exact typed correctness**, then valid JSON, then lowest precommitted seed as tie-break. Record all attempted seeds, failures and selection decisions. **One** untouched final confirmatory test, not repeated winner selection on the holdout.

## Independent pre-test custody

A genuinely separate custodian must select and seal the future four-family holdout, not simply choose an exposed fixed dataset seed in public code. Intended sample budget: **512 independent canonical task instances per family, 2,048 total**, distinct from train, development, historical original evaluation, code/agent test examples and any intervention examples. Build new examples from a fresh, custody-held generator seed and independently screened exact canonical fingerprints; report inability to prove semantic non-overlap as an explicit scope restriction. The generator is public; merely withholding the seed is *not* a cryptographic defense against a model trained on the public grammar.

The custodian must commit SHA-256 of canonical **sorted** `{sample_id,family,prompt,target}` objects and an independently inspected exposure manifest **before training/outcome observation**, with Ed25519 signed statement, trusted public key identity established separately, date and source revision. Protected prompts, references and raw generations remain outside public Git. The scoring module accepts the expected external hash, checks exact item identities, deduplication/exposures and independent task validation. Hashes alone do not prove that a custodian is honest or that the pretraining corpus never saw the grammar.

## Primary endpoint and gates (predeclared)

**Primary:** paired strict JSON **exact typed answer correctness** over 2,048 cases. Secondary: well-formed JSON object rate. Baseline and selected candidate use identical system/user chat template, greedy decoding, max_new_tokens=64, tokenization/stop/EOS, same evaluation items, same frozen scorer. Captured raw outputs, full token receipt, input/target hashes and EOS/truncation flags must be independently checked against the actual frozen model and checkpoint hashes; the pure scorer alone cannot authenticate them.

No code fences or pseudo-JSON postprocessing; duplicate keys, NaN/Infinity, type coercion, truncation and wrong keys are failures. A blind independent evaluator recomputes all scores from original raw text.

For a positive internal held-out claim **ALL** must hold:
- Paired candidate exact-correctness **absolute gain >= 5 percentage points**.
- Exact two-sided McNemar p-value **< 0.05** on discordant paired examples.
- Deterministic 2,000-draw paired-bootstrap 95% CI lower bound **> 0** (seed 152).
- Candidate valid-JSON rate no worse overall; every task family's exact correctness no worse.
- Correct→incorrect regressions at most 1% of the 2,048 matched items.
- Every candidate generation has recorded EOS and no truncation.
- External custody/provenance, original checkpoint membership and raw-generation evidence must separately pass before an *externally certified* claim. Missing a receipt means **inconclusive**, not success.

The paired bootstrap and McNemar assumptions include independent observations. Shared templates and generators induce correlated tasks; results must be reported as *within-family* evidence with caveats, not an unconditional generalization. This sample budget is an initial precision/power design based on a 5pp meaningful threshold and 2048 matched cases; **a formal blinded discordance-rate pilot/power calculation must be completed before irrevocably freezing and generating the final holdout**, and sample size must only be changed at that stage, never after evaluating the candidate. This protocol records the conservative current design, not a claim that statistical power was independently validated.

## Audit / outcome

Publish a public-safe summary with baseline/candidate counts, wins/losses/ties, per-family paired counts, matched confidence interval, exact p-value, all deterministic gates, multiple-seed variance and cost. Preserve protected raw rows and immutable model/tokenizer/adapter hashes in custody, not Git. Independent custodian must verify final raw output receipts and sign the custody statement; model promotion and autonomous dispatch are separately gated, not granted by this score.

Accept only one of `internally supported under frozen narrow task scope`, `not materially improved`, `regressed`, or `inconclusive`. `src/hephaestus/evaluation/paired_holdout.py` is intentionally a **pure scorer**, not a training orchestration, raw-output authenticator or external certificate issuer.

### Prerequisite gates before actual runs
- [ ] Independent original R10 weights/tokenizer/chat template and adapter cold recovery; verify SHA identity and licensed data.
- [ ] Formal pre-test power/precision and near-duplicate semantic screening, with independently sealed 2,048-item heldout ledger (or amended design **before** any outcomes).
- [ ] Fresh real training seeds, development-only selection and authoritative optimizer/manifest receipts.
- [ ] One final held-out evaluation performed by a separate custodian; independent check of raw generations, EOS and source identity.
- [ ] Publish all attempted runs, analysis, negative results, raw custody hashes and audit disposition.
