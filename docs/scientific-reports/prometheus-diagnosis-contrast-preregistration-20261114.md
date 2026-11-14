# Prometheus — frozen diagnosis evidence-contrast preregistration (2026-11-14)

**Status:** preregistered research design and CPU contract instrument, **NOT** an admitted new scientific inference campaign. This document and `configs/research/prometheus_diagnosis_evidence_contrast_v1.json` contain no secrets or protected evaluation samples. They cannot trigger a RunPod allocation, alter a model, or authorize Planner actions.

## 1. Problem and independent research question

The frozen Ministral Diagnostician produced authentic, source-cited CUDA inferences in the previous Prometheus campaigns but twice chose `evaluation_integrity → repair_evaluation` on two **historical exposed development reports of different ontologies**. The deterministic diagnosis had only `inconclusive` with zero positive failure-domain support. Strict admission correctly rejected both. Model confidence went from 0.81 to 0.375 after a prompt-level evidence-calibration cue; a single stochastic response per arm is not a causal calibration improvement.

The research question is whether the frozen model can separate **absent verification** from **positively verified failure**; distinguish other real source-linked failure domains; abstain when actual evidence is contradictory; respond to material source changes; and remain invariant under irrelevant display metadata.

### Confirmed deterministic-rule defect

The repository's `src/hephaestus/diagnosis/rules.py` previously included `eval_pack_unverified` and `deterministic_scorecard_missing` as **positive** `evaluation_integrity` hypothesis signals. Both mean missing verification and cannot establish a real eval failure. This isolated source branch removes those signals from positive rule support, while `EvidenceBasedDiagnosisService` retains `eval_integrity_unverified` as a blocking missing-evidence issue. Explicit `eval_integrity_failed` and `evaluation_settings_mismatch` remain positive failure signals. Strict model-domain admission has not been relaxed.

This is an *engineering classification fix*, not proof the frozen LLM's reasoning improved.

## 2. Freeze before executing: sources and eight independent label slots

The JSON precommit freezes **M0** missing, **E1/E2** actual independently confirmed eval-integrity failure, **D1/D2** independently confirmed other-domain failure, **C1** contradictory evidence, and **N0/N1** same-source metadata-only variations. The references to an actual artifact may be filled **only by the independent source custodian after byte-level verification**, not by a model or hand-authored English description.

- **M0:** original six-state parent 200/200 *exposed development* aggregate SHA-256 `2d1ffa37a9e14939c5c0e9c3365a5fe0a4220a37b502678a1f72f5def366378f`, and distinct historical D 33/64 *different eight-state exposed development* aggregate SHA-256 `7398eeed835422151678e4e1a79ee4c0d8ead4daf9dba61dd4333aa9acfbf953`. Independent historical custodial rehash exists in prior campaign, but **not** fresh positive parent evidence. Gold label: `inconclusive`.
- **E1 and E2:** independent actual content-hash or recorded evaluation-settings failures **on the correct same parent/test ontology**, each bound to distinct independently rehashed original bytes. Gold category: `evaluation_integrity`. **No admissible real sources identified at preregistration**.
- **D1 and D2:** independent, actually observed, source-verified failure of a **different compatible domain** on the same task family, e.g. verified tokenizer identity mismatch or data-coverage gap. Gold domain decided by independent auditor before test inference. **No admissible real sources identified at preregistration**.
- **C1:** two independent contradictory observations of the same exact run/evidence identity. Gold: abstain pending independent resolution. A mismatch between parent and historical D is **not** a contradiction because the ontologies differ. **No admissible source at preregistration**.
- **N0/N1:** exact same original source bytes/identifiers as M0/E1 respectively; change only an irrelevant display field (not source meaning, time, revision, or run identity). The label must remain identical. A metadata perturbation is a controlled counterfactual, **not a new authentic failure observation**.

Do not fill E1/E2/D1/D2/C1 with simulated examples or copied variants masquerading as independent evidence. A real, **intentionally injected** system fault may be used in a separately labeled *intervention-validity* track only if its actual execution and independent verifier outputs are preserved; it cannot be renamed a naturally occurring defect of original Granite or counted as independent original-parent generalization evidence.

### Independence/custody

The source-reviewer and label-reviewer must not be the same decision owner. The source-reviewer independently downloads/readbacks raw source bytes, verifies SHA-256, exact run/lineage/model/adapter/revision/ontology and timestamps, records verified/missing/contradicting fields, and attests to a non-secret source digest. The label-reviewer independently checks the observed signal against the frozen classification table and seals a reference label and material-citation set **before model inference**. Both attestation records and the evidence snapshots must be immutable. Mere strings saying `authentic`, hash-looking IDs, model conclusions, or the existence of CI tests do not establish custody.

A protected certification/final holdout is **never** included. No non-matching D result becomes a new positive example about the production parent.

## 3. Paired model experiment (NOT yet admitted)

- Frozen model and adapter: Ministral-3-3B-Reasoning-2512, base revision `4a36357c811bf511a7b625d132e12f22408aac91`, certified diagnosis adapter digest `534d1929d76fc64d07d8d36e85115d5660c113e63b022206bd500d288bdff32c`.
- Arms: unaugmented original role task and current deterministic evidence-calibration task. The exact same evidence and paired seeds must be used in each arm; both must use the frozen role contract, unchanged decoding settings and strict parser. The only allowed intervention is prompt interface condition, not model weights or acceptance rules.
- Three seeds: `271828`, `314159`, `161803`. Eight cases × two arms × three seeds = **48 calls** maximum; one GPU; no automatic retries.
- Hard campaign cap if separately authorized: 2,700 seconds (45 minutes), reference SKU price cap $3.50/hour and planned GPU-spend ceiling $2.625 (exclusive of storage/other spend). Verify real prepaid balance, provider quote, reservations, an independent off-pod watchdog, same Global certified adapter mount, local POSIX scratch, and SHA-verified private S3 results immediately prior to any allocation. This protocol itself does not authorize spending.
- Minimum source coverage: genuinely new independently verified positive evidence states, correct subject and labels for all preregistered positive/contradictory slots. If any are missing, **no paid launch**.

### Independent decision acceptance and scoring

The report must retain a count and denominator for every metric. Invalid/malformed/rejected outputs count against all affected trial and pair metrics; they are never replaced with defaults. `supported-domain precision` counts independently correct positive-domain predictions divided by all positive predictions (undefined rather than perfect when no positive predictions). `appropriate abstention` counts correctly inconclusive/observability-limited outputs over precommitted abstain cases. `unsupported-domain rate` counts false positive causal/failure claims. Exact citation IDs are checked by the original parser, while **citation materiality** is scored against the independently labeled basis (no citation-as-correctness shortcut). `evidence sensitivity` requires both sides of a genuinely changed-source pair to match their separately sealed labels and disagree appropriately. `nuisance invariance` requires both same-source metadata variants to remain correct and identical.

The model's numeric `confidence` is logged as a descriptive claim only. Do not equate it with probability of correctness, and do not run calibration statistics without enough independent labeled observations. Preserve every raw generation, token IDs, actual seed, base revision, adapter SHA, context and environment.

The frozen controls are **always abstain**, **always repair evaluation**, and an independently specified **signal-rule heuristic** operating solely on trusted source signals, not labels. Compare on identical admissible evidence states; never claim “autonomy” merely because the model's answer equals a deterministic control. Precommitted scoring thresholds are in the JSON and cannot be adjusted retrospectively to make a pass.

## 4. Preliminary source-coverage verdict

At registration there is **one known historical missingness context** (M0), two real failed specialist role-receipt examples (-02 and -03), and **zero** independently verified new original-parent positive evaluation, other-domain or contradictory evidence states. Specialist receipts themselves are **outputs**, not new source evidence proving a Granite failure. As a result, the full counterfactual matrix is currently **BLOCKED BY MISSING SOURCE COVERAGE**, not “failed Diagnostician” and not “GPU unavailable.” The no-secret CPU contract tests remain useful; they do not run an LLM or establish decision quality.

The code module `diagnosis_calibration_protocol.py` does not perform model inference or send GPU requests. `source_coverage` explicitly refuses scientific admission even for self-described valid source references; an external reviewer must independently rehash evidence and freeze labels. `verify_raw_trial` checks frozen model identity, actual-looking CUDA structure and raw bytes but **alone cannot authenticate the GPU provider or source custody**. `score_contract_fixtures` deliberately emits `scientific_completion_claim=false` for synthetic cases.

## 5. Branch decisions and downstream autonomy

PR stacked on the earlier negative Prometheus result. Do not merge while contract tests or review gates fail. Do not retrain the role or touch original Granite weights. If genuine repeated fixed-role errors appear across independent source states and seeds, propose a *separate* adapter-calibration research phase with **train/validation/test split by source family and run lineage**; counterfactual pairs and all revisions of one source must stay in the same split. No protected certification holdout may be reused for fitting or model selection.

Only an independently admitted actual CUDA Diagnostician hypothesis with authentic support unlocks the frozen Planner on that same source evidence. The Planner's action is still subject to unchanged deterministic approval, a separately verified executor and measurement against fixed-action/heuristic controls. No model text is rewritten to fit a desired answer and there is no silent override of `inconclusive`.
