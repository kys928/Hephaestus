# Provisional Coder qualification — parent Ministral, not a Coder model search

**Date:** 17 November 2026. **Scope:** accepted local pipeline gate and original parent-model wiring. **Status:** temporary model uncertified. No revision to the frozen five-role certification record.

## Decision and deferral

- Temporary Coder: `mistralai/Ministral-3-8B-Reasoning-2512`, immutable revision `81eaece1948f3875421d9a45bc55487d10e2d894`. This is exactly the original *Planner's parent family/revision*, **without** its certified LoRA.
- No model search, baseline shoot-out, model promotion or adapter training in this phase. The future complete model-selection/certification campaign is [Investigation: Coder #215](https://github.com/kys928/Hephaestus/issues/215); deliberately deferred until complete system-wide certification of the original stack.
- Temporary Coder config: `configs/models/provisional_ministral_coder_v1.json`. The frozen five-role registry and Planner inference contract remain unchanged.
- `ProvisionalMinistralCoder` uses the original pinned local Transformers/MistralCommonBackend CUDA loader, one visible GPU, no Ollama, no remote model fallback and no Planner LoRA. It is text-only and has **zero file, tool, test, approval or merge authority**. Actual GPU invocation requires externally approved compute and the original pinned model snapshot; no such invocation was performed in this acceptance.

## Research question

Can **Hephaestus itself**:

1. identify a *specific, independently reproduced* semantic defect rather than an import/collection failure;
2. choose an allowed minimal repair strategy from a controlled registry;
3. request a bounded code candidate from temporary parent Ministral;
4. independently determine whether the candidate is correct on unchanged, reviewed tests;
5. quarantine a wrong first candidate, optionally correct it inside **at most one local retry**, never propagating the failed candidate to another stage;
6. produce a content-addressed, replay-verifiable review handoff **only after** an independent sandbox has reproduced red-on-baseline and green-on-candidate?

### What was genuinely tested

`ProvisionalCoderQualification` implements the local controller. A `ReviewedRepairCase` pins exact Git HEAD, source bytes, **fixed test bytes**, one reviewed failing pytest node, and Diagnostician entrypoint. Its error policy refuses wrong checkout, missing/changed test, no semantic failure, collection or import problems, forged/stale evidence, out-of-scope proposals and unverified candidate results.

- Local fault discovery: run reviewed unchanged tests in isolated sandbox; only exact matched failure with a semantic assertion may proceed.
- Strategy: deterministic `minimal_exact_span` at one approved file/test. The model cannot choose a different repair strategy or modify the test.
- Patch: `ExactSpanPatchProposer` produces a bounded exact one-span candidate. The original source is never edited.
- Independent execution: `ConstrainedSelfRepair` reruns unchanged baseline and candidate in separate no-network, read-only, unprivileged Docker copies.
- Local feedback: default **one attempt**, optional **two attempts maximum**. `first_attempt_correct` and `repaired_after_feedback` are separately reported. A failed proposal never creates a trusted handoff.
- Handoff: `verify_review_handoff` reopens exact source/strategy/fault/trial digests. Production verification additionally requires genuine parent-only Ministral inference receipts; it will not accept an injected CPU model fixture.

### Independent results and limits

- [Source-pinned CPU focused/full CI #44163249783](https://github.com/kys928/Hephaestus/actions/runs/44163249783): **38 focused passed; full 1,382 passed / 90 skipped / 0 failed**. These are finite-control tests with declared fake Coder and fake runner, not actual model competence.
- [Real rootless Docker acceptance #44163646848](https://github.com/kys928/Hephaestus/actions/runs/44163646848): real baseline red, proposed candidate green, original checkout unchanged, trusted review handoff digest created. **Its Coder was an explicitly declared test double**, and the strict live parent authentication gate correctly refused to certify it.
- Prior development real model example: [Qwen2.5-Coder 7B #44152786543](https://github.com/kys928/Hephaestus/actions/runs/44152786543) demonstrates a genuine model-generated patch, one failed attempt and independently verified second-attempt success; this is **not** a current Ministral performance/selection result.
- The first native pinned-parent Ministral inference and a fully integrated autonomous Diagnostician + Planner + Coder + independent Verifier + Evaluator + Judge original-stack campaign **have not been run**. Do not claim full-stack certification, first-pass model reliability, or GPU success from software tests.

## Release and certification gates

These remain OPEN:

- [ ] Independently authorized GPU deployment of original pinned parent model with exact cost/time and external Pod termination controls. A generic Diagnostician grant must **not** authorize Coder inference.
- [ ] Verify actual parent weight identity, original tokenizer backend, local cache/source revision, genuine generated token receipts, bounded decoding and retry budgets; no Planner LoRA.
- [ ] Complete a real unknown-to-Coder defect -> Planner-authorized strategy -> actual parent patch -> independent sandbox -> qualified handoff; include no-bug/infra negative controls.
- [ ] Demonstrate exact source/test hashes, immutable acceptance handoff, crash/replay/idempotency, no failed candidate propagation and no automatic merge. Preserve every unsuccessful attempt and distinguish first-pass from bounded-retry success.
- [ ] Independently review source security, approval boundaries and integration with original production eight-phase control spine.
- [ ] Only after full-stack certification, launch candidate comparison/training/certification under [issue #215](https://github.com/kys928/Hephaestus/issues/215).

This work follows the Hephaestus rule: **a model proposal is not truth, and a verified local patch is not permission to edit production**.
