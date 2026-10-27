# Real existing-adapter CPU PEFT + five-role research integration — verified

**Executed:** 2026-10-27, exact source SHA `0e4af338452c592f1369459069ac9d23fa019fd2`  
**Dedicated CI:** [run #40565752780](https://github.com/kys928/Hephaestus/actions/runs/40565752780): **6/6 tests passed**, real PEFT, generation, model-role and replay steps completed.  
**General CPU regression:** [run #40565759735](https://github.com/kys928/Hephaestus/actions/runs/40565759735): all jobs passed.  
**Exact original GitHub Actions evidence:** [artifact #12761674822](https://github.com/kys928/Hephaestus/actions/runs/40565752780/artifacts/12761674822).  
**Downloaded ZIP SHA-256 independently checked:** `b9da918887c880498426ad3a205fa67e85e8515d4598c1c5f25a90da66c8840b`.  
**Content custody:** 59/59 member files in the ZIP verified against the run's original `cpu-peft-custody.json`; no missing or mismatched bytes.

## Real adapters used, not simulated scientific workers

- Created 100 author-written, mechanically generated arithmetic-to-JSON examples, verified the exact 80/20 development split, and pinned task-specific JSON correctness and schema checks before training. This has a genuine local development-use basis, **not any third-party training license or production dataset selection approval**.
- Materialized real trainable JSONL, `TrainableDataContract`, preprocessing evidence and tokenizer unknown count. No other dataset was imported, no synthetic acquisition approval was issued. The separate `ApprovedSourceCpuPeftHandoff` directly delegates to **the existing production `StagedExperimentFinalizer._data`**, and fails closed unless workflow-owned discovery, selection, matching scoped acquisition approval, actual receipt, processed bytes, contracts, hashes and tokenizer evidence all exist. The branch has a negative regression test for absent approvals. **No live external source was cleared or acquired.**
- Used `TransformersTrainingLifecycleService`, **eight true CPU LoRA AdamW optimization steps**, the real worker's finalized adapter weights (`adapter_model.safetensors`) and checkpoint integrity manifest `sha256:10befde256880497f8ee8c5f4bf4d872961671d64fd441aa6b8988e085705c4f`. The development base is a small random-initialized locally generated GPT-2-style causal LM; it is not a pretrained production checkpoint.
- Used **`TransformersCausalLMGenerationBackend` on both frozen baseline and post-PEFT candidate**, scoring observed text with existing `evaluate_behavioral_sample` and `JudgePolicy`. The saved decoding settings symmetrically suppress special-token-only output from the tiny random model; no synthetic response is inserted. A genuine saved baseline checkpoint and a verified adapter checkpoint are used.
- Used `CpuCausalProposalScorer` from the existing research role bridge, with the **pinned actual 135M pretrained CPU model** and verified real PEFT/Judge evidence, to perform five actual role-conditioned research inferences (Controller, Diagnosis, Planner, Evaluator, Judge). All five scored multiple admissible candidate continuations and generated hash-verified model-inference receipts. All five run from **the same** pretrained 135M research LM, not the original distinct five CUDA masters. Model action flags remain false.

## Findings: a real completed experiment, not a useful improved model

- Task-specific held-out correct arithmetic: **0/20 baseline → 0/20 candidate**.
- Typed valid JSON on these development checks: **0/20 baseline → 0/20 candidate**.
- The deterministic Judge action was `continue_lineage_best`; **no promotion, no substantive improvement**. Recommendation: investigate arithmetic generalization.
- Five role-conditioned research model preferences from actual verified results:
  - Controller: `noop` / `noop`;
  - Diagnosis: `data_coverage` / `replace_or_mix_dataset`;
  - Planner: `replace_or_mix_dataset` / `branch_new_experiment`;
  - Evaluator: `evaluation_invalid` / `request_recheck`;
  - Judge: `blocked` / `continue_lineage_best`.
- **These are suggestions from conditional likelihood**, not independent evidence that data coverage *caused* the failure or that the evaluator is invalid. Only later deterministic evidence and governed approval may authorize a dataset or evaluation change.
- Correctness requires more than an emitted schema. PEFT/base training from random initialization in eight steps cannot reasonably be mistaken for a pretrained language-model improvement result. Repetition and independent protected evaluation remain unproven.

## Exact stage separation and remaining integration

This run independently demonstrates the **real PEFT worker + real generation + real deterministic evaluation and Judge + five policy-bounded CPU role inference receipts**, all in one CPU workflow. The data prep used original generated development material; approved-source handoff was exercised as a **fail-closed contract**, not with live approved HF training data.

The **canonical 45-operation** campaign remains **11/45**. This run neither invoked all actual `PHASE_SUBSTEPS` nor satisfied production admission; it cannot claim promotion or scientific certification. `compose_verified_staged_dependencies` explicitly requires `CertifiedRoleReasoner` with five genuine frozen **CUDA** roles. This branch **does not change that rule** or call its model-less CPU research stack a production replacement.

To execute a single genuine **45/45** trace, next: (1) obtain a legitimately approved real-source training contract/receipt, (2) feed that contract into the production `StagedExperimentFinalizer` and existing PEFT lifecycle, (3) use the genuine staged five-role service bindings and semantic evaluation that require actual model identity and approval, and (4) run and replay all 45 operations in one governed workflow. The GPU phase follows only after successful CPU controlled composition under an explicitly approved research protocol, not by weakening the production five-role CUDA guard.

**Independent review verdict:** the CPU *adapter integration vertical slice* is complete, SHA verified and noncertified; actual useful language-model improvement and continuous governed 45-stage execution remain unproven.
