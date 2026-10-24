# Production PEFT lifecycle

Extends the existing Transformers lifecycle, subprocess worker, checkpoint manifest,
and generation backend. No second trainer or paid workload was introduced.

## Execution path

`ProductionCompositionRoot.training_lifecycle` → existing guarded lifecycle →
`TransformersTrainingLifecycleService` → `hf_worker` → immutable component manifest →
`TransformersCausalLMGenerationBackend` → existing generation/evaluation service.

Supported modes are `full_finetune`, `lora_finetune` and `lora_continuation`.
The latter requires a SHA-256 directory identity and base-identity provenance for
its parent adapter, matching rank/alpha/dropout/targets/bias, and explicitly declares
`adapter_warm_start`. It resets optimizer/scheduler state. It is not represented as
faithful continuation of another optimization trajectory.

Interrupted-run resume preserves the same immutable job, adapter weights, AdamW
state, scheduler state, Python/Torch/CUDA RNG state, counters and deterministic
sequential batch position. No shuffle or unrecorded dataloader state is supported.
Checkpoint receipts retain parent identity and exact base/tokenizer revisions.
Only LoRA parameters enter AdamW; a non-adapter trainable parameter fails closed.
The pre-existing masking, tokenization, finite-gradient checks, step budgets and
resource estimator are reused. LoRA estimates remain conservative full-model bounds.

Generation verifies the checkpoint component manifest, adapter provenance and exact
base identity, loads the pinned base offline and attaches the exported adapter.
The global certified-role registry and routing flags are unchanged.

Recovery also fixes a pre-existing controller-restart gap: a child that completed
before its parent recorded completion can now be collected using its matching
run/fingerprint result and verified checkpoint, without starting another process.
Missing terminal evidence still blocks; PID liveness alone is not completion.

## Validation

Full suite with optional CPU ML dependencies installed: **578 passed, 2 skipped**.
Three `real_cpu` tests execute actual locally initialized tiny GPT-2 LoRA optimization,
adapter warm start, hash rejection, adapter generation, and interrupt/reconstruction/
resume. The resumed adapter tensors exactly match the uninterrupted run. Models and
datasets are local synthetic fixtures, not downloaded production weights.

CI has a separately named real CPU PEFT job, distinct from dependency-free regression
and provider/model simulations. Install with `.[test,peft-training]` and CPU PyTorch.
Run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_peft_training_lifecycle.py -q`.

This is CPU optimization evidence, not GPU validation, useful Evaluator improvement,
five-role model intelligence, scientific certification, or production readiness.
