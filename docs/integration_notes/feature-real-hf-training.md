# Integration note: useful real Transformers causal-LM training

## Branch, base, and scope

- Branch: `feature/real-hf-training`
- Immutable base: `552200b95b027ed91ac76167619ac9478100e811`
- Backend ID: `transformers_causal_lm`

This feature adds bounded full fine-tuning for causal language models through
PyTorch and Hugging Face Transformers. It is a separate lifecycle implementation;
the existing dependency-free `LocalTrainingLifecycleService` and its two-parameter
fixture worker remain unchanged and continue to provide fast lifecycle tests.

The feature does not select an experiment hypothesis, acquire data, bypass
approvals, evaluate semantic quality, generate text, promote checkpoints, change
shared contracts, or wire the orchestrator.

## Public classes and helpers

- `hephaestus.training.TransformersTrainingLifecycleService`
  - structurally conforms to `TrainingLifecycleService`;
  - implements `launch`, `status`, and `control`;
  - runs training in a separate OS subprocess;
  - supports honest preparing, running, interrupting, interrupted, resuming,
    completed, failed, and cancelled states.
- `hephaestus.training.hf_worker`
  - imports optional ML dependencies only inside the subprocess;
  - performs real autoregressive full fine-tuning;
  - persists metrics, checkpoints, resume state, and final handoff evidence.
- `hephaestus.backends.hf_causal_lm.transformers_training_capability`
  - detects the optional capability without importing PyTorch or Transformers.
- `directory_content_identity`
  - hashes every regular component of a local model/tokenizer directory.
- `validate_checkpoint_manifest`
  - verifies the finalized checkpoint component set and all SHA-256 hashes.

## Optional dependencies

Core Hephaestus imports and the fixture lifecycle require no ML dependency. The
real backend requires `torch`, `transformers`, and `tokenizers`. `accelerate` and
`peft` are detected and recorded if installed but are not required and are not
used by the baseline full-fine-tuning path.

The repository currently has no Python packaging metadata or governed training
extra. Until a packaging-owned change adds one, install the capability explicitly
in the runtime environment, for example:

```bash
python -m pip install "torch>=2.2" "transformers>=4.40" "tokenizers>=0.19"
```

Every prepared job records the exact installed framework versions. Absence
returns a failed `TrainingRunHandle` with the blocking
`transformers_training_unavailable` / `unsupported_capability` issue; it never
falls back to the fixture or reports a real model run.

## Model and tokenizer support

The service supports models loadable by `AutoModelForCausalLM` and tokenizers
loadable by `AutoTokenizer`.

Local loading requires:

- explicit model and tokenizer directories;
- a `sha256:<digest>` content identity for each directory;
- exact architecture family, parameter count, vocabulary size, context length,
  and special-token IDs;
- `trust_remote_code=false`.

Remote loading is opt-in and requires all of:

- provider and external-download enablement;
- immutable 40–64 hexadecimal model and tokenizer revisions;
- explicit `local_files_only` mode and cache directory;
- license and provenance evidence;
- one or more approval references;
- remote code disabled.

The loader passes `token=false`, so it cannot silently consume ambient Hugging
Face credentials. Authenticated/private registry loading remains an integration
dependency until the existing secret-reference resolver is wired explicitly.

The worker verifies loaded architecture, parameter count, vocabulary, context
capacity, and tokenizer special-token IDs. It does not silently replace a
tokenizer, revision, padding token, or EOS token.

CPU float32 and optional CUDA float32/float16/bfloat16 execution are supported.
The required test model is a locally constructed, one-layer GPT-2 configuration;
no required test downloads a model. This proves the mechanics of real parameter
updates, not useful large-scale model quality.

## Trainable-data requirements

Training consumes three distinct artifacts:

1. the existing `TrainableDataContract` JSON record;
2. the contract's actual processed `trainable.jsonl` artifact;
3. the data factory's `processing_evidence.json` sidecar.

The proposal must provide content hashes for all three. Preparation verifies:

- `trainable-data.v1` schema;
- contract-to-processed-artifact reference equality;
- processed bytes and SHA-256 identity;
- non-empty bounded JSONL rows containing `text`;
- explicit wrapper and prompt/target-boundary evidence;
- checked, positive tokenizer compatibility evidence for the exact tokenizer;
- row, byte, and declared token bounds.

The contract record is never treated as the dataset.

## Tokenization and labels

The worker uses causal next-token labels. For `record_kind=prompt_target`, it
masks the prompt and boundary tokens with label `-100` and trains only on the
target plus EOS. Plain-text rows train on all non-padding tokens. It records:

- right truncation to the explicit context length;
- right batch padding with the pinned pad token;
- EOS append-if-missing behavior;
- prompt-masking policy and ignored label token;
- source, encoded, dropped, truncated, and prompt-masked sample counts;
- total token count.

Samples with no supervised token after truncation are dropped and counted. Empty
output, incompatible special tokens, and total-token bound violations fail.

## Configuration and resource estimates

`normalized_training_config.json` records the seed, model/tokenizer/data identity,
architecture, vocabulary and special tokens, optimizer, scheduler, learning
rate, warmup, batch size, gradient accumulation, steps/epochs, clipping, weight
decay, precision, device, deterministic dataloader settings, checkpoint/log
cadence, framework versions, environment summary, and tokenization policy.

A canonical SHA-256 fingerprint covers compatibility, normalized configuration,
and resource-estimate inputs. The backend requests deterministic PyTorch
algorithms and records any limitation; it explicitly does not claim perfect
determinism across every hardware/framework combination.

Before launch, `resource_estimate.json` conservatively estimates parameter,
gradient, AdamW optimizer-state, activation, peak, dataset, and checkpoint bytes,
plus expected steps. Observable CPU or CUDA free memory is recorded. A prepared
configuration is blocked when estimated peak memory exceeds 80% of observed
available memory. The estimate is evidence with stated uncertainty, not a runtime
guarantee.

## Runtime artifacts

Each run writes under `<artifact_root>/<run_id>/`:

- `prepared_job.json`
- `normalized_training_config.json`
- `resource_estimate.json`
- `events.jsonl`
- `metrics.jsonl`
- `metrics_summary.json`
- `tokenization_summary.json`
- `runtime.log`
- `incidents.jsonl` when needed
- finalized `checkpoint_step_<n>/` directories
- `checkpoint_record.json`
- `resume_token.json`
- `runtime_result.json`
- `final_result.json`
- `handle.json`

Step metrics include loss, learning rate, step, fractional epoch, token/example
counts and throughput, gradient norm, elapsed time, and observable memory. A
non-finite loss or gradient norm fails safely. Training loss is explicitly marked
as optimization evidence, not semantic evaluation evidence.

## Checkpoint integrity and generation handoff

Every checkpoint is first written to `checkpoint_step_<n>.partial`. The worker
saves model, tokenizer, optimizer, scheduler, RNG, step/epoch, compatibility, and
configuration fingerprint state. It hashes each component, writes a canonical
component manifest, marks `partial_write=false`, and atomically renames the
directory only after finalization.

Partial directories are never resumable. The lifecycle independently verifies
the final component set, every SHA-256, the manifest hash, and the checkpoint
record before exposing the checkpoint reference.

`loading_instructions.json` gives the separate generation bridge the finalized
model and tokenizer artifact references, architecture and revision identity,
backend, remote-code setting, and integrity-manifest reference. Generation is not
implemented here.

## Interrupt, cancellation, and resume

- Interrupt sends a graceful signal and becomes `interrupted` only after a real,
  manifest-verified resumable checkpoint exists.
- Cancellation sends a graceful termination signal and becomes `cancelled`; it
  is never reported as successful completion.
- Resume is allowed only from `interrupted` and reloads model, tokenizer,
  optimizer, scheduler, and RNG state from the verified checkpoint.

Resume requires exact model/revision, tokenizer/revision, architecture, training
mode, data contract/hash, processed data/hash, optimizer, scheduler, configuration
fingerprint, backend, and checkpoint-manifest compatibility. Loose or partial
loading is not available.

## OOM and failure behavior

CPU and CUDA OOMs are normalized to `out_of_memory` incidents containing the
device, attempted batch/context, resource-estimate reference, observed error type,
and conservative suggested reductions. Suggestions are evidence only: the
service does not silently change batch size, context, accumulation, or any other
experiment variable and does not retry automatically.

Non-zero exits, missing terminal artifacts, invalid result records, hash drift,
partial checkpoints, and process-restoration ambiguity become explicit failed
handles and persisted incidents.

## Tests and limitations

`tests/test_real_hf_training.py` always tests dependency-free imports, explicit
unsupported capability, contract/hash/tokenizer/special-token failures, remote
download governance, remote-code refusal, and partial-checkpoint refusal.

When the optional training dependencies are installed it additionally tests:

- real tiny causal-LM parameter updates on CPU;
- persisted metrics and complete artifacts;
- component-manifest checkpoint integrity;
- graceful interrupt and strict resume;
- incompatible revision and configuration-fingerprint refusal;
- cancellation;
- controlled CPU OOM normalization;
- repeated deterministic tiny CPU runs within the stated tolerance;
- optional CUDA execution, skipped when CUDA is absent.

There is no required or optional online test in this branch. Remote acquisition
is supported by policy and loader boundaries but is deliberately not exercised in
CI. Authenticated private registries, LoRA/PEFT, distributed training, shuffling, streaming datasets, multi-node
recovery, automatic batch tuning, and semantic generation evaluation are not
implemented. Checkpoints may be large because the correct baseline stores full
model plus AdamW state.

## Final orchestration wiring

The final integration branch should:

1. choose this service only when model selection declares
   `transformers_causal_lm` compatible;
2. preserve planner-issued experiment constraints and all resolved approval
   evidence;
3. pass the real `TrainableDataContract`, processed JSONL, and processing-evidence
   references/hashes from the data factory;
4. persist the returned handle and evidence references through existing state
   boundaries;
5. let the runtime monitor inspect events/incidents without altering the recipe;
6. hand finalized loading instructions to generation;
7. let evaluation and Judge phases determine quality and promotion separately.

Packaging metadata for a formal `training` extra and generation-bridge wiring are
integration dependencies owned outside this branch.
