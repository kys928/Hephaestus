# Integration note: first bounded scientific training

## Scope

This integration binds the first verified non-fixture scientific dataset/tokenizer/model state into the real `TransformersTrainingLifecycleService`, executes one bounded GPU training run on RunPod with Network Volume `cviwpryzao` mounted at `/workspace`, independently verifies the finalized checkpoint through the RunPod S3 interface, and tears the Pod down.

This is a lifecycle proof at `smoke_test` stage. It is not a semantic-improvement or promotion claim.

## Immutable input bindings

The successful run used the previously independently verified bootstrap state without changing its scientific content identities:

- dataset: `Salesforce/wikitext`
- immutable dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- raw dataset: `sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`
- processed trainable dataset: `sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c`
- tokenizer directory identity: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`
- random-init model directory identity: `sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39`
- bootstrap bundle: `sha256:6774e92d2b595353a18211ffa772fb82b362d462ee2e3c26144f705d26525436`
- source experiment: `experiment-60bff7cb4f478f91`

The runtime binding rewrites only local filesystem references required by the lifecycle. The processed JSONL is copied byte-for-byte and rehashed before launch; model and tokenizer directories are re-identified from their actual mounted component bytes before launch.

## Approval boundary

The original proposal required the named approval `model_selection_approval`. The explicit operator request to execute the first bounded scientific run is recorded as:

`approval://operator/explicit-request-2026-09-04-first-bounded-scientific-training`

Launch proceeds through `AutonomousExperimentCoordinator.launch_approved(...)`; the required approval key is not removed or bypassed.

## Successful live execution

GitHub Actions run: `33866198758`

Exact execution commit: `29e64530bc54560df84e4b4d93959b8dfdee38e2`

Run ID: `first-bounded-scientific-training-001-33866198758`

RunPod Pod: `gm5vea0zpdm8y8`

Environment:

- datacenter: `EU-CZ-1`
- Network Volume: `cviwpryzao`
- GPU observed by PyTorch: `NVIDIA GeForce RTX 4090`
- GPU memory observed: `50,864,390,144` bytes
- CUDA runtime: `12.6`
- PyTorch: `2.14.0+cu126`
- Transformers: `5.16.1`
- Tokenizers: `0.23.2`
- Python: `3.12.3`

Bounded recipe:

- steps: `100`
- batch size: `8`
- context length: `256`
- optimizer: `adamw`
- scheduler: `linear`
- learning rate: `5e-4`
- warmup: `10` steps
- gradient clipping: `1.0`
- weight decay: `0.01`
- seed: `1729`
- dtype: `float32`
- device: `cuda`
- shuffle: disabled for deterministic sequential loading
- checkpoint cadence: final step only (`100`)
- wall-time guard: `1200` seconds

Lifecycle resource evidence recorded 54,878 processed rows, 15,920,505 dataset bytes, and a conservative estimated peak of 42,577,920 bytes against 50,450,006,016 bytes of observable free CUDA memory at preparation time.

## Training evidence

The lifecycle completed with no issues and process return code `0`.

Metrics summary:

- optimizer steps: `100`
- examples processed: `800`
- tokens processed: `35,854`
- final training loss: `7.302044868469238`
- final gradient norm: `0.9406019449234009`
- finite metrics: `true`
- measured optimization-loop elapsed time: `2.313926801085472` seconds
- measured throughput: `15,494.8721727847` tokens/second
- maximum CUDA memory allocated by the training process: `245,344,256` bytes

These metrics prove bounded optimization executed. Training loss is not treated as semantic evaluation evidence.

## Final checkpoint integrity

Final checkpoint path:

`/workspace/hephaestus/scientific/v1/runs/first-bounded-scientific-training-001-33866198758/checkpoint_step_100`

Checkpoint manifest identity:

`sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3`

Independent S3 verification re-read all seven checkpoint components and reproduced the manifest identity. Important components include:

- trained `model/model.safetensors`: `sha256:0f8898ad054bc52798599b7048c2c98523916ea0dca729200c1ee433c6f09001` (`7,503,792` bytes)
- model config: `sha256:e4bae2d91ba1c40722209bf43fadb83dff7260cfb7d39f7d37d89a0e500ced80`
- tokenizer JSON: `sha256:3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25`
- training state: `sha256:d1301193c198c18793ec5d4ff9e0720368e6aa47929f4c76ee9023ee3c660b82` (`15,053,567` bytes)
- generation handoff/loading instructions: `sha256:25ca4b00b47d989a9cb16f449157b68fd51579c7dbc2ad30ac1cb271e7b7aeab`

The trained model weight SHA differs from the random-init input weight SHA, providing direct byte-level evidence that the model weights changed during optimization.

The independent verifier also checked the runtime-bound processed dataset and reproduced the original processed-data SHA-256 exactly.

Run inventory:

- objects under the run prefix: `22`
- total run bytes: `23,165,359`
- inventory hash: `sha256:d852c63df163548d20c44fbb4cdf3533566a0ea46f5aa11472e57cc797c8cd22`

## Cost and teardown

The successful Pod reported `0.74` credits/currency units per hour. Observed launcher lifetime was `128.78562032` seconds, giving an estimated cost of `0.0264725997`. This is an estimate from observed elapsed time and the Pod's reported hourly rate, not a billing ledger.

The Pod was deleted successfully with HTTP `204` after verification. `funds_unavailable=false`.

An earlier Pod reached bootstrap but failed before training because Ubuntu's PEP 668 protection rejected installation into the system Python. That Pod was also deleted successfully and had an estimated observed cost of about `0.07179`. The production bootstrap now creates an isolated virtual environment with `--system-site-packages`, retaining access to the image's CUDA-enabled PyTorch without mutating the externally managed system Python.

Attempts rejected by the RunPod request schema before Pod creation incurred no GPU runtime.

## GitHub evidence

Successful workflow: `33866198758`

GitHub Actions evidence artifact:

- name: `first-bounded-scientific-training`
- artifact ID: `9934115699`
- uploaded ZIP SHA-256: `6b5218f6996f95da31094366c70970c3774a81f8e8accb7faf46b48560938d1e`

The artifact contains the launcher record and independent volume verification record. Generated checkpoints, weights, datasets, runtime state, and logs remain on the Network Volume and are not committed to Git.

## Scientific boundary after this run

The first real training lifecycle is now proven end-to-end:

verified inputs → approval binding → real Transformers lifecycle → CUDA optimization → finalized checkpoint → component-hash verification → generation handoff → independent S3 readback → Pod teardown.

The next scientific boundary is evaluation. The checkpoint must be exercised through the frozen generation/evaluation bridge and compared against the random-init baseline using deterministic and semantic evidence before Judge exit can make any promotion/reject/branch decision. No claim of model quality improvement is made from training loss alone.
