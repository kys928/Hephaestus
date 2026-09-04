# Second controlled experiment — frozen execution evidence

This directory freezes the verified evidence produced by GitHub Actions workflow `33889959923` at exact execution commit `e4f6298e911524f3ad91673ec4b8aac8d45f1bfb`.

## Controlled experiment

- experiment: `experiment-d0e911d6bd1fb7ae`
- candidate run: `planned-run-b8e558e54effac85`
- baseline training run: `first-bounded-scientific-training-001-33866198758`
- primary variable: `dataset_mixture`
- candidate processed dataset: `sha256:bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac`
- random initialization: `sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39`
- tokenizer: `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`

The scientific recipe remained fixed at 100 optimizer steps, batch 8, context 256, AdamW, linear scheduler, learning rate 0.0005, warmup 10, weight decay 0.01, gradient clipping 1.0, float32 CUDA, seed 1729, deterministic sequential loading, and checkpoint at step 100.

The generic tokenization admission ceiling was raised from 20M to 40M only after an earlier zero-step failure. That failed attempt was archived intact and contained no checkpoint or non-zero optimizer-step evidence. The ceiling adjustment does not change the fixed first 100 sequential training batches.

## Verified training result

- checkpoint manifest: `sha256:d06829b4738b2abc170a5a96d4a76c8dbe9a32869f0a8071c26e3b1550e31a64`
- trained model weights: `sha256:35f771de43686daa87146d1a7ca21a1346ed846d3c76516aca77155ef1d47fe9`
- initial weights: `sha256:1bc32006ae216fb4d6b7dd5ce66241e487d552ffcebbbb87c6a029cd22074ce1`
- weights changed: true
- optimizer steps: 100
- examples processed: 800
- tokens processed: 165,788
- final training loss: 5.219187259674072
- final gradient norm: 2.4454634189605713
- optimization elapsed: 2.783882185816765 seconds
- throughput: 59,552.8075306676 tokens/second
- finite: true

Training Pod: `krf96688rz1wca`. It was deleted successfully after independent checkpoint verification. Estimated observed training-Pod cost was approximately 0.0546044 at 0.74/hour; this is execution metadata, not a billing ledger.

## Frozen semantic evaluation and Judge

Evaluation run: `second-controlled-semantic-evaluation-001-33889959923`.

The same frozen `semantic_behavior_v1` pack generated and independently verified 36 samples:
- 18 baseline samples from the previous Wikitext-trained checkpoint;
- 18 candidate samples from `planned-run-b8e558e54effac85`.

Soft aggregate scores:
- baseline mean: 0.03333333333333334
- candidate mean: 0.4138888888888889
- overall delta: +0.3805555555555556

Dimension changes:
- repetition: +1.0, improved
- excessive_length: +1.0, improved
- coherence, continuation, instruction_adherence, malformed_structure, relevance, and termination: equivalent within available deterministic evidence

Despite the positive soft aggregate, the candidate failed frozen hard checks for all 18 task/seed pairs. Deterministic precedence therefore produced:
- primary outcome: `regressed`
- deterministic gate: `failed`
- recommendation: `reject_candidate_evidence`
- confidence: 0.7071067811865476
- variance risk: low
- formal significance claim: false

Judge exit:
- verdict: `blocked`
- next action: `reject_checkpoint`
- action was not applied by the workflow

Evaluation Pod: `h10x5xp5hbkarz`. It was deleted successfully. Estimated observed evaluation-Pod cost was approximately 0.0120621 at 0.74/hour; this is not a billing ledger.

## Artifact provenance

GitHub Actions artifact:
- artifact ID: `9943623899`
- ZIP digest: `sha256:54c62975c41fe368cc5e1c1ee194a1e4961e4c7dcde744e509fc75563daa5afa`
- workflow: `33889959923`

The ZIP itself is not committed. Its four JSON payloads are frozen here verbatim and checksummed in `SHA256SUMS`.
