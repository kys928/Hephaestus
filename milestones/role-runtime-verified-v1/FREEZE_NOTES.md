# role-runtime-verified-v1 — research-only evidence freeze

**Evidence status:** verified runtime integrations; **not** scientific certification and **not** autonomous production authorization.

This snapshot freezes exact source revisions, registered five-role base model/LoRA identities, original artifact hashes, bounded CPU and CUDA result scopes, and known gaps. It is deliberately separate from pending service-integration PRs #97, #98 and #99. The frozen branch/release does **not** claim these draft PRs have merged into production.

## Demonstrated

- Genuine five-role pinned CUDA inference: Controller, Diagnosis, Planner, Granite Evaluator, and Judge each produced a valid response with a SHA-256-addressed inference receipt, using original models/revisions and SHA-checked original LoRA archives. No proposed action was executed.
- CUDA hardware: RTX PRO 6000 Blackwell Server Edition, PyTorch 2.14.1+cu132, CUDA 13.2, single GPU. Original GPU run id `hephaestus-five-role-cuda-40190358524`, [Actions run](https://github.com/kys928/Hephaestus/actions/runs/40190358524). Five-role generation took 93.54 seconds; the launcher deleted the Pod.
- CPU research-only binary-feature network: five actual inference passes from reloaded trained tensors; 22 AdamW steps on 80 synthetic training examples. This is **not** a causal language model or a scientifically independent role judge.
- Separate tiny GPT2 two-step genuine LoRA optimizer and checkpoint/generation/evaluation/Judge test. This is **not** improvement of Granite.
- A real evidence-backed control spine completed first **5 of 45** operations; the sixth, Planner intervention ranking, was blocked and subsequent 39 were not executed. See [CPU CI](https://github.com/kys928/Hephaestus/actions/runs/40190607106).

## Durable evidence and integrity

The milestone manifest pins original GitHub Actions ZIP hashes, contents hashes, immutable Git source identities, five LoRA adapter S3 paths and SHA-256 digests, and original GPU terminal result SHA-256. A one-shot release workflow independently re-downloads and hashes both original Actions archives, re-reads the RunPod volume's original GPU terminal object, and verifies each of the **five original role inference receipt bytes** by SHA-256. Only after all checks does it attach newly repackaged ZIPs containing the **exact verified original JSON member bytes** (the original GitHub Actions ZIP SHA-256 values remain pinned in the manifest, but ZIP container bytes differ) to a nonexpiring GitHub Release `role-runtime-verified-v1` and create its commit-bound tag. The workflow is designed to **refuse an existing tag**, not silently move or replace one.

The original GPU evidence is also retained on the RunPod network volume. The CPU evidence in the release consists of measured JSON summaries and SHA references; temporary CPU model and PEFT weights and original individual CPU state blobs were **not** independently archived. Do not infer those bytes are recoverable from the published summaries.

## Not demonstrated

1. One uninterrupted complete **45-operation** CPU scientific-worker campaign.
2. One uninterrupted complete **45-operation** GPU scientific-worker campaign.
3. Independent semantic/near-duplicate original-text leakage scan or real sealed-data custody review.
4. Controlled, repeatable Granite improvement over frozen real baseline.
5. Independent external scientific certificate, permission to auto-promote or production deployment.

## Next gates

1. Freeze this exact nonproduction runtime research result and verify release asset hashes.
2. Add genuine CPU causal-LM role reasoning behind a separately governed contract; integrate all 45 real scientific domain services without mocks, forced verdicts or invented approvals. Validate content hashes across the entire campaign in one run.
3. Replay that same integrated 45-step scientific workflow with the already validated frozen five-role CUDA engine.
4. Independently seal and audit holdout data, test Granite improvement with repeated seeds and quantitative controls, and only then pursue an external certificate.

Do not substitute a successful synthetic smoke test for scientific performance evidence.
