# Issue #102: original GPU activation boundary (2026-10-28)

**Pending external GPU authorization; no RunPod GPU execution.** CPU-only source preparation and contract tests are not GPU scientific evidence.

## Actual prepared source and authorization request

- Original continuous native CPU 45/45, 96 AdamW receipts: https://github.com/kys928/Hephaestus/actions/runs/40753588466
- Original Tier A source and corrected 96 chat/EOS records independently verified: https://github.com/kys928/Hephaestus/actions/runs/40762010344
- FRESH prepared Tier A CUDA native source, SQLite, artifacts and unsigned request: https://github.com/kys928/Hephaestus/actions/runs/40770612111 ; artifact ID 12852412701 ; artifact ZIP sha256:1f5c970ea210d0c6fe87a9ce37b760da31052613c284d02ade6ac4fe206fc5e8.
- Actual pending unsigned request SHA: sha256:f4a9726bc42044a3a62369ac1245ba32d3ff2be413480b2958cb6dc36fa41c49.
- Native source policy SHA: sha256:79b949c946b9c979eb4ab9830aad53280c4ded25cee5d3bb978d3eddbcc22dbb.
- Exactly nine GPU-consuming operation IDs: five certified original CUDA-role inference points, one CUDA optimizer lifecycle, and three CUDA generated-task evaluation operations. This does NOT authorize them.
- Separate fixed GPU experiment spec: configs/research/tier_a_original_cuda_45_v1.json.

## External compute authorization

The operator inspects the unsigned request and independently signs it using the offline scripts/operator_sign_original_cuda_request.py with a private owner-only file (mode 0600), real operator ID and UTC expiry. Production signing keys NEVER belong to the training model, repository, or GPU worker. A signed request must be saved with the byte-identical prepared source on the mounted original network volume. The trusted operator public key and hash must be independently provisioned to the GPU worker in a protected environment. No valid external production signature has been supplied or obtained in this conversation.

## Blocking requirements before paid RunPod allocation

1. Verify the actual signed grant, pinned source/run/operation IDs, hard time ceiling, volume identity and source byte hashes.
2. Inspect actual available one-GPU device, hourly price, approved total cost, immutable five-role HF model snapshots, original adapter archives, separate 135M target, pinned Docker image and persistent work directory.
3. Use the existing RunPodExecutionAdapter.create_bounded_gpu_pod, with a durable idempotency/reconciliation store. Do not blindly issue duplicate pod-create requests.
4. Establish an independent always-on pod termination supervisor (including failures and cancellation). Internal model walltime limits DO NOT stop RunPod billing. Without that real supervisor, paid autonomous launch is NOT permitted.
5. The signed GPU execution lease is started only when execution is authorized using AuthorizedCudaResearchContext; the older CPU source preparation clock is not borrowed as GPU spending authority.
6. On the original uninterrupted CUDA 45-stage completion, verify 5 certified CUDA role inference receipts, 96 original actual CUDA optimizer receipts, original generated-answer baseline/candidate receipts and original no-promotion Judge outcome.
7. Freeze content-addressed private evidence using scripts/freeze_original_tier_a_cuda_archive.py and retain_original_cuda_evidence_s3.py, then cold-read the private archive in an independently reviewed environment with original-cuda45-independent-s3-readback.yml. GitHub receives public-safe hashes only.

**Current scientific status:** CPU 45/45 verified; original CUDA 0/45; independent original historical source leakage/Granite comparison/Ed25519 scientific certification not obtained; issue #102 must remain open. This PR is execution-admission code and source preparation, NOT a provisioned pod or external science certificate.