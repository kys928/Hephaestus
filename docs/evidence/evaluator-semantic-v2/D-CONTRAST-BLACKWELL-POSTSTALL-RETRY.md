# D counterfactual inference: post-stall Blackwell retry (approved, one use)

## Confirmed prior teardown
User instructed deletion of failed A40 pod `bnvazbgs2acrp4` on 2026-10-23. Identity-verified deletion through RunPod REST and GET 404 was recorded by [Actions #39982753659](https://github.com/kys928/Hephaestus/actions/runs/39982753659) at **15:58:00 UTC**. The previous A40 S3 approval claim is consumed and **must never be replayed**. A40 reached RunPod RUNNING but produced no S3 bootstrap/model/inference events and no remote container logs after about 54 minutes. Root cause not definitively known; suspected platform container startup stall, not model quality.

## New live hardware choice
User prioritized NVIDIA RTX PRO 6000 Blackwell Server Edition. Fresh read-only provider checks found a positive **Low** one-GPU Secure Cloud price quote **in EU-CZ-1**, the location of our already-existing S3 network volume `cviwpryzao`, at **$2.49/hour**. [CPU+stock evidence #39985643302](https://github.com/kys928/Hephaestus/actions/runs/39985643302), and [off-line gate #39985643310](https://github.com/kys928/Hephaestus/actions/runs/39985643310). Availability is volatile: another live exact quote must pass before S3 approval claim and pod provisioning.

Use **one 96GB RTX PRO 6000 Blackwell Server Edition, Secure Cloud EU-CZ-1**, with the existing volume mounted at `/workspace`. This avoids cross-region S3 latency and reuses any existing Hugging Face, pip, venv or LoRA cache. No fallback GPU or region is implicitly authorized for the same attempt.

The user gave explicit authorization for one fresh retry after terminating the stalled A40. Remaining balance was independently confirmed from RunPod account GraphQL as **at least $2.00** at 16:18 UTC (only threshold status logged). The newly authorized spend is conservatively capped at **$1.80 estimated GPU total** with 120-second teardown reserve, <= **$2.49/hr**. One pod-create attempt only, protected by distinct atomic S3 once claim `heph-D-contrasts-BLACKWELL-EURIS2-20261023-SECOND-USER-AUTH` (historical name includes the formerly considered Iceland candidate, but current approval/config explicitly require EU-CZ-1). Do not increase budget or retry after create attempt.

## Diagnostic engineering fixes
- Replace the generic PyTorch CUDA13 image used by the stalled A40 with official `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, pinned in the explicit pod request. CUDA12.8 supports Blackwell. Use 100GB ephemeral container disk.
- Inject **early S3 bootstrap receipts** into every container startup phase: first stage `container_started`, repository checkout, environment install, model pack build, and `runner_started`. The existing D Python runner additionally commits per-stage immutable S3 events (HF model snapshot, adapter download/validation, tokenizer, BF16 base weights, LoRA, generations).
- If the launcher sees no remotely readable container-bootstrap record after **600 seconds**, it fails closed and deletes/verifies deletion of this single pod. This is a pre-model startup safety trip, not a model judgment.
- Continue spending checks every 10 seconds and mandatory teardown. The runner wall is 2430 seconds and launcher max 2550 seconds, bounded further by the $1.80 cap. This cannot guarantee 56 generations before budget or deadline.

## Frozen science (unchanged)
- Base IBM Granite 3.3 8B Instruct, frozen revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.
- D step-96 LoRA archive SHA256 `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`.
- **Inference only**, no gradient steps/retraining, checkpoint/weights unchanged.
- 60 single-factor counterfactual pairs, **56 unique deterministic prompts** with 112-token maximum; 48 state changes + 12 precedence invariants. Immutable prompt oracle and trace hashes, no sealed certification data.
- Success requires exactly 56 saved generations and independent CPU/S3 scientific rescore of all 60 pairs plus verified pod deletion. Partial records are diagnostic, not scientific promotion.

[CPU validation #39985643310](https://github.com/kys928/Hephaestus/actions/runs/39985643310) PASS: 67 tests, `bash -n` on generated startup script and `actual_allocation=false` dry run; no paid pod yet. Draft branch only; no merge. This scope does not authorize any third pod if the next Blackwell provisioning fails or stalls.
