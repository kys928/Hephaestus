# Explicitly authorized A40 D inference launch marker

The user approved an available VRAM-sufficient GPU, a fresh inference run, and spending up to approximately $2 on 2026-10-23. This commit is the **single** GitHub Actions launch trigger.

Exact hardware: 1x NVIDIA A40 (48 GB, Ampere BF16) Secure Cloud EU-SE-1, confirmed by read-only per-GPU stock request [#39974240624](https://github.com/kys928/Hephaestus/actions/runs/39974240624) as Low $0.59/h immediately before launch.

Original S3 source/model data: network volume `cviwpryzao` in EU-CZ-1. It is not mounted cross-region. The A40 pod requests the pinned read/write S3 API over HTTPS and saves its own independent progress evidence under a new run ID. No source training/eval data changes.

Immutable scientific scope: step-96 Granite 3.3 8B D adapter SHA256 `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`, same 56 deterministic prompts/60 paired contrasts, BF16 greedy max 112; inference-only, no retraining, certification or deployment.

One actual GPU provisioning attempt, new unused atomic S3 approval ID `heph-D-contrasts-A40-EUSE1-20261023-ONE-PAID-ATTEMPT`, maximum $0.59/hour and **$1.90 estimated total including teardown reserve**. Launch may fail closed if a live stock check is insufficient; only an explicitly verified zero-pod preallocation capacity failure may be rechecked. Any create request consumes the attempt; no second pod. Independent CPU/S3 scientific audit and verified delete required for success.

No merges. One trigger. The marker commit message below is the user-approved launch request.
