# RunPod capacity-gate repair — verified live response and single guarded restart

**Root cause confirmed using live read-only GitHub Actions diagnostics, 2026-10-19.**

The exact Secure Cloud one-GPU Blackwell quote at EU-CZ-1 changes shape when capacity is scarce:
1. `stockStatus: "Low"`, finite acceptable price, `availableGpuCounts: null`.
2. Regional `lowestPrice` object with `stockStatus: null`, `uninterruptablePrice: null`, `availableGpuCounts: null`.

Both are documented nullable GraphQL fields; neither establishes usable capacity. Previous `inspect_response` wrongly raised `ValueError` on these shapes and concealed the stage, so the preallocation workflow repeatedly failed as if the provider schema were broken.

**Code repair:** The guard now explicitly returns `confident_capacity_signal: false` with a safe reason for regional Low/None, null regional quotes, or absent quotations. A high/medium 1-GPU regional quote still requires a finite price <= $2.49/hour, exact GPU and EU-CZ-1, and either an explicit available GPU count including 1 or a null count interpreted through the provider's region/gpuCount=1 scoped stock query. Unexpected or partial schemas remain fail-closed with whitelisted, non-sensitive diagnostic error codes. Never allocate on Low/None/null stock.

**Verification:**
- [Read-only live schema diagnosis #39294740876](https://github.com/kys928/Hephaestus/actions/runs/39294740876) showed Low/null count before first fix.
- [Read-only live diagnosis #39295480432](https://github.com/kys928/Hephaestus/actions/runs/39295480432) showed all three regional quote fields null.
- [Final corrected CPU preflight #39295945568](https://github.com/kys928/Hephaestus/actions/runs/39295945568) completed successfully.
- [Final read-only live diagnostic #39295945596](https://github.com/kys928/Hephaestus/actions/runs/39295945596) completed successfully with `existing_gate_result: false`, no Pod.
- Neither variant consumed an S3 single-use pod-creation claim or incurred GPU spending.

The following marker resumes only the previously user-approved availability watch on this verified branch, without changing the earlier 60 matched pairs / 56 unique prompts, the frozen D step-96 adapter SHA256 `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`, GPU type (NVIDIA RTX PRO 6000 Blackwell Server Edition Secure Cloud 1x), EU-CZ-1 attached volume, one actual pod-create attempt, price <=$2.49/h, estimated total <=$0.75, training prohibition, certification prohibition, or scientific scoring.

If the live check still reports insufficient capacity, it must fail before any allocation and may be checked again hourly. Any actual pod allocation attempt consumes the approval and must never be repeated automatically. Keep PR #76 draft.
