# User-approved same-volume RTX PRO 6000 Blackwell Workstation fallback

The user requested an RTX 6000 Blackwell GPU if possible, or any VRAM-sufficient alternative, after deleting the previous stalled A40. Server Edition Secure Cloud stock in EU-CZ-1 returned None during the first guarded workflow; Workstation Edition 96GB was independently observed in EU-CZ-1 at $2.59/hour Low stock. Source and results remain on the existing EU-CZ-1 volume cviwpryzao (locally mounted).

This fallback uses the **EXACT SAME approval_id and S3 one-shot claim key** as the pending Server Edition workflow. Atomic IfNoneMatch=* ensures one request can claim the right to create a pod, never both. The previously exhausted A40 claim is unrelated and may not be reused. Any successful pod-create attempt consumes all options; no repeat.

New pod must use <=$2.59/h and <=$1.80 estimated total GPU with teardown reserve; 10-minute no-S3-bootstrap automatic deletion, official RunPod CUDA12.8 image, persistent cache where available, independent full D step96 scientific rescore (56 unique deterministic prompts, 60 pairs). No training, certification, or production promotion.

CPU/shell preflight [#39987525922](https://github.com/kys928/Hephaestus/actions/runs/39987525922) passed 67 tests and no-GPU dry run. The marker on this commit triggers a *capacity-gated* workflow; if Workstation stock is unavailable it exits before S3 claim, with no GPU charge.
