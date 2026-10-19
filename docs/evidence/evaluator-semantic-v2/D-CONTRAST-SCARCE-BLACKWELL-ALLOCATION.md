# One user-requested Blackwell allocation attempt on scarce regional stock

On 19 October 2026 the user stated that an RTX PRO 6000 Blackwell GPU was available and directed Hephaestus to secure it. We verified RunPod's published guidance: `Low` means very few GPUs remain available, not that the region is necessarily empty. The former regional preflight required Medium/High as an additional operational caution; that condition prevented a user-requested attempt even though RunPod advertised scarce positive stock.

**Scope of this single launch:** try exactly one matching `NVIDIA RTX PRO 6000 Blackwell Server Edition` Secure Cloud GPU in `EU-CZ-1`, paired to existing network volume `cviwpryzao`, on the same D step-96 checkpoint (SHA-256 `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`), unchanged 60 paired scientific contrasts and 56 deduplicated deterministic prompts. The same full model evidence and independent rescore remain mandatory. No training, promotion, certification, extra pod, new approval ID, or budget increase.

The revised stock-check only allows provider High/Medium/**Low** if the exact one-GPU regional Secure Cloud quote has an unambiguous GPU identity, finite hourly price <= **$2.49**, and GPU counts that are either nullable under the region-scoped gpuCount=1 query or explicitly include one. Null/None stock, empty price, bad counts or unknown schema still fail closed. A Low signal **does not guarantee** actual provisioning. The launcher still enforces one *atomic S3 approval claim*, one create-pod attempt, and an estimated **$0.75 total cap** with reserved teardown and verified pod deletion.

The prelaunch CPU suite passed in [Actions run #39299468828](https://github.com/kys928/Hephaestus/actions/runs/39299468828), and the old low-stock attempt #39296135549 did not enter the pod-creation step or consume this new claim. The hourly monitor is paused to avoid overlapping attempts.

This commit marker launches this user-directed one-shot low-stock attempt. Do not replay after any creation request or uncertain post-claim outcome.
