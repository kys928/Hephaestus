# User-requested Blackwell restart — one use only, 2026-10-23

Approved user instruction: terminate the stalled A40 and redo the frozen D evaluator inference, first trying RTX PRO 6000 Blackwell Server Edition and otherwise an available memory-sufficient GPU.

The failed A40 pod `bnvazbgs2acrp4` was verified deleted via RunPod REST GET 404 on 2026-10-23 at 15:58 UTC. It is not to be resumed; its old S3 one-shot claim is consumed. No parallel pods may run.

New attempt: **NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB x1 Secure Cloud EU-CZ-1**, the region of existing mounted volume `cviwpryzao`. Exact positive Low stock and $2.49/hr were checked in [read-only stock/balance #39985643302](https://github.com/kys928/Hephaestus/actions/runs/39985643302). Account credit was at least $2.00 (threshold check only). GPU estimated total spend <=**$1.80**, inclusive of 120s teardown reserve, under one new distinct S3 approval/claim; a create-pod attempt consumes the authorization forever.

Safety: official CUDA12.8 RunPod PyTorch image, 100GB ephemeral disk, persistent volume locally mounted; early container-bootstrap milestones are S3-visible, hard delete if no first bootstrap within 10 minutes. No retraining/model change, frozen IBM Granite 3.3 8B + D step96 LoRA SHA256 5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1, 56 unique deterministic prompts / 60 matched contrast pairs. Success only with full saved outputs, independent read-only scientific score and verified pod deletion.

CPU-only validation [#39985643310](https://github.com/kys928/Hephaestus/actions/runs/39985643310): 67 tests green and Bash syntax correct; no allocation. The final workflow must recheck stock and price before claiming S3, and attempt exactly one pod creation. Once a paid create was attempted, never automatically rerun the job even on failure.

This single marker commit triggers the already user-authorized run, not a second permission request.
