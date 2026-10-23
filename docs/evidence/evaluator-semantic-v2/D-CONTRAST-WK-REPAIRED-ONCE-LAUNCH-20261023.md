# Repaired Blackwell Workstation D contrast — one authorized new paid attempt

User said “there is one. take it colleague!!!!” at ~18:36 Europe/Amsterdam 2026-10-23, explicitly requesting allocation of a 96GB RTX PRO 6000 WK shown in RunPod EU-CZ-1 for $2.59/hr. Earlier WK attempt GitHub #39987687857 was created and deleted at 18:33, before this new request, after a 15-second bootstrap failure; prior one-use S3 approval is consumed and never reused.

A previously pending Server Edition workflow #39986768639 unexpectedly allocated an obsolete second pod `e6p97v9ixa3nk5`, and was cancelled. A dedicated exact-ID cleanup workflow #39989354238 verified its DELETE (204) and subsequent missing status (404). **Only after verified deletion and CPU repair validation may this new attempt launch.**

The new claim ID is `heph-D-contrasts-WK-EUCZ1-20261023-BOOTFIX-USER-ONE`. Exactly 1 create request, 1x RTX PRO 6000 Blackwell Workstation Edition EU-CZ-1 Secure Cloud, fresh exact positive quote <= $2.59/h, existing EU-CZ-1 volume mount; $1.80 estimated GPU hard cap with 120-second teardown reserve, 10-minute missing-bootstrap cutoff, verified deletion mandatory. No automatic follow-on pod.

Root cause of prior bootstrap failure remains not positively established because the previous local progress file was overwritten by `bootstrap_failed`; however the prior S3 state had `venv_ready:true` with no earlier history and the old launcher reused a CUDA13 venv with new CUDA12.8 container. Remediation: clean CUDA12.8-specific venv, append-only stage S3 receipts plus persisted ERR handling. A stale runner internal 0.75 guard is now aligned with approved 1.80 and tested. CPU validation including complete protocol, shell syntax, dry-run, no allocation: **68 passed** in GitHub #39988832123.

Frozen scientific scope: Granite 3.3 8B Instruct base revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b and D step96 LoRA SHA256 5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1; 56 unique immutable prompts / 60 one-scientific-variable counterfactual pairs, greedy max112 tokens. Inference-only. Independent CPU/S3 re-score required; no retraining, certification, production promotion or merge.

This document's commit is the ONLY paid one-shot trigger on the new repair branch. If stock unavailable fail with no pod, no claim or charge. If pod created, never replay automatically.
