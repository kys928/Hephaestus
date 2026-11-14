# Prometheus Unbound — supervised Global GPU diagnostic launch

This implementation is stacked on PR #205 and PR #201, and must not be interpreted as autonomous Granite repair, training, final independent evaluation, certification or production promotion.

## Source identity

The exact role registry is `configs/models/hephaestus_role_model_stack_v1.json` and the original five certified role identities are unchanged. The original parent Granite development aggregate is restored as **unchanged artifact bytes** from GitHub Actions artifact `13233709387` (file `granite-original-parent-dev-aggregate.json`, SHA-256 `2d1ffa37a9e14939c5c0e9c3365a5fe0a4220a37b502678a1f72f5def366378f`). Historical eight-state D aggregate is restored from artifact `13217803440` (file `granite-d-readonly-rescore.json`, SHA-256 `7398eeed835422151678e4e1a79ee4c0d8ead4daf9dba61dd4333aa9acfbf953`). Both are *exposed development observations*, and ontologies differ.

The controller refuses changed bytes, missing/expired GitHub artifacts or a non-exact source commit. It downloads each single-entry ZIP through authenticated GitHub Actions, verifies SHA-256, and stages exact reports with immutable readback into original RunPod regional S3. The GPU worker reads and rehashes the reports again before model inference.

## Physical storage and execution

- `cmw51pjsk000007l9a0mv1jga` — Global Volume; **read-only immutable model/adapter input** on the GPU Pod, mounted at `/workspace-global`. All certified archive hashes and sizes are independently checked before PEFT load.
- `cviwpryzao` — Czech original regional **S3 API bucket**, accessed through `https://s3api-eu-cz-1.runpod.io/`; private source staging, durable inference results and independent watchdog control plane. It is **not** mounted on cross-region GPU Pods.
- `/tmp/prometheus-global-campaign` and `/tmp/prometheus-hf-cache` — Pod-local POSIX filesystem; raw model receipts and scratch. No live SQLite or mutable optimizer/checkpoint state is stored on Global.
- One `NVIDIA RTX PRO 6000 Blackwell Server Edition`, RunPod Secure Cloud, EU data centers only. Fresh live provider confirms actual mount, GPU SKU, data center, image and price.

## Evidence lifecycle

1. An **independently scheduled off-Pod watchdog** publishes one immutable `watchdog-ready.json` and a <=90-second S3 heartbeat. Its hard deadline is external to the launcher.
2. The launcher checks the original account's current prepaid balance and already running spend. It refuses autopay ambiguity and creates an immutable one-shot S3 launch intent **only if** a fresh watchdog is alive.
3. Only then may an exact hardware Pod be created, and actual provider attributes are independently read back. Missing stock or incorrect mount/price fails closed.
4. Pod fetches the SHA-verified source archive, refuses unexpected tar members, installs pinned scientific code and uses the real frozen `CertifiedRoleReasoner` pipeline.
5. Scientific worker runs the real Diagnostician and Planner first. Its original research logic permits `collect_more_evidence`, `repair_evaluation`, or `stop` only; it does not authorize training, new adapter selection, protected evaluation or promotion. If `collect_more_evidence` is admitted, the bounded original Granite public-development measurement is performed.
6. GPU-side envelope retains raw inference receipts, model proposals, decisions, failed attempts, environment execution metadata and logs in per-file SHA-verified **private S3**, publishing `result/manifest.json` LAST. A separate `status/worker-exit.json` references that manifest's SHA.
7. The off-Pod controller independently rehashes **every** receipt in S3, checks the actual scientific-decision and observed work, and verifies exact Pod teardown. The separate watchdog also terminates exact matching Pod after worker exit or at its deadline.

A failed/expired launch never counts as science. Interrupted GPU setup may lack full forensic receipts if it fails before the envelope starts; the watchdog still enforces termination. A S3 outage likewise refuses success rather than promoting a partially uploaded result.

## Execution isolation and secrets

Keep all credential-bearing, one-time workflow definitions on an **isolated operations branch**. Do NOT merge them into `main` or weaken `tests/test_active_workflow_allowlist.py`. The credential-free reusable supervision modules may be reviewed and merged through normal PRs. GitHub Actions secrets `RUNPOD_API_KEY`, `RUNPOD_S3_ACCESS_KEY_ID`, `RUNPOD_S3_SECRET_ACCESS_KEY`, `HF_TOKEN` and one-time `GH_TOKEN` are injected at runtime only. No secret values, raw model contents or protected source text should appear in public Actions logs.

## Acceptance status

- [x] Verified original source ZIP artifacts available in GitHub Actions.
- [x] Global-only role archive adapter (PR #205).
- [x] New Global-only CUDA Pod launcher, external watchdog, local-science envelope and SHA-verified S3 result manifest (this PR).
- [x] Offline negative path tests authored; final GitHub CI verdict pending independent observation.
- [ ] Live credentials, current RunPod account balance and Global Volume access independently checked on this execution.
- [ ] Authenticated separated watchdog live and verified prior to Pod allocation.
- [ ] Original source-pinned real CUDA Diagnostician and Planner inference receipts obtained.
- [ ] Every private S3 receipt read back and independent exact Pod deletion confirmed.
- [ ] Evidence-dependent model decision established through counterfactual controls.
- [ ] Scientific intervention and genuine model improvement independently proven.

A single successful GPU role execution supports **model-backed decision capability**, not causal model-repair improvement. That stronger claim requires counterfactual evidence tests, fixed/heuristic controls and independent measured candidate comparisons.
