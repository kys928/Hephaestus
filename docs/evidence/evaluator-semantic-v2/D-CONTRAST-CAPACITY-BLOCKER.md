# D counterfactual inference — capacity-blocked preallocation attempt

**Status:** no GPU allocated; no model generations; controlled experiment scientifically **not executed**.

The user authorized one separately bounded D step-96 counterfactual inference experiment on 19 October 2026. Experiment run: [GitHub Actions #39269912043](https://github.com/kys928/Hephaestus/actions/runs/39269912043).

## Scientifically frozen design

- 60 independently authored counterfactual pairs (48 state-changing, 12 oracle-invariant)
- 120 pair-side appearances represented by **56 unique deterministic prompts** after deduplication
- Every pair changes exactly one visible evidence field, with a fixed nonauthoritative throughput distractor where present
- Exactly one existing D step-96 checkpoint, SHA-256 `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`
- Fixed greedy decoding; exact raw state JSON and separately labelled exploratory wrapper parsing
- Strict budget: one pod, GPU hourly price at most $2.49, estimated total GPU ceiling $0.75
- No training, selection, certification, production action, or scientific-gate modifications

[Pre-launch CPU gate #39269781006](https://github.com/kys928/Hephaestus/actions/runs/39269781006): **31 tests passed**, followed by a dry run confirming **56 unique prompts** and exactly the intended inference runner.

## What happened

Preallocation catalog quote verified GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`, at **$2.49/hour**, stock status `Low`.

The sole pod creation request returned **HTTP 500: `create pod: There are no instances currently available`**. The infrastructure helper recorded `INTERFACE_REPAIR_V2_CAPACITY_JSON {"attempt": 1, "status": "unavailable"}`, and the GitHub Actions job failed as capacity-blocked. There was **no pod ID** to run inference on or delete. The independent-evaluation job was correctly skipped. No paid model inference occurred and estimated metered GPU runtime was **$0** (provider billing not independently audited).

The S3 atomic once-claim was written before pod creation, intentionally preventing accidental repetition. The checked-in approval has now been changed to `capacity_blocked_no_pod_allocated`, and the launcher refuses further paid execution. No automatic allocation retry or alternate hardware choice was made.

## Scientific conclusion

**No D model responses exist for this experiment.** CPU oracle validation does not demonstrate that the Evaluator can produce correct transitions, and prior D scores cannot substitute for this controlled 60-pair result.

The exact source, input manifest, model fingerprint, scoring code, one-run approval and failure evidence are preserved. A future retry requires a separate authorization decision and a fresh single-use claim, with the same scientific constraints and checkpoint identity. Reusing the prior S3 claim or silently increasing budget would be inappropriate.

No scientific improvement, regression or certification claim can be made from this capacity-blocked attempt.
