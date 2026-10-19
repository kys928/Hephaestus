# Capacity-triggered retry — one matching Blackwell pod only

The user authorized **retrying the same D factor/decision matched-contrast experiment once when the originally required GPU becomes available**. Budget and science remain unchanged; this is not a request for a new ablation or automatic repeated paid attempts.

Frozen D checkpoint SHA256: `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`.

- Scientific panel: identical 60 one-visible-variable pairs (48 state-changing, 12 expected invariant).
- Exactly 56 deduplicated prompts, fixed greedy decoding and unchanged scoring.
- GPU exactly `NVIDIA RTX PRO 6000 Blackwell Server Edition`, Secure Cloud, 1 GPU, tied to network volume `cviwpryzao` in **EU-CZ-1**.
- Preflight live per-datacenter GPU GraphQL quote must report `High` or `Medium` and include 1-GPU count at $2.49/hour or lower. `Low`, `None`, unsupported schema, mismatched region/type/price are NOT triggers.
- GitHub Actions job ends with 75 on insufficient stock and can be safely rerun (no pod, no S3 claim). The *first* successful preflight passes through to precisely one guarded pod-creation request and atomic S3 once-claim; any later run cannot create another pod on that approval.
- $0.75 estimated GPU cost ceiling including reserved teardown, no $1.50 contingency, one GPU allocation attempt after stock becomes credible, verified pod deletion.
- Independent read-only rescoring on successful evaluation; preserve failure logs otherwise.
- No training, gate adjustments, certification, production merge, role dispatch or alternate GPU.

This branch is isolated; old no-capacity claim and original PR #75 remain untouched. The original authorization was blocked before pod allocation at [Actions #39269912043](https://github.com/kys928/Hephaestus/actions/runs/39269912043).

The capacity job's eventual first triggering push is **not** itself a paid job unless the exact regional signal passes. An hourly conditional monitor should **re-run only the original failed capacity job**, not create fresh approvals or independent executions. On successful completion or any post-allocation failure, disable the monitor and report result. Human or platform cancelation should not be treated as free-to-retry when pod creation might be uncertain.
