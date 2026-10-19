# RunPod regional stock-schema root cause and verified fix

Date: 2026-10-19. Read-only source evidence: [safe regional diagnostic #39294740876](https://github.com/kys928/Hephaestus/actions/runs/39294740876).

**Root cause:** `lowestPrice(input: {gpuCount: 1, secureCloud: true, dataCenterId: "EU-CZ-1"})` returned the exact expected Blackwell GPU ID, valid hourly price <= $2.49, `stockStatus: "Low"`, and `availableGpuCounts: null`. The previous parser incorrectly treated null as an invalid count-list schema and raised `ValueError`, hiding the real low-stock condition. The `availableGpuCounts` field is nullable in RunPod's published GraphQL LowestPrice schema.

**Narrow fix:** `inspect_response()` now recognizes nullable `availableGpuCounts` and relies on the already regional, single-GPU, Secure Cloud stock quote when counts are absent. If explicit counts exist, they must still contain `1` to allow a capacity signal. Stock `Low` or `None` always blocks paid provisioning, as does over-ceiling price, unknown schema, conflicting counts, or an API error. Whitelisted, secret-safe diagnostic error codes replace opaque `ValueError` logging.

**Verified:** Safe read-only live response after fix: [#39295040925](https://github.com/kys928/Hephaestus/actions/runs/39295040925) returned `existing_gate_result: false`, stock `Low`, `availableGpuCounts` null, with 23 checks passed. The full corrected CPU preflight [#39295299931](https://github.com/kys928/Hephaestus/actions/runs/39295299931) passed including the new null-count, pricing, and response-leakage tests.

**Execution unchanged:** Exact D step-96 checkpoint SHA256 `5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1`; 60 scientific pairs, 56 unique fixed-decoding prompts, exact RTX PRO 6000 Blackwell Server Edition secure-cloud 1x in EU-CZ-1, hourly GPU <= $2.49 and hard estimated total <= $0.75. No training, alternative GPU, budget increase, certification, or promotion.

This one marker restarts the user's previously authorized **single-use guarded regional capacity watch** on the corrected code. Insufficient regional stock must fail **before** pod request or S3 once-claim; rerun only the *failed preallocation stock check*. On any actual pod-create attempt, consume the one-use approval and do not automatically retry.
