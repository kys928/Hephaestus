# Corrected Blackwell one-shot D contrast inference trigger

The previously pinned user-approved workflow run 39986151557 was **cancelled in its free stock-poll stage** before the paid inference step, S3 claim, or RunPod create call.

The user already authorized **one actual paid retry** after verified A40 deletion. This marker retries only the *preallocation* GPU stock poll with the repaired, immutable scientific runner; it does not authorize a second pod after any create attempt.

Frozen model: Granite 3.3 8B revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b + D step-96 adapter SHA256 5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1. No training. Data: 56 fixed unique prompts, 60 one-variable counterfactual pairs, greedy 112-token cap.

Fixes since cancelled run: runner's approval guard accepts the already authorized $1.80 estimated ceiling; the CUDA 12.8 image uses a new isolated venv path rather than potentially loading CUDA 13 venv packages; the S3 network volume is correctly mounted in EU-CZ-1. The full CPU suite passed 68 tests in workflow 39986577363. Exact region/GPU stock is ephemeral, so the paid job checks it again.

Limit: 1x RTX PRO 6000 Blackwell Server Edition, Secure Cloud EU-CZ-1, <= $2.49/hour, $1.80 estimated total GPU, one pod create call, 600-second missing-bootstrap cutoff, S3 evidence and independent science verification required. All unsuccessful stock polls exit without paid provisioning.

No merges, deployment, model changes, or additional GPU fallback allocations allowed by this marker.
