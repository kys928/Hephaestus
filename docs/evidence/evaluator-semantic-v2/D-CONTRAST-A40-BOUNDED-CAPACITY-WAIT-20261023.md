# Re-arm one previously approved A40 attempt after capacity-only preflight failures

The previous guarded workflow [#39974466173](https://github.com/kys928/Hephaestus/actions/runs/39974466173) attempted no pod: both run attempts failed on free per-region Secure Cloud stock preflight with no S3 approval claim and no GPU allocation.

Scientific model, 56 deterministic inputs, 60 one-variable matched contrast pairs, one-pod approval ID and estimated $1.90 total ceiling are unchanged. Compute uses a 48GB NVIDIA A40 Secure Cloud in EU-SE-1 at <=$0.59/hour; model and generated evidence use external S3 HTTPS against original EU-CZ-1 network volume cviwpryzao. No cross-region mount, training, certification or promotion.

New bounded guarded workflow changes only **when** it rechecks provider inventory: validate CPU scientific code first (64 tests passed in [#39974850543](https://github.com/kys928/Hephaestus/actions/runs/39974850543)), then poll exact positive A40 stock for <=15 minutes at 30-second intervals without claiming S3 or creating a pod. The paid launcher makes its own immediate exact quote check prior to the once-only S3 claim and RunPod create.

If stock remains unavailable, the workflow fails free. Only if run logs positively establish no allocation-step execution may GitHub failed-job checks be repeated. The very first create-pod request consumes the user-authorized one-shot attempt permanently; even a provider-side failure must not automatically retry. Successful completion still demands independent read-only rescore of all saved outputs and verified teardown.

This launch marker uses the same explicit 2026-10-23 user authorization, not a second monetary approval or pod allocation.
