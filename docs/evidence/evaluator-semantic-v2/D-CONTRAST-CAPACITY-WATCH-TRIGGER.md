# Initiate capacity-gated one-use Blackwell D retry

This is the single watched experiment requested by the user. The commit marker starts *only* the preflight run. If GPU stock is Low/None for a single Secure Cloud RTX PRO 6000 Blackwell Server Edition at EU-CZ-1, the job deliberately ends in a retriable failure **without requesting a pod**. Scheduled monitoring will re-run the failed job; no GitHub pushes or additional approval records are needed. Once stock reports Medium/High at <=$2.49/hour, the workflow makes exactly one GPU pod creation attempt with a separate immutable $0.75 approval and the original frozen D scientific protocol.

A RunPod no-capacity response after that attempt consumes the new once-claim and ends monitoring; never silently request a second pod. No training, alternate GPUs, model changes, promotion, or relaxed scientific criteria.
