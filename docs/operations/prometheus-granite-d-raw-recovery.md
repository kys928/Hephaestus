# Prometheus — original Granite D sample recovery (read-only)

This is a **historical, exposed eight-state development re-score**, not evaluation of the current frozen six-state production Evaluator. It supports issue #195 without prescribing a repair experiment or spending money on GPU training.

## Provenance and scope

- Original D run: `evaluator-semantic-v2-D-38945578442`; original source commit `ffa0458d5850f9d572a06d5ee6660fd63fc8acc6`.
- Previously downloaded and SHA-checked independent D-screen GitHub artifact: workflow `38947323158`, artifact `12066013104`, ZIP SHA-256 `25ff27dea3153f601876d3ffb6bf5628cba734596f89364e7911286a1e4173ff`.
- D's original screen is 64 previously exposed development cases (eight per scientific state), **not** a new holdout.
- SHA-256 identities of its two exact sample shards were recovered from the independent historical factor-probe artifact `12072909151`, workflow `38962439250`.
- Exact frozen development pack hash: `ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912`.
- Exact 64-case ordered screen hash: `98c3346fde66cc92d355bae019de811cb9c7614746073d1e4f93e6a069dbe11c`.

## Workflow

The CPU-only GitHub Actions workflow `.github/workflows/prometheus-granite-d-raw-readonly.yml` checks out the current source and the historical immutable source commit separately. With **no S3 credentials yet injected**, it regenerates the original 64 exposed reference IDs/expected labels and checks the frozen development and screen hashes.

Next it runs offline fake-S3 regressions. Only after these steps pass does it inject the repository's existing S3 read credentials into a single step that uses **GET only** for the two exact, hardcoded D shard keys. It checks the original byte lengths and SHA-256 digests against independent archived receipts before parsing raw projected model outputs.

The new `scripts/audit_granite_d_historical_readonly.py` independently scores complete raw JSON decisions using the ontology-scoped scorer merged in PR #196; it compares strict per-case judgments against the historical stored Boolean scores and compares recorded per-state counts against the archived assessment.

Outputs are **aggregates and SHA receipts only**. The action must not publish raw generations, prompts, private reference rows, model weights, credentials or protected final assessment data.

## What this does not establish

Hash agreement with historical source receipts proves which bytes were rescored. It does **not** independently certify model weights, tokenizer, semantic reference-label truth, data noncontamination, or the historical model's execution provenance. Development data remain exposed, and fresh independent certification is not claimed.

Do not treat an eight-state D score as a direct measure of the six-state certified original Granite Evaluator. Do not infer a causal training diagnosis from raw decision mistakes alone. The correct next step is to pass independently verified *observational* metrics to Hephaestus's Diagnosis and Planner, with model and ontology identities clearly bound. The system must choose its own proposed action under frozen policy; Sol must not inject a preferred intervention.

No paid GPU, new model inference, training, final holdout access, production activation or promotion is authorized by this workflow.
