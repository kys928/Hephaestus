# Campaign readiness at PR #115

Reviewed baseline: `f7ec89f166615319107b13edbc4633dc69a9f120` (2026-10-28). This is an engineering readiness assessment, not an execution receipt or scientific clearance.

## Verified dependency chain

The exact published heads of PR #101, #103–#111 and #113–#115 are ancestors of the reviewed baseline. PR #112's latest head `bafe3f153dfbc3708b959d6e9b224cce815d3f09` is not an ancestor: its late optional-import regression fix was carried forward in PR #114 by `7411e8f`. Its only changed file, `tests/test_historical_recovery_and_one_source.py`, is identical at PR #112 and this baseline. This local ancestry check does not merge or independently approve any PR.

## Existing durable state is necessary, but not a runnable campaign

`create_fresh_campaign_workspace` exclusively creates a new directory, authentic SQLite origin state and SHA-addressed bootstrap. It correctly records zero completed operations and no inherited approvals. `verify_campaign_workspace` reopens original state and verifies the same campaign identity and artifact bytes. No existing script calls the initializer to launch a production campaign. Choosing a genuinely persistent mount, retaining that exact SQLite database and artifact root, and supplying the native orchestrator's real dependencies remain launch prerequisites. A pytest temporary directory or an uploaded count report is not production custody.

The historical 11-operation temporary CPU smoke cannot be imported or resumed. New work starts at operation 1. Test counts do not advance campaign counts.

## CPU-first execution contract requires explicit engineering

The production composer (`src/hephaestus/production/staged_composition.py`) and resume admission (`governed_campaign_admission.py`) require a `CertifiedRoleReasoner` with `PinnedTransformersRoleGenerator(device="cuda:0")`. The generator itself rejects CPU and requires exactly one visible CUDA GPU. This is already documented in `docs/integration_notes/real-staged-science-core-20261025.md`.

Consequently, the current code does **not** support the requested genuine CPU 45/45 campaign through the production entrypoint. The separate CPU research role bridge and pretrained 135M CPU advisories are not the five original frozen masters. They cannot satisfy production admission merely because they perform real inference. A deliberate, separately reviewed CPU scientific execution contract is required before CPU 45/45 can become an executable prerequisite for later CUDA replay. Keep the certified CUDA identity checks intact; do not relabel CPU substitutes as the frozen production stack or skip the CPU prerequisite to allocate GPUs.

## Required scientific launch inputs

- A new workflow/run/lineage/stage identity and a persistent private mounted state/artifact directory.
- An immutable `staged-diagnosis-input.v1` matching that run, actual hash-bound diagnostic source evidence, and a real pinned baseline run artifact.
- Independently accepted recipe and baseline allowlists; actual training lifecycle and generation/evaluation services.
- A scientifically admitted source: exact selected file bytes, frozen tokenizer, exact processing profile and reviewed source-row/window commitment.
- An accepted full-corpus `DatasetSourceReview`, separate screening scope, rights evidence and all four audit reports (`pii`, `exact_near_duplicate`, `semantic_contamination`, `frozen_evaluation_leakage`).
- Original-text training, development, diagnostic and holdout custody manifests and the separately authorized protected comparisons. A source-only scope is insufficient.
- The authentic selection and acquisition grants for the native staged request. User compute authorization does not mint those records or supply scientific audit evidence.
- For later CUDA replay: the five exact model revisions and SHA-verified role adapters in `configs/models/hephaestus_role_model_stack_v1.json`, matching frozen runtime profiles, and cache/volume readback before paid inference.

The registry supplies exact model revisions and adapter hashes; it does not prove that those objects are cached on an allocated machine. WikiText remains HOLD until independent rights/privacy/contamination/custody evidence is supplied. An exact materializer removes a preprocessing defect; it is not that evidence.

## Runpod access path

Repository workflows already inject compute and S3 secrets from GitHub Secrets, with non-secret placement variables. `runpod-live-smoke.yml` names the existing Hephaestus contract: network volume `cviwpryzao`, `EU-CZ-1`, S3 endpoint `https://s3api-eu-cz-1.runpod.io/`, control API `https://api.runpod.io/v2`. These checked-in expectations are not a live account/balance/cache verification. Reuse the credential-bearing Actions path instead of copying secrets into commits or logs. Compute authorization is present in the current user request; source admission and CPU execution prerequisites remain separate.

## Validation performed

`PYTHONPATH=src python -m pytest -q tests/test_durable_campaign.py tests/test_governed_campaign_admission.py tests/test_staged_production_composition.py tests/test_certified_role_stack.py -rs`

Result: **19 passed**. These check actual local SQLite/hash custody, admission boundaries, composition rejection paths and registry integrity. They execute no training, inference, protected-reference audit or paid allocation and establish no completed scientific operations.
