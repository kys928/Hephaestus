# Hephaestus collaborative autonomy protocol

**Status:** Proposed standing project working practice; implementation through draft PR #214, based on user-engineer direction of 17 November 2026. This is a project record, not a claim that the architecture already enforces every rule below.

## Research relationship

**User and assistant define scientific objectives, constraints and acceptance gates. Hephaestus conducts admitted research.** The assistant's responsibility is to inspect actual repository, runtime and evidence, identify the missing autonomous capability, implement it into Hephaestus, prove it against real services and source-pinned CI, and preserve review and spending boundaries. Do **not** manually micromanage experiment-by-experiment actions when the failure reveals a missing reusable Hephaestus capability.

- Start from existing verified source and persistent state; recover before repeating expensive research.
- Prefer single integrated native control-plane services and existing durable queues/storage over new parallel schedulers.
- Clearly separate **code implementation**, **verified CPU/software behavior**, **actual frozen-role inference**, **independently evaluated scientific improvement**, **external action authorization** and **production activation**.
- For substantial engineering, inspect -> implement -> run exact-source tests -> inspect real logs -> diagnose failures -> repair -> rerun -> document immutable evidence -> open review-only PR. A successful unit test never substitutes for real service or scientific validation.
- Each claim must have exact commit/run, source hashes, input identity, actual outputs, and explicit scope/limitations. No synthetic completed science, fake 45-stage receipts, invented GPU inference, or success invented from model output.
- Use offline/no-secret isolated build/execution for model patches; fixed tests and independent measurements cannot be rewritten by the proposing model.
- Do not auto-merge, auto-promote, bypass human approvals, or spend GPU money without independently pinned authorization and shutdown controls.

## First-pass correctness / error containment

The goal is **not** to pretend a model can be mathematically guaranteed correct on its first generation. The goal is **zero unverified outputs crossing a critical phase boundary**. A first proposed patch may be wrong; it must be detected, quarantined and repaired or abstained **inside that phase**, before any downstream stage consumes it.

1. On entry, verify typed schema, existing run/lineage IDs, source SHA/custody, parameter/recipe identity, upstream proofs and exact action approval.
2. Before inference or edits, reproduce the actual fault on a clean reviewed original source; distinguish test collection/infrastructure failure from the intended semantic regression.
3. Freeze old source, fixed evaluator, policy and test suite. Produce a candidate in a separate disposable worker with no secrets or unrestricted filesystem/GitHub access.
4. Independently run regression, full impacted tests, invariants and relevant security/contamination checks; require **red on unchanged source -> green on candidate**. No passing candidate without independent verification; no green test if baseline was an unrelated import error.
5. Fail closed and **do not advance control spine** on ambiguous / missing data, stale source, unverified model identity, failed sandbox, inconclusive scientific labels, inadequate review or inconsistent evidence. The retry budget is bounded, with each attempt and failure recorded, and the same original acceptance gates applied every time.
6. Only the validated, content-addressed, policy-admitted output may become an input to the next phase. Persist an atomic handoff and its dependency checks; use idempotency and replay-verification across process crashes.
7. Report final state to the user including what was done, what was rejected, why and which independent approvals remain. Leave reviewable PRs in draft until independently accepted.

## What a Coder is

Existing certified roles are controller (Qwen3-4B-Instruct), Diagnostician (Ministral-3-3B-Reasoning), Planner (Ministral-3-8B-Reasoning), Evaluator (Granite-3.3-8B-Instruct) and Judge (Phi-4-mini-instruct). Their existing narrow seven-key role LoRA contracts do **not** authorize code changes or certify them as code models.

If Hephaestus is to author patches itself, add a separately registered, bounded **Coder** capability. The **Planner/Ministral-3-8B-Reasoning base model** is the first *in-stack candidate for code-fix reasoning* but must be compared under pinned reproducible code-repair tests against the existing Controller/Qwen3 4B, Evaluator/Granite 8B and dedicated code models. Prefer a separate code-capability adapter and contract, not overloading the certified Planner LoRA or granting the Planner a direct editor. The Coder proposes text only; independent deterministic services alone choose target file, execute code in isolation, check invariants and submit review PRs.

A local Ollama-hosted Qwen2.5-Coder model used in GitHub Actions was **downloaded on a temporary isolated CPU runner solely for a repair experiment**, not a sixth resident Hephaestus specialist or its own training output. An initial genuine source defect was rejected for 1.5B, 3B and first-pass 7B proposals; a two-attempt 7B feedback loop produced an independently verified red-to-green one-test result in run #44152786543. It is **one bounded demonstration**, not first-pass reliability or general self-repair certification.

## Durable evidence / status

- [Draft PR #212](https://github.com/kys928/Hephaestus/pull/212): research intake, reviewed S3 source verification, evaluation-only job and initial isolated repair.
- [Draft PR #214](https://github.com/kys928/Hephaestus/pull/214): signed real-CUDA composition, multiple reviewed catalogs, branch/restart/rollback handoffs and bounded coder feedback.
- [Full v2 CI #44153969248](https://github.com/kys928/Hephaestus/actions/runs/44153969248): 1,375 passed, 90 skipped, 0 failed; no GPU claim.
- [Actual 7B coder two-attempt proof #44152786543](https://github.com/kys928/Hephaestus/actions/runs/44152786543): first failed patch, second sandbox red-to-green, no live source mutation.
- [End-to-end exit gate #213](https://github.com/kys928/Hephaestus/issues/213): still open; do not equate code-unit proof with autonomous production scientific execution.

**Project rule:** Hephaestus owns the repeatable research loop. Human/assistant review covers scientific objective, independent trust roots, external spending, irreversible actions and the eventual activation decision.
