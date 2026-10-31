# Issue #151 — post-integration security and operability review

**Review date:** 2026-10-31. **Audited effective mainline:** `ad4e133b063220e11d3916320433ca737a295058` through #160, #161 and follow-up #162. **Reviewer model:** same connected engineering assistant used for earlier integration; this is a *fresh, evidence-based code-path review and negative-path test*, **not** an organizationally independent third-party penetration test or external scientific certificate.

## What was actually inspected

| Boundary | Concrete files | Findings / remaining limits |
| --- | --- | --- |
| Eight-phase coordination and action journal | `production/loop.py`, `staged_driver.py`, `actions.py`, `state.py` | Driver requires completed 45-operation trace; Judge action has an append-only prepare/apply recovery journal. Generic injected driver must itself be idempotent; the real staged driver resumes durable completed operations. |
| Approval, action registry, promotion | `policy/action_registry.py`, `action_authorization.py`, `approval_policy.py` | Unknown and forbidden actions fail closed; protected promotion checks actual operator decision, lineage, hashes, stage pack and scientific report. **Previously missed auto-continuation membership/status gaps corrected in #160/#162.** |
| Data/optimization/evaluation | `production/staged_finalization.py`, `staged_training.py`, `training/tier_a_cuda_target.py`, `evaluation/tier_a_cuda_target.py` | Actual training gated, finalized references checked, CUDA execution requires device and resource grant, each optimizer restart receipt verifies checkpoint/source/step order. This is original Tier-A source-specific, not general from-scratch pretrained model proof. |
| Original five frozen model roles | `production/model_roles.py`, `original_cuda_identity.py`, `llm/certified_runtime.py`, `providers/models/active_evaluator.py` | Role-backed stage opt-in, pinned registry unchanged, original frozen inference remains advisory; swapping active Evaluator requires separate exact-scope promotion/activation receipt and health checks. |
| Paid compute and mutable provider | `production/signed_cuda_grants.py`, `cloud_budget.py`, `providers/runpod/execution.py` | Original scientific GPU grants verify external Ed25519 authority per operation; RunPod adapter has durable create/reconcile intent, ambiguous allocation fail-closed and termination receipts. **Boundary limit:** low-level RunPod adapter is a credentialed provider library, not itself a universal policy engine; only approved call sites may invoke it. |
| Immutable readback, recovery and lineage | `storage/verified.py`, `production/recovery.py`, `state/lineage_store.py` | SHA-256 checked JSON read, bounded infrastructure retries only before known scientific progress, authenticated lineage updates, no implicit checkpoint promotion. |
| CI and evidence | `.github/workflows/`, `docs/evidence/original-cuda45-r10/README.md` | Five maintained CPU workflows; original protected evidence index and Git ancestry unchanged; CPU suites validate logic but are not standalone CUDA receipts. |

## Independent code-path findings

1. **F1 — confirmed, fixed by [#160](https://github.com/kys928/Hephaestus/pull/160):** auto-allowed `continue_from_checkpoint` could install foreign or tampered caller-provided `best_checkpoint_ref` without lineage membership/manifest verification. Added strict source checks and tests; original signed promotion still independently gated.
2. **F2 — confirmed, [#160](https://github.com/kys928/Hephaestus/pull/160)/[#162](https://github.com/kys928/Hephaestus/pull/162):** `poisoned`, `archived`, `deprecated` or `blocked` lineage could have been moved to active by auto continuation/rejection after a terminal action. Fix blocks that path and tests real abort-then-revive attempt.
3. **F3 — targeted crash-recovery test in this review PR:** `ProductionLoopRunner` writes a durable Judge action before its final loop-state snapshot. It is essential to prove that the *actual* `StagedProductionCycleDriver` resumes the completed 45-operation state rather than reinvoking training/replaying the Judge mutation after a crash at this boundary. The added injected-failure test demonstrates the intended exactly-once result **for the real staged driver**, not for arbitrary third-party `ProductionCycleDriver` objects.
4. **Not claimed covered:** exhaustive line-by-line audit of every one of the >600 original R10 commits, live authenticated provider allocation, secrets penetration test, independent original private-source or Granite evaluation, or future arbitrary model/dataset safety. These belong to separate issue #102/#152 and further hardening.

## Acceptance and owner waiver

The owner explicitly deprioritized GitHub branch-protection configuration in the discussion on 2026-10-31. This review does **not** represent it as enabled; GitHub currently reports main branch unprotected. This is an **owner-accepted governance risk**, not a passing protection gate. Any administrator can implement rulesets separately. Existing rollback instructions are in `docs/operations/release-checklist.md`.

Close the *source-integration* scope of #151 only after #160, #162, the injected post-action crash test and all merged mainline CPU CI pass and the effective Git tree still preserves R10 source/evidence identities. Do not say an externally independent security auditor signed off, and do not use this closure to assert #102 scientific certificate or #152 behavioral improvement.

## Residual tracked work

- Inactive scaffold/prompt TODO cleanup: [#159](https://github.com/kys928/Hephaestus/issues/159).
- Genuine externally signed private-source and historical Evaluator custody: [#102](https://github.com/kys928/Hephaestus/issues/102).
- Prospective improvement evidence: [#152](https://github.com/kys928/Hephaestus/issues/152).
- Non-core injected `ProductionCycleDriver` implementations must supply their own stable operation-level idempotency; this scoped audit only tests and accepts the repository's actual staged production driver.
