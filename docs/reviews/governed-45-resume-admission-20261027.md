# Governed 45-operation campaign — exact-state activation audit (2026-10-27)

## Result: production campaign still BLOCKED; no 12/45 claim

This PR extends [PR #110](https://github.com/kys928/Hephaestus/pull/110) to address the original campaign **inside its real persisted governed orchestrator** instead of adding another independent optimizer/generation sidecar. It has **not** resumed the original workflow because the actual authoritative workflow SQLite database, scoped acquisition approval, licensed cleared training source, and a runtime with five separately loaded frozen CUDA masters are **not available in this GitHub Actions job**.

### Implemented

- `src/hephaestus/production/governed_campaign_admission.py` requires a pre-existing `StagedWorkflowState`, correct run/lineage/workflow/stage identities, identical canonical 45-step order, contiguous completed prefix, and **SHA-256-authenticated original `substep_result` plus each required scientific decision/source record**. Rejects missing historical state, forged or mismatched status/identities, missing original science records, gaps and completion claims without all 45 receipts.
- `resume_real_governed_campaign` refuses missing scoped-approval provider, nonreal worker bindings, any of the five absent or wrong `ModelBackedStage` role placements, CPU-only shared research scorer, or changed certified role stack. Requires the real `CertifiedRoleReasoner` / `PinnedTransformersRoleGenerator` on `cuda:0`, the original 45 operation binding inventory, and the **same** persisted campaign identity. It delegates any actual continuation to the original `GovernedStagedOrchestrator.resume`; the native source clearance and approval gates still decide whether step 12 executes. **This is not a role-execution or scientific quality certificate.**
- Read-only `scripts/inspect_governed_45_campaign.py` accepts only an existing SQLite and artifact directory plus explicit workflow identities and reports the *real* waiting operation and its stored request. The tool cannot report 11 completed without the real receipts. This invocation was **not run against the original campaign database**.
- Integrity unit tests use **clearly labelled test-only synthetic staged state** and prove refusal of missing approvals, forged receipts, foreign identities, missing original science records and a CPU research model masquerading as five production CUDA role models. They are NOT original-campaign operation completions.

### Executed evidence

At executable commit `b0706d7c3216a8a752e3c6506326c01ee3e23a7c`:

- [Genuine-state admission CI #40570013450](https://github.com/kys928/Hephaestus/actions/runs/40570013450): **34 passing tests** (suite includes prior native staged orchestration and approval tests).
- [Live metadata-only S3 readiness CI #40570015178](https://github.com/kys928/Hephaestus/actions/runs/40570015178): **11 passing tests**, then real S3 HEAD and bounded LIST calls; succeeded.
- [General CPU production integration CI #40570019985](https://github.com/kys928/Hephaestus/actions/runs/40570019985): success at identical source commit.
- [Run evidence artifact #12765122035](https://github.com/kys928/Hephaestus/actions/runs/40570015178/artifacts/12765122035): independently downloaded; ZIP SHA256 `3f771e5631340b16bfac91dd9a57ec1ee84306f975c4e12f6ca0ab34dc4f8203`; raw report SHA256 `425962defd6369c590562b0762f4dc7f96e87c2d5a22f9e44795a71d2220bbc1` matches the separate file custody record.

S3 HEAD returned **all 5 registered frozen role adapter objects present at their exact expected registered byte lengths**: Controller, Diagnosis, Planner, Evaluator, Judge. Original frozen tokenizer object also present. This is proof of *remote key existence and matching byte count only*, **not** SHA verification of actual contents, successful frozen model loading, or full-role inference.

A capped `ListObjectsV2` metadata inspection saw **3,000 keys**, was **truncated**, and found **zero candidate persisted state database objects within those scanned prefixes**. This **does not prove the authoritative SQLite/state doesn't exist elsewhere**. Source review already establishes the repository's historical operator selection approval lists only `dataset_selection_approval`, **not `dataset_acquisition`**, and therefore cannot unlock operation 12. This audit neither reads a corpus nor issues a grant. No S3 writes, model inference, GPU launch or protected holdout read.

## Exact remaining blockers before a genuine 45-operation run

1. **Provide/reference the actual original persisted workflow state and content-addressed operation-receipt store** from the true earlier 11-stage campaign. A PR comment, test fixture or independent development runner is not interchangeable with those bytes.
2. **Obtain a scientifically eligible source and an authentic, scoped data-acquisition approval decision** for the selected exact immutable candidate. Current SAF/WikiText screening shows unresolved rights/privacy/evaluation-leakage and/or tokenizer limits; original `dataset_selection_approval` is insufficient.
3. **Provision the real five frozen CUDA role models in an authorized GPU runtime** with SHA-checked frozen adapters; check the full original composition and limited resource budget. The five registered S3 HEAD checks are useful storage readiness, not execution or GPU spending approval.
4. Resume through the unchanged original 45-step `GovernedStagedOrchestrator`; verify **one complete 45-operation SHA-backed trace and genuine checkpoint/generation/evaluation receipts**. Until then record **11/45 from separately documented historical claim**, not 11 newly verified by this inspector.

**Final verdict:** activation entrypoint built, guarded, and CPU tested; production not activated. This is separate from independent CPU PEFT experiments, all of which remain noncertified and not counted as 45-operation progress.
