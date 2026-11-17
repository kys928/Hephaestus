# Hephaestus research autonomy phase 2 — audited code and honest deployment gate

**Date:** 2026-11-17. **Base:** [draft PR #212](https://github.com/kys928/Hephaestus/pull/212). **Status:** further opt-in engineering; no live RunPod CUDA trial or merged production activation.

## What changed

### 1. Genuine frozen-role inference composition

The new `research.live_runtime` and `research.deploy_worker` compose:
- **unchanged certified** Ministral-3-3B Diagnostician, exact original revision and adapter hash;
- the original `RoleMasteryContract`, original `CertifiedRoleReasoner`, `PinnedTransformersRoleGenerator(device="cuda:0")` and SHA-verifying `VerifiedRoleAdapters`;
- GET-only RunPod S3 source retrieval through the same admitted historical catalog and v2 dataset coverage probe;
- persisted source observations, raw original role-inference receipts, three precommitted seeds per source and an explicitly non-certified result;
- the existing durable `SQLiteJobQueue` and `DurableWorker` rather than a new scheduler.

The CLI `hephaestus-research submit` queues natural objectives. `hephaestus-research serve-signed-cuda` is a separately deployed opt-in worker and fails closed without a complete externally signed operator grant, independently pinned key/catalog/model identities, real CUDA hardware and S3 source credentials. It does **not** create a RunPod Pod, sign its own grant, authorize training, change model weights, or run a model as a fallback.

A signed, single-study **Ed25519 compute grant** binds run/lineage, catalog SHA, frozen registry SHA, the diagnosis role, device, provider/region/SKU, exactly six inference calls, 384 output tokens per call, bounded GPU time, expiry, and negative training/code-edit/promotion authority. The new grant cannot re-use the original 45-operation training grant.

A single-use durable grant reservation is written **before** inference; it prevents a crashed or partially completed study from spending the same authorization again. Reservations are fenced through the existing host-local execution-owner mechanism, but the writer fence is **released** during expensive inference so the durable worker can heartbeat. A completed study reopens its immutable receipt on idempotent retry. Per-call expiry and elapsed-time checks are performed before and after inference, but a separate active watchdog is still required to terminate a hung model invocation at its absolute compute deadline.

**Not yet proven:** real GPU inference for this new research job; Pod/provisioning wiring; independent external scientific adjudication. No actual signed grant was available or silently synthesized.

### 2. Broader historical evidence discovery

`research.reviewed_discovery.ReviewedEvidenceDiscovery` accepts multiple independently hash-pinned, allowlisted source catalogs, each containing a real original/candidate family. Search is deterministic and lineage/family scoped; ambiguous matches stop with a typed error instead of choosing a convenient source or treating a random retrieval as a gold label.

This expands beyond one hardcoded source family, but it is **not unrestricted web/S3 search**. A new source must be added to an independently reviewed catalog before the crawler can measure its bytes. The existing `VerifiedSourceAcquirer` still rehashes actual S3 bytes and independently checks row counts.

### 3. Real governed next-experiment handoffs

The previous `StagedActionIntentService` rejected all next-run branch/restart/rollback actions. The optional `governed_next_experiments=True` path now allows **only** those three intents to proceed to the existing approval-checked `GovernedActionExecutor`. The original default remains blocked.

`GovernedNextExperiment` verifies the checkpoint's *retained external producer commitment* (not merely an on-disk hash), original lineage membership and presence of scoped external approval before any action mutation. The production loop applies the action **once**, derives the next run ID and baseline from the actual committed action and writes an immutable handoff. Branching uses the real created child lineage; restarting and rollback retain their source lineage.

The staged production cycle driver will use the exact new run identity only after reopening the immutable handoff and verifying the original approved action. This does not pretend a next training run exists if the real dependency factory cannot bind its new run/lineage/source inputs. The 45-step training spine is unchanged, and stage/data/recipe/eval-policy modifications remain blocked until real distinct handoff materializers exist.

### 4. One genuine small-coder defect attempt

Actual defect found: the v2 Diagnostician dataset probe maps missing `text=None` into string `"None"` through `str(row.get("text",""))`, leading to `text_rows=1` for a row with no text. A new, independently run test on the **unchanged production source** reproducibly failed in an offline Docker container.

A real local CPU Qwen2.5-Coder 1.5B inference generated a JSON edit with a recorded prompt/output hash, but changed the unrelated training-dynamics function. The unchanged test remained red and the patch was correctly **rejected**. See [run #44150247334](https://github.com/kys928/Hephaestus/actions/runs/44150247334). No repair or live source mutation was claimed.

The next controlled iteration optionally narrows source context to a **test-verified Python entrypoint** and uses an exact one-span proposal instead of requesting a whole-file rewrite. The model cannot choose the target file, tests, permissions, or GitHub merge. Model success will be reported **only** after real red-to-green on the same unchanged regression.

### What constitutes completion

- Independently reviewed phase-2 integration/authorization change; no automatic merge.
- A signed and externally approved RunPod study run using actual original Ministral model/adapter, verified S3 records, actual raw inference receipts, fixed seeds, and independent resource termination.
- Multiple accepted science source families and a verified source-selection policy beyond the historical two-dataset fixture.
- Evidence that the true production driver loads the new source/configuration on subsequent run handoffs; no cross-lineage or stale-input substitution.
- A real coder edit that turns the genuine missing-text regression red-to-green; if the model fails, retain the failure evidence and abstain instead of merging a plausible-looking patch.

The user's research objective should drive the loop, but ***scientific truth, spending permission and protected state must not be manufactured by a model***.
