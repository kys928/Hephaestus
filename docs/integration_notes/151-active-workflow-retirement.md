# Issue #151 — remove historical single-campaign Actions from active main

## Decision and reproducible inventory

Review performed against exact integrated main commit `1df26eab17168d19ff9336f6691d4f8c9df69fc2` (Git tree `276350877afaa5dfab17db82474018679bc708a3`). Of **18** active `.github/workflows/*.yml` definitions, **5** are maintained CPU contract/integration checks. **13** were one-off research, launch, archive, dataset-prep, S3 inspection, or historical operator workflows created for earlier experiments.

These 13 definitions are **removed only from the current working Git tree**, not erased, squashed or rewritten. Their original Git blob identities and the complete R10 and older parent commit history remain reachable through existing ordinary merge commits (#153–#156). The deletion commit itself remains visible in the new ordinary-merge ancestry. This is workflow deactivation, not loss of scientific evidence.

No GPU, S3 API, paid RunPod call, model download, approval grant, user-facing role dispatch, checkpoint promotion or protected scientific source mutation is performed by this change.

## Kept: maintained, CPU-only checks

| Active workflow | Reason |
| --- | --- |
| `autonomous-loop-integration.yml` | Original deterministic phase/approval regression |
| `certified-role-model-stack.yml` | Pinned certified-role config/contract regression; no GPU allocation |
| `integration-151-foundation-no-gpu.yml` | Read-only R10 integration foundations and provenance guards |
| `integration-151-r10-runtime-cpu.yml` | R10 service contracts and CPU integration checks |
| `production-autonomy-integration.yml` | Python 3.11/3.12 full composed CPU integration |

## Retired: one-off historical workflows (recoverable from Git)

| Retired definition | Why it no longer belongs in the active default branch |
| --- | --- |
| `archive-role-runtime-verified-v1.yml` | Completed one-time evidence archival/release task with potentially write-capable token |
| `bootstrap-first-scientific-experiment.yml` | Old project-specific dataset, diagnosis, bootstrap or S3 provider workflow; future operators need a fresh reviewed entry point |
| `first-autonomous-dataset-discovery.yml` | Old project-specific dataset, diagnosis, bootstrap or S3 provider workflow; future operators need a fresh reviewed entry point |
| `first-bounded-scientific-training.yml` | Historical GPU/RunPod-capable experiment or live smoke; must not reuse old launch authorization |
| `first-post-failure-diagnosis.yml` | Old project-specific dataset, diagnosis, bootstrap or S3 provider workflow; future operators need a fresh reviewed entry point |
| `first-selected-dataset-preparation.yml` | Old project-specific dataset, diagnosis, bootstrap or S3 provider workflow; future operators need a fresh reviewed entry point |
| `first-semantic-evaluation.yml` | Historical GPU/RunPod-capable experiment or live smoke; must not reuse old launch authorization |
| `first-targeted-post-failure-diagnostics.yml` | Old project-specific dataset, diagnosis, bootstrap or S3 provider workflow; future operators need a fresh reviewed entry point |
| `freeze-second-controlled-evidence.yml` | Historical GPU/RunPod-capable experiment or live smoke; must not reuse old launch authorization |
| `runpod-live-smoke.yml` | Historical GPU/RunPod-capable experiment or live smoke; must not reuse old launch authorization |
| `runpod-volume-inventory.yml` | Old project-specific dataset, diagnosis, bootstrap or S3 provider workflow; future operators need a fresh reviewed entry point |
| `second-controlled-experiment.yml` | Historical GPU/RunPod-capable experiment or live smoke; must not reuse old launch authorization |
| `verify-first-scientific-bootstrap-volume.yml` | Old project-specific dataset, diagnosis, bootstrap or S3 provider workflow; future operators need a fresh reviewed entry point |

Most historical workflows were already restricted to a historical feature branch for their `push` event. That restriction did **not** make an old `workflow_dispatch` operator entrypoint desirable in default `main`. Removal makes the maintained set explicit and reduces accidental legacy launch routes without pretending these files were triggering GPU on every main commit.

## Regression guard

`tests/test_active_workflow_allowlist.py` checks the five exact current workflow definitions and refuses active GitHub Actions files that wire repository secrets. A new privileged workflow would require a deliberate reviewed change to the allowlist/guard, a new approval and updated security documentation. CI never serves as the actual compute-authorization authority.

## Remaining security and scientific boundaries

- No use of prior RunPod funding/grant IDs. All new allocations require a new scoped, independently verified approval through the deterministic runtime.
- Original R10 executed source commit: `d6bcec7b61bcd3b285a4bcf4cdd50036440b8edf`; independent real 45/5/96 evidence and immutable checkpoint/archive hashes are unchanged.
- Historical #99/#112/#140 original branch heads are preserved as parents by #156; their PRs were closed as superseded, not merged as separate scientific successes.
- [Issue #102](https://github.com/kys928/Hephaestus/issues/102) external custody, contamination review and certification remain open.
- [Issue #151](https://github.com/kys928/Hephaestus/issues/151) needs final post-merge independent review and branch protection; a successful Python suite is not an external audit.
