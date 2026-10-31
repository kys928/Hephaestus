# Issue #151 — full verified R10 lineage integration review (2026-10-31)

## What is and is not being merged

This branch is a **genuine Git two-parent merge** (not a squash, rebase or cherry-picked-only history) of:

1. Existing `main` at `e09a6876646ca4d27180096ad940156e3f0d3ccb` (including evidence-contract PR #153 and report PR #150).
2. The **exact original successful R10 executed source** `d6bcec7b61bcd3b285a4bcf4cdd50036440b8edf` (and its entire research, training and release lineage).

The hand-built merge commit is `8d2af83ce792e46d7ecafe6407ee7f7b7191cb10`. It has those two exact Git commit parents, thereby making original R10 commits traversable from the eventual integrated main. It uses the R10 tree as the reviewed source-of-truth for research code and tests, with explicit reconciliation below. The commit must **not** be described as a review of each of its 606 commits; review the effective tree and critical safety deltas independently.

## Tree reconciliation

At the preserved starting tips, `main` had 527 tracked files and R10 had 897 tracked files. Comparing both complete (nontruncated) recursive Git trees:

- Seven main-only paths were retained verbatim: original role-runtime release workflow/manifest/notes, first R10 scientific report, #151 evidence-contract integration note/CI, and release-archiving script. The existing `main` `runpod-live-smoke.yml` workflow was also preserved instead of accepting the divergent R10 version.
- **109 R10-only, historical, one-off `.github/workflows/*.yml` files are EXCLUDED** from the new active tree, without deleting them from Git history. This includes historical paid r1–r10 launch/prepare/preflight YAML, one-time provider diagnosis, S3 forensic reads, manual approval-patching helpers and old narrowly-scoped GPU campaign actions. Neither new default CI nor a pull request should re-enable them.
- Where 32 same-path blobs differed between the two tips, this integration candidate starts with **the later successful R10 source version** for application/test/runtime/config files, subject to integrated negative-path and existing-main tests. These include original deterministic control, data preprocessing, GPU/provider implementation and original CUDA runtime. There is no global wildcard style reformatting.
- The original R10 `src/hephaestus/production/`, CUDA training, deterministic services, licensed source review, scientific evaluation, strict five-role source-checked inference, runpod provider/reconciliation, unsigned authorization request and independent cold verifier are now in the branch for actual review.
- R10 root-level `evidence/reviews/*.json` are **research summary records only**, not private source bytes; audit their public-safe scope independently before mainline merge.
- `pyproject.toml` in the candidate is the R10 variant with pinned optional `peft-training`, `role-inference`, `independent-certification`, and `dataset-preparation` extras; no heavy model weights are vendored into Git.
- Historical `scripts/` and `configs/research/` contain source-specific runner references; these are **not a supported default production CLI** and should be labeled/archived in subsequent focused maintenance. Do not auto-run past campaigns or re-spend old grants.

### Security and scientific constraints

- `automatic_role_dispatch_enabled=false` remains untouched.
- The governed eight-phase control spine and external-signed GPU funding remain mandatory.
- The original R10 archive, original checked source commit, role bytes and checkpoint hashes remain immutable and hosted outside Git; this branch imports application source, not a new scientific run.
- The independent R10 evidence index `docs/evidence/original-cuda45-r10/README.md` and the original `role-runtime-verified-v1` GitHub release/manifest remain unchanged.
- The merger is **not external historical certification**, arbitrary production deployment or checkpoint promotion.
- No new or reused RunPod Pods are created by integrating Git code; removed historical workflows stay readable through the exact old original R10 commit.

## CI / go/no-go

- [ ] Verify the R10 original source and existing main commit are **both** exact ancestors of the candidate; ensure Git didn't flatten or rewrite history.
- [ ] Verify absence of historical paid workflow definitions and active model dispatch; verify all existing main archival references remain.
- [ ] Compile all imported modules and test contract/authorization negatives.
- [ ] Reproduce full pinned CPU suite on a clean GitHub runner without GPU allocation or private billing credentials.
- [ ] Check core install, optional extra dependency consistency, CLI package install, and python 3.11–3.12 compatibility where available.
- [ ] Audit differences in main's 32 conflicted same-path blobs, provider no-implicit-spend policy, immutable admission and action boundaries before merging.
- [ ] Run PR-independent security review and ensure no generated artifacts/secrets embedded in the Git tree.
- [ ] Normal **merge commit** only into main; NEVER squash, rebase or force-push the frozen scientific lineage.
- [ ] Record actual main merge SHA and fresh clean-checkout CI. Refresh active GitHub branch/workflow routing. Mark #151 complete **only** when integrated baseline and historical PR reconciliation have been reviewed.

## After integration

The goal is a maintainable research-engineering service layout with one `src/hephaestus/` package and explicit `production/`, `training/`, `evaluation/`, `data/`, `storage/`, `policy/` and `schemas/` responsibilities. Only active CPU CI and operator workflows should remain in the live `.github/workflows/`. Research/paid experiments remain immutable and recoverable via Git parents, source SHA and indexed evidence, **not silently erased**.

A new behavioral-improvement campaign is covered by [issue #152](https://github.com/kys928/Hephaestus/issues/152), while [issue #102](https://github.com/kys928/Hephaestus/issues/102) retains external historical holdout/custody/promotion tasks. Neither should be misreported as complete due to this source integration.
