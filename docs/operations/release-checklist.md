# Integration, documentation release and rollback

Use this checklist for production-code or documentation integration on `main`.

## Review before merge

- [ ] Confirm the base branch and exact head SHA; read the effective diff, not only commit titles.
- [ ] Keep one responsibility per reviewable PR. Preserve historical ancestry through **ordinary Git merge commits**; no squash, force-push or history rewrite of scientific source.
- [ ] Check that only intended public-safe code/docs/tests changed. Never import data/weights/credentials, immutable science bytes or obsolete paid workflows.
- [ ] Validate all relative documentation links and retained stable URLs.
- [ ] Run focused tests plus Python 3.11/3.12 checks. For research runtime changes, run pinned original CPU regression and activation/approval negative-path checks.
- [ ] Verify a clean install and supported local CLI/demo before claiming usability.
- [ ] Record evidence source SHA, reviewed PR number, original archive reference and current merge SHA.
- [ ] Obtain independent review for large security- or production-critical diffs. Record the actual reviewer and scope; same-agent adversarial review is not independent review or external scientific certification.

## Required repository governance

The [accepted #151 review](../reviews/151-post-integration-security-audit-20261031.md) records the owner's branch-protection waiver. `main` protection was not enabled by that acceptance or this remediation campaign. Maintain ordinary PR merges, exact-head checks and scientific ancestry; repository rulesets remain a separate maintainer choice. Do not reopen historical #151 acceptance over the waived setting.

Do not set required check names without verifying currently emitted check contexts; otherwise a misnamed gate can accidentally prevent every merge. Keep risky external-provider workflows separately and explicitly approved.

## After merge

- [ ] Verify `main` points to the expected two-parent merge commit; compare original research source against `main` to confirm ancestry.
- [ ] Verify current active Actions, docs nav, package import, CPU tests and restricted role dispatch behavior.
- [ ] Check that publication does not rewrite the scientific outcome or imply checkpoint promotion.
- [ ] Update the issue owning the actual work with exact evidence. #151 remains closed for its historical scope; #152/#102 require their genuine scientific/independent custody exit criteria.

## Recovery or rollback

Prefer a new small fix-forward PR when low-risk. For a bad merged integration, review the full dependent commit graph and use a new revert PR targeting the **merge commit** (e.g. `git revert -m 1 <merge-sha>`, only after confirming first-parent semantics). Test the resulting candidate, preserve Git history, and document which production capabilities are removed. Do **not** delete original R10 source commits, archive objects, checkpoint hashes or protected evidence to make a rollback look cleaner.

Historical integration record: [full R10 Git ancestry](../integration_notes/151-r10-full-history-and-runtime.md), [55-stack ledger](../integration_notes/151-historical-stacked-pr-ancestry.md) and [workflow retirement](../integration_notes/151-active-workflow-retirement.md).
