# Provider-backed durable Hephaestus scientific artifacts

**2026-10-28 — draft governed storage policy, no external upload/credential changes.**

GitHub remains the system of record for reviewable, source-controlled evidence indices, verified original SHA-256 values, exact workflow/commit identity, status and external references. Once proof-of-publication and privacy review succeeds, an immutable GitHub Release can freeze an eligible *public-safe* research-summary evidence bundle. Original protected holdouts/PII and unapproved weights must never go into a public release.

For versioned binaries and trainable datasets, **Hugging Face is the preferred model+dataset backend**: separate model and dataset repos, private where needed, commit-pinned revision, SHA-256-verified individual files, correct rights and model/dataset cards. Hub Xet is designed for large ML binaries, unlike ordinary code Git.

**Kaggle datasets are a second dataset-only backend.** Identify owner/slug, exact positive version, original file digests and provenance; default new datasets to private, and require an explicit approved publication action to make public. Do not use an unqualified latest dataset version or assume Kaggle's file history is equivalent to scientific attestation.

A checked-in `external-evidence-pointer.v1` record must name provider, exact repository, revision/tag/version, artifact classification, visibility, reviewed bytes digest and review commitment. The verifier denies public protected material, mutable HF revision aliases, mismatched Kaggle versions and generic public model/weight storage on GitHub Releases. This is *reference schema*, not a publisher or verification that an external server actually contains bytes.

**Future publishing workflow:** (1) validate source/license/privacy classification and permitted publication scope, (2) compute complete archive digest, (3) perform narrow upload with least-privilege short-lived provider credentials in a protected GitHub Actions environment, (4) read back remote exact revision bytes and verify SHA-256 against local original, (5) commit only a pointer and post-verification receipt, (6) never overwrite/recycle scientific release tags. If external authentication, host quota, rights, or immutability checks fail, stay `awaiting_upload` and continue local operation. Never forge external publication receipts.

Active SQLite/WAL, mutable optimizer checkpoints and interrupted runs still require a **separate persistent private writable workspace**; neither GitHub Releases nor a Hugging Face/Kaggle immutable version is a live transaction database.

Hugging Face Hub docs: https://huggingface.co/docs/hub/storage-limits ; https://huggingface.co/docs/hub/repositories . Kaggle CLI dataset versioning: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md . GitHub immutable releases: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases .

Issue #102 unchanged: original frozen CUDA full-spine and external independent scientific custody/certification cannot be replaced with a repository hash or platform-supplied upload checksum.
