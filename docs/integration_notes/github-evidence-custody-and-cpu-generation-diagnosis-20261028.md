# GitHub evidence custody and generated-answer failure diagnosis (research draft)

Date: 2026-10-28. This document is an *implementation plan and forensic observation*, not a new scientific certificate.

## Original evidence inspected

These are authentic original ZIPs downloaded from GitHub Actions run [#40687705730](https://github.com/kys928/Hephaestus/actions/runs/40687705730), not newly generated answers:
- CPU seed 101 artifact 12816440131; original ZIP SHA-256 `e095d6881cbe2089922ee71690eef63d25967513fc0b24ec78374714801552a0`.
- CPU seed 202 artifact 12815168903; original ZIP SHA-256 `8ab116b7c04e98cb40e4753c0e487f648e41aa91200d0616ee82f5667b208a40`.
- CPU seed 303 artifact 12814983569; original ZIP SHA-256 `fe6410e0800e51b9edcb7f0a4a8bf8e39b366392480edb7d745ee1a44eef05dc`.

The read-only `scripts/audit_cpu_generation_format.py` independently verifies the ZIP member SHA manifest and original SHA-addressed source objects, then reports aggregate formatting categories without publishing original prompts or text.

**Measured failure:** in each campaign all 24/24 baseline and 24/24 candidate generated strings began with `\`\`\`python` code fences: 72/72 baseline, 72/72 candidate. JSON validity and exact task correctness stayed 0/24 in each run. Mean teacher-forced held-out target NLL decreased by approximately 0.132 across seeds. That reduction is predictive adaptation, not useful JSON generation or evidence of mathematical reasoning.

**Code observations:** the original `CpuProspectiveEvaluator.generate` prepends plain `\\nAnswer: `, does not use the instruct model's chat template, and generates at most 16 tokens. `CpuExactTokenTrainer` trains on the first 8 records of the 96 training examples using a single example per AdamW step. These are hypotheses for controlled testing; do not call them proven causes until ablated.

## Controlled diagnosis before any GPU

Preserve original raw receipt archives unchanged. Precommit new ablation protocol, frozen sample identities, metrics, model/tokenizer weights, optimizer settings and cost limits *before* running new trials.

Order of experiments (one primary changed variable per comparison):
1. **Inference format only:** baseline/candidate at original 16-token limit, plus separate precommitted 64-token arm. Test whether JSON can finish; do not silently replace the original 16-token benchmark.
2. **Wrapper only:** raw `Answer:` against the exact pinned instruct tokenizer chat template; isolate any improvement from training changes. Use an explicitly new evaluation protocol and source-domain formatting contract where needed.
3. **Target exposure only:** compare current 8 sequential single-example updates with balanced full-family coverage and more steps, under a separately approved budget. Track tokens, tokens/sec, losses and per-family correctness; do not claim a full epoch without one.
4. **Output control only:** controlled JSON response prefix or grammar-constrained decoding, clearly distinguished from *unconstrained model learning*. Report validity and semantics independently.
5. **Replicate a winning single-variable configuration:** at least three CPU seeds and an untouched independent final test. Report JSON validity, exact answer accuracy, regressions, NLL, and uncertainty. **A win in formatting alone does not demonstrate reasoning gains.**

Do not repeatedly inspect, retune, and claim blind success on the same frozen holdout. Use a separate development suite for hypothesis selection and a genuinely isolated final suite for confirmatory comparison.

The original zero/24 results and all past prompts remain untouched. An ablation that changes chat formatting or generation length is a *new experiment*, not a silent replay of the original frozen protocol.

## Use GitHub as the evidence registry

**Git repository:** only small, schema-validated manifests, run/commit IDs, SHA-256 indexes, source/license/provenance digests, baseline-versus-candidate summaries, release URLs, certification status, and a complete chain of scientific decisions. Never put growing checkpoint binaries or raw SQLite append histories in Git commits.

**Immutable GitHub Release:** after explicit publication eligibility, store the exact closed original CPU/GPU evidence archive(s), checkpoint bundle(s), SQLite snapshot, and independent external attestation *if one actually exists*. GitHub supports immutable release assets and attestations when the repository enables immutable releases. Require a dedicated tag per scientific run, immutable release enabled and verified, SHA verification after upload, and no tag recycling. The release is a final frozen record and not a live writable database. Keep asset sizes below GitHub release limits.

**Live durable state:** continue to use a private persistent object volume for active SQLite/WAL, model weights, optimizer states, and protected corpus text. Never run a resumable worker directly out of GitHub Releases or claim a copied ZIP is resumable without path/identity remapping and an explicit recovery test.

**Governed publisher:** provide the model with a narrow `propose_evidence_publication` action, not unrestricted repository write access. A deterministic policy approves *only* the exact SHA-addressed, public-safe artifact class and destination repository/tag. Public release rejects raw protected holdouts, PII, secrets, embargoed data, uncertain licenses, unsigned role identities or unapproved model weights. The publisher uses a dedicated short-lived least-privilege GitHub Actions token; no model chooses arbitrary assets, broad GitHub API actions or release credentials. Release upload is initially disabled until release immutability, asset classification, retention, and recovery verification are configured.

**External certification:** a GitHub-generated release attestation proves integrity/origin of the published asset, **not independent correctness of scientific claims**. External reviewer identity, independently scoped original train/development/holdout inspection, signed custody declaration, original Granite comparisons and human review are separate requirements of issue #102. Self-issued signatures, re-created historical manifests, or a green GPU CI workflow cannot satisfy them.

## GPU replay / #102 gates

- Before paid GPU: demonstrate meaningful *unconstrained* CPU output correctness above baseline and acceptable regression behavior with frozen evaluation identities and real checkpoints.
- Freeze intervention, code, full data/license manifests, tokenizer, original five CUDA model+adapter SHA identities, and the 45-stage workflow contract.
- Perform an **actual single** 45-operation GPU run of the original scientific service graph with **all five original frozen CUDA roles** and their signed adapters; check per-step SHA custody, approval gates, real inference/training, Judge disposition and replay.
- Separately satisfy independent evaluation custody, historical leakage assessment, signed reviewer attestation, Granite comparisons, and human review. Model promotion remains blocked if any acceptance condition fails.
- Publish only verified public-safe findings to a new immutable GitHub release and attach the reference to issue #102; close the issue **only when all original gates are demonstrably satisfied**.

Original #102: https://github.com/kys928/Hephaestus/issues/102 . This draft does not run a GPU, grant new paid authorization, sign a certificate, merge branches or modify the original frozen role identities.
