# Prometheus — original Granite D sample recovery (read-only)

This is a **historical, exposed eight-state development re-score**, not evaluation of the current frozen six-state production Evaluator. It supports issue #195 without prescribing a repair experiment or spending money on GPU training.

## Provenance and scope

- Original D run: `evaluator-semantic-v2-D-38945578442`; original source commit `ffa0458d5850f9d572a06d5ee6660fd63fc8acc6`.
- Previously downloaded and SHA-checked independent D-screen GitHub artifact: workflow `38947323158`, artifact `12066013104`, ZIP SHA-256 `25ff27dea3153f601876d3ffb6bf5628cba734596f89364e7911286a1e4173ff`.
- D's original screen is 64 previously exposed development cases (eight per scientific state), **not** a new holdout.
- SHA-256 identities of its two exact sample shards were recovered from the independent historical factor-probe artifact `12072909151`, workflow `38962439250`.
- Exact frozen development pack hash: `ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912`.
- Exact 64-case ordered screen hash: `98c3346fde66cc92d355bae019de811cb9c7614746073d1e4f93e6a069dbe11c`.

## Completed one-time verification

The CPU-only branch workflow run **41609821443** completed on source head
`29eeb7b385b0bf07592fe78b12dffb1c05a0fdd9`. It independently
checked out the original historical reference generator at the pinned source
commit, regenerated the original 64-case development reference order, and
verified its SHA-256 before using S3. Offline fake-S3 checks: **46 passed**.

Only then did the ephemeral runner use existing repository-held S3 credentials
to read the two exact, previously exposed D shard objects via GET. Byte lengths
and SHA-256 hashes matched their archival receipts. Strict raw-output re-score:

- **33/64 exact decisions**, matching the archived independent D summary.
- **0 per-case disagreements** with the original stored decision scores.
- Full per-state historical counts reproduced.
- Original raw generations and development case text were **not** published.
- No new model inference, optimizer updates, pod or paid GPU launched.

Source receipts: [GitHub run 41609821443](https://github.com/kys928/Hephaestus/actions/runs/41609821443),
aggregate artifact ID `13217803440`; verified aggregate report SHA-256
`7398eeed835422151678e4e1a79ee4c0d8ead4daf9dba61dd4333aa9acfbf953`.

**The temporary S3-authenticated workflow was deliberately removed from the
proposed merge tree after this one-time run.** The maintained
`tests/test_active_workflow_allowlist.py` forbids active workflows from loading
repository secrets. We preserve that guard without adding an exception.
The read-only audit script, offline regressions, original run receipts and
this documentation remain. Any future live S3 read requires its own
appropriately reviewed, scoped execution; none is scheduled by this change.

The code in `scripts/audit_granite_d_historical_readonly.py` uses the
ontology-scoped raw scorer merged in PR #196. Its safe output is aggregates
plus SHA receipts; it never accepts arbitrary S3 object keys as CLI input.

## What this does not establish

Hash agreement with historical source receipts proves which bytes were rescored. It does **not** independently certify model weights, tokenizer, semantic reference-label truth, data noncontamination, or the historical model's execution provenance. Development data remain exposed, and fresh independent certification is not claimed.

Do not treat an eight-state D score as a direct measure of the six-state certified original Granite Evaluator. Do not infer a causal training diagnosis from raw decision mistakes alone. The correct next step is to pass independently verified *observational* metrics to Hephaestus's Diagnosis and Planner, with model and ontology identities clearly bound. The system must choose its own proposed action under frozen policy; Sol must not inject a preferred intervention.

No paid GPU, new model inference, training, final holdout access, production activation or promotion is authorized by this evidence-recovery code.
