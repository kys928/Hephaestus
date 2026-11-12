# Hephaestus S3 → Global Volume migration reconciliation (2026-11-12)

**Status:** evidence transfer and independent readback in progress. The original Czech regional volume is **retained**; **no deletion, training-state activation, or model certification is authorized** by this record.

## Sources of truth and immutable targets

- **Code, tests and governance:** `kys928/Hephaestus` GitHub repository; do not treat a Pod's local repository clone as authoritative.
- **Original scientific research:** RunPod regional Network Volume `cviwpryzao` in `EU-CZ-1`, S3 API endpoint `https://s3api-eu-cz-1.runpod.io/`. S3 credentials remain in GitHub Actions Secrets.
- **Read-mostly export destination:** Global Volume `cmw51pjsk000007l9a0mv1jga`, dedicated restore prefix `/workspace/hephaestus/hf-restores/export-129d690ef9ae24ef141fbb0c`. The existing R9 archive and unrelated Global Volume objects are not overwritten.
- **Private curated export:** `DavidDulovic/hephaestus-trained-models`, `DavidDulovic/hephaestus-training-data`, and `DavidDulovic/hephaestus-scientific-evidence`. The original 8,986-file HF export did **not** include every original source object, and an HF token was not present among the checked GitHub Secrets. Do not claim private HF manifest parity from S3 parity.

## Original S3 inventory evidence

GitHub Actions [source audit #43256848721](https://github.com/kys928/Hephaestus/actions/runs/43256848721) completed successfully. Full per-object inventory is stored privately at:

`s3://cviwpryzao/hephaestus/recovery-audit/source-inventory-43256848721.json`

SHA-256 of inventory JSON bytes:
`046dde9ebe9a712665081794586d502de5722f7f815a61567377414b680f5fb8`.

This inventory was classified by **path and extension only**; category membership is not a legal, immutability, model-activation, or contamination clearance.

| Source category | Objects | Bytes | Storage decision |
|---|---:|---:|---|
| Model and adapter candidates | 2,800 | 20,022,723,520 | Match hashes and lineage against the already verified 869-item curated Global model export before additional publication; no blind duplicate copying |
| Dataset candidates | 248 | 215,499,371 | Compare to curated four-file Global dataset set; isolate trainable-data custody, review rights and held-out contamination |
| Research evidence / unclassified | 6,858 | 551,720,496 | S3 evidence restorer independently classifies 8,224 candidates; its **distinct** receipt and independent readback are the acceptance gates |
| Archives requiring inspection | 78 | 4,186,775,055 | Original R9 TAR has a known Global marker; do not assume other archives duplicate it or lack private data |
| Mutable or operational state | 636 | 1,838,287,758 | Keep on *regional* network storage with suitable SQLite/WAL semantics and single-writer discipline; do not load directly from Global |
| Protected evaluation / custody | 2,271 | 31,636,498 | Retain access-controlled at the original regional source until independent rights and authorization review |
| Third-party upstream model cache | 36,180 | 18,995,970,957 | Excluded from curated export; treat as re-fetchable cache only if pinned revisions and hashes are reproducible |
| Miscellaneous unclassified | 157 | 23,624 | Requires purpose inspection; no automatic certification |

Audited root counts: `hephaestus/scientific/` 12,738 objects (25,924,826,156 bytes); `hephaestus/research/` 88 objects (921,815,731 bytes); `Hephaestus/artifacts/` 222 objects (24,435 bytes); `hephaestus-cache/` 36,180 objects (18,995,970,957 bytes). Additional [supplemental audit #43258381025](https://github.com/kys928/Hephaestus/actions/runs/43258381025) found `.cache/` 238 objects (128,307,475 bytes), `codex-worktrees/` 1,537 objects (3,437,901 bytes), and `hephaestus-tmp/` 11 objects (573,342 bytes). These are cache, development-worktree, and temporary source roots; retain them on original storage for now, but **do not promote them as portable immutable research artifacts** without individual selection and review.

## Restoration evidence and boundaries

- Existing **869 trained model files** and **four curated training dataset files** were independently SHA-256 verified on Global Volume by the original HF-to-Global v3 script, per the user-supplied run log.
- HF v3 selected **8,113 evidence files**. The Czech S3 source-classifier currently selects **8,224** candidates (~607 MB). These scopes differ because the original HF exporter and new S3 classifier use different rules; equality must be proven by exact path + bytes + SHA-256, **not by the similar item count**.
- The current bounded source-S3 recovery is [Actions run #43248214667](https://github.com/kys928/Hephaestus/actions/runs/43248214667), controlled off-Pod with prepaid balance checks and Pod teardown. The worker performs source SHA-256 → Global write → Global readback and a final full Global rehash before writing a private `S3_EVIDENCE_RECEIPT.json`. The receipt is a **source-parity report**, not a Hugging Face 8,986-file export certificate.
- A dependent [independent receipt verification job #43257715054](https://github.com/kys928/Hephaestus/actions/runs/43257715054) waits for successful source recovery *and Pod teardown* before provisioning a bounded read-only CPU verifier. This job reads the S3-anchored receipt digest, rehashes all Global files from a separate mount, and writes a private completion observation to original S3.
- Neither worker automatically promotes models, touches frozen protected evaluation data, writes to live SQLite state, or deletes the source volume.

## Operational topology for future Hephaestus Pods

For an active CPU/GPU training Pod **in EU-CZ-1**, mount the original *regional* volume at a distinct state path (e.g. `/workspace-state`) and the *Global* volume at a separate immutable-assets path (e.g. `/workspace-global`). Pin and verify the actual API-returned volume IDs and mount paths before any scientific run. For Pod requests, the adapter uses:

```json
{
  "mounts": {
    "network": [{"volumeId": "cviwpryzao", "path": "/workspace-state"}],
    "global": [{"volumeId": "cmw51pjsk000007l9a0mv1jga", "path": "/workspace-global"}]
  }
}
```

Clone the pinned GitHub revision onto the **container disk** (not the object-backed Global Volume). Keep Python environments, compilation, temporary download buffers and software locks on the container disk. Persist active SQLite/WAL, optimizer state, mutable checkpoints and protected comparison custody exclusively on the regional mount with single-writer discipline. Reference sealed model/data/evidence bundles on the Global mount **read-only** and require receipt/lineage admission before loading them.

**Cross-region limitation:** the Czech regional volume cannot be mounted in a different data center. A cross-region compute worker may access the Global immutable library but must use an independently admitted state backend or a verified snapshot/restore process before resuming mutable state. An R9 archive alone is not an authorized operational recovery.

## Acceptance checklist

1. [ ] Source recovery Actions run #43248214667 succeeds, reports `COMPLETE` and the exact Pod is deleted.
2. [ ] S3 evidence receipt is present and its SHA-256 matches the digest anchored in private S3 progress metadata.
3. [ ] Independent verification run #43257715054 rehashes all receipt-listed Global evidence and reports `INDEPENDENT_VERIFIER_COMPLETE=true`, then deletes its Pod.
4. [ ] Reconcile 8,224 S3 evidence files vs 8,113 curated HF files by exact inventory membership and byte hashes; S3 receipt alone cannot prove HF manifest completion.
5. [ ] Enumerate remaining cache/worktree/tmp roots, confirm the model/dataset/archives overlap or outstanding items, and keep ambiguous artifacts at source.
6. [ ] Test a fresh Pod: pinned GitHub code, both correct mounts, offline verified model load, read-only data/evidence resolution and regional SQLite integrity/WAL custody.
7. [ ] Only after separate policy review, decide whether any original CZ storage can be retired. No automatic deletion.

**Budget and billing:** source recovery caps Pod unit rate at $0.60/hr, compute estimate at $3, and runtime at four hours; dependent independent verification has a separate bounded Pod and cleanup. Global Volume request, storage and data-transfer fees are not covered by the CPU-only price ceiling.
