# Frozen evidence: first semantic evaluation

This directory preserves the exact UTF-8 launcher and independent-verification JSON bytes emitted by the first real `semantic_behavior_v1` baseline-versus-trained evaluation.

- evaluation run: `first-semantic-evaluation-001-33869352751`
- GitHub Actions run: `33869352751`
- GitHub artifact ID: `9935299093`
- execution commit: `5f4fe4ebf413355b8842555dcfa09c4feef404ed`
- RunPod Pod: `l9l8jdfqxy9fpq`
- frozen eval pack: `semantic_behavior@1.0.0`
- eval pack hash: `ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad`
- trained checkpoint manifest: `sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3`
- comparison outcome: `regressed`
- Judge exit action: `reject_checkpoint`

`launcher.json` and `verification.json` were re-downloaded from the original GitHub Actions artifact and SHA-256 checked before being committed. They are intentionally not reformatted.

The original ZIP is not duplicated in Git because it is only a transport container. Its exact SHA-256 is retained in `SHA256SUMS` so the original workflow artifact can still be matched byte-for-byte.

The Judge decision was recorded but not applied. This evidence proves the generation/evaluation/Judge chain worked; it does not mutate lineage, promote or delete a checkpoint, or authorize further training.
