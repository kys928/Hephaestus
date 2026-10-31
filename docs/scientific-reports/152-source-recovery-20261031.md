# #152 original-source recovery — 2026-10-31

This is byte-level source recovery and metadata verification, not a new model
improvement result or external scientific certificate.

## Directly recovered and verified

Downloaded the original [CPU repair Actions artifact](https://github.com/kys928/Hephaestus/actions/runs/40746973738),
artifact12842406174, at original run source
`bec0a33db45277f35cfe47c3328ae78b0a58e6a3`. Its five-member ZIP is 1,722,899 bytes.

| Object | Verified SHA-256 |
| --- | --- |
| Original ZIP | `41351555907d6804038e4a9a066df5832a7ac85983b86ec41658eb1067e3bf0e` |
| Adapter weights | `a782d0ca344559659893a64d8bc1e8a90f597cf9cf0bbef06b433c555c1da195` |
| Adapter configuration | `0dc05bcd1249c1ca8177016b93a4c1c301b0f05ae12427514f8026576aa245ad` |
| Original report | `9a097599396cc142a7a6bed3083b523d1c1ec92be0ffa5fe331dac4bd0ce0601` |
| Reconstructed original 96 training rows | `602d8fb5a59d2db02cd6a9239a98096ef55c0d906ed7307c1e15f275ec2fb68d` |
| Reconstructed original 24-row development artifact | `4c9d54866719545f9a54466cd6da7b9aed4b79f935160d61e42e0639675c3f1f` |

The ZIP hash matches GitHub's retained artifact digest and the existing repository
record. Its actual adapter file hash matches the original report. Reconstructed
train/development bytes use original generator seed20261028 and match the
original report's exact retained hashes. The current generator file also matches
the recorded `1923382f27a35c3826a9d81623200533b22b8f1ea35142d5f6c1bcaae0cad101`.
No recovered archive or historical dataset was modified or committed to Git.

The adapter configuration directly confirms `q_proj`/`v_proj`; the historical
recipe records rank8/alpha16, learning rate5e-4, seed1117, 96 examples and 96
optimizer steps. These historical report counters were read, not independently
re-executed during recovery.

## Identities still requiring original-byte verification

- Base: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`; recorded weight SHA
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
- Tokenizer: recorded SHA
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
- The actual tokenizer/chat-template bytes must be checked before generation.
  `training/cpu_chat_json_recipe.py` records the system message and chat/EOS
  transform; that code alone does not replace tokenizer-byte verification.
- The separate private R10 target checkpoint/archive retains the identities in
  the [original evidence index](../evidence/original-cuda45-r10/README.md). This
  recovered CPU adapter is not substituted for that R10 checkpoint.
- The original private role-training corpus, historical Granite comparisons and
  external signing authority under #102 remain outside this recovery.

The official [pinned Hugging Face model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md)
states Apache-2.0. The authored generated-data provenance records CC0 mathematical/
symbolic task data, without harvested examples. These are source/license records;
independent contamination review and exposure-ledger completeness are still
separate requirements.
