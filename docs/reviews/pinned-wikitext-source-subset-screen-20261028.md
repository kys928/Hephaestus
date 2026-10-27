# Exact pinned WikiText continuation selection — source-only follow-up

This branch does not claim the old temporary 11-stage smoke is a production state.
It is stacked on the new zero-history campaign bootstrap PR #113.

Scope: reuse the exact reviewed and checksum-pinned train file from the accepted
source-only `quarantine-source-screening-scope.v1` (no private reference access
or training). The new screen reads only the specified WikiText-2 raw train
Parquet bytes and original S3 tokenizer, compares both SHA digests against the
existing operator source-screening scope, and checks the exact source-card
rights declarations.

Conservative selection profile:
- preserve native normalization/newlines; never silently flatten or re-tokenize
  with a different vocabulary;
- remove exact normalized duplicate rows, all detected PII-regex candidates,
  heuristic non-English content, explicit public-pack lexical overlap, short
  rows and any row whose *raw* frozen-tokenizer unknown rate exceeds 1%;
- verify independent 128-token prefix and 32-token target windows *within*
  each row, stride 64, no cross-row stitching and no EOS insertion;
- emit only aggregate rejection counts, number of qualifying rows/windows and
  an index/source-bytes SHA commitment. No selected row text or trainable data.

This is source screening, not license clearance, privacy certification,
semantic contamination testing, independent private holdout access, or native
dataset-selection/acquisition approval. The preserved original WikiText
CC BY-SA 3.0/GFDL vs card 4.0 attribution ambiguity remains unresolved.
Source selection must be independently reviewed and approved under the
existing `IndependentDatasetReviewPolicy` and the original stage-12
`acquisition_approval_gate`. A full 45-stage run, GPU execution and model
promotion are *not* claimed.
