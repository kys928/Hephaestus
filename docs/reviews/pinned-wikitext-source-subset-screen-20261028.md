# Pinned WikiText-2 continuation source: executed whole-train eligibility screen

**Evidence run:** [GitHub Actions 40603259789](https://github.com/kys928/Hephaestus/actions/runs/40603259789) PASS, actual frozen tokenizer and pinned source read, 8 focused tests.  
**Raw evidence artifact:** [12779935305](https://github.com/kys928/Hephaestus/actions/runs/40603259789/artifacts/12779935305). Independently downloaded and checked two archive members.  
**ZIP SHA256:** `a270c6a13df2394735ce97f59698a92301f1ae9aa95a7f8ce07b3f38cde42d92`  
**JSON report SHA256:** `87f6e803cebb2baca15ce4bb388e327a3401358107c12f145bafef7e27ff016f` (matches checksum member).  
**Source-selection commitment:** `sha256:87ae1f4ab0f71693f0f20e733ebd7aa8ff25edc252f51f9ea306ef74ac7fe69e`.

## Actual candidate identity and retained source contract

- Provider `huggingface`, dataset `Salesforce/wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Only `wikitext-2-raw-v1/train-00000-of-00001.parquet`, 6,357,543 bytes, SHA256 `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
- Original frozen tokenizer content SHA256 `3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25` and original tokenizer artifact ref `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`.
- Original frozen public eval-pack file SHA256 `cbbc110822877a07b5e768d049064dbc9eadd5aaf0d004444ff072c5f0af5601`.
- Source card SHA256 `d5aabf145d341366745ac703b5844c442c143943b6c77929481395c822b1c28c`.
- Accepted source-only scope ref `sha256:d7b3e42bad911f14095fb4b1bd9d0caa434a7eea7bec52472b70effcdffced69` authorizes **screening only**; not training or protected-reference access.

## Measured full-train screening result

| Outcome | Exact measured rows |
| --- | ---: |
| Whole pinned training file scanned | 36,718 |
| Eligible unique rows (English, no flagged PII, sufficient contiguous tokens) | **7,232** |
| Empty rows | 12,951 |
| Under 160 tokens | 9,204 |
| Heuristic non-English rows | 5,217 |
| Normalized repeated rows | 2,060 |
| Regex PII candidate rows excluded | 54 |
| Total rejected | 29,486 |

Eligible set = **1,755,120 original frozen-tokenizer tokens, 0 unknown**, **13,380 genuine within-row 128-prefix/32-target windows** at stride 64. No row stitching, altered tokenizer, EOS insertion, edited source text, or flattened-newline backdoor. No lexical overlap with the six public pack prompts was flagged, but that is NOT an independent sealed reference audit. The source indices plus normalized source-content hash are bound into the commitment above, not exported as raw text.

## Honest clearance outcome: HOLD

This proves *technical eligibility for a specific English LM continuation source subset only*. It does not prove JSON or constrained-short-answer capabilities. Neither source text nor accepted token sequences were materialized as training input.

The pinned card's metadata says CC BY-SA 3.0 + GFDL, while its licensing prose says CC BY-SA 4.0. This difference is preserved and **not** resolved as a legal judgment; original article/revision attribution and license obligations need an independent authorized rights review. All 54 regex-flagged rows are excluded by screening rule, but no exhaustive PII/semantic privacy clearance has happened. The 3 lexical near-duplicate candidates from the earlier quarantine scan also need review beyond exact normalized deduplication.

Required independent source-level admission remains missing:
1. Externally accepted source-rights and attribution decision, with exact candidate/files/processing profile and provenance.
2. Separate full-corpus source screening scope and **four original-text** training/development/diagnostic/holdout reference custody manifests, signed external custodian and proper permissions.
3. Matching full-corpus `pii`, `exact_near_duplicate`, `semantic_contamination`, and `frozen_evaluation_leakage` audit reports attached to one authentic `DatasetSourceReview`. No unreviewed private text or evaluation set may be read in CI.
4. Original staged `dataset_selection_approval` **and** `dataset_acquisition` approval issued independently for the selected candidate/selection decision. Then, and only then, use `IndependentDatasetReviewPolicy.require_clearance` and the existing native operation 12, followed by real acquisition/preprocessing/training.
5. New original durable campaign launched from operation **1**, on a persistent mounted store. Historical 11/45 temporary smoke cannot be imported. The original project CPU-first vs certified CUDA production admission also needs an explicit honest execution-mode plan; don't call advisory CPU scorer certified CUDA.

**No training, source acquisition, S3 writes, GPU spending, promotion or certification was performed by this screening.** It provides a precise source-scoped *candidate* for the independently authorized scientific review, not an approval.
