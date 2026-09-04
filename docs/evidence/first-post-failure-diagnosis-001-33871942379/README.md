# First post-failure diagnosis evidence

This directory freezes the exact CPU diagnosis proof produced by GitHub Actions workflow run `33871942379` at commit `1bd3c49ba2c52d58e46f20180d2a94586fd73e67`.

The diagnosis consumed a conservative projection from the already-frozen training and semantic-evaluation evidence. It first verified that the candidate checkpoint, model, tokenizer, processed-data, experiment, and run identities matched the preceding chain and that Judge exit had rejected the candidate without applying an action.

Result:

- status: `inconclusive`
- leading failure domain: `inconclusive`
- confidence: `0.0`
- issues: none
- missing baseline evidence categories: none
- causation claimed: false
- recommendation executed: false
- recommended intervention kind: `collect_more_evidence`

This outcome is intentional and scientifically conservative. The evidence proves the behavioral regression is real and that several integrity/runtime alternatives are contradicted, but it does not contain an explicit causal signal sufficient to distinguish undertraining, optimizer/scheduler problems, data coverage, overfitting, model-family limitation, or another finite failure domain.

`report.json`, `chain.json`, and `input.json` are byte-for-byte copies of the successful workflow artifact. `SHA256SUMS` records their hashes plus the digest of the original Actions ZIP; the ZIP itself is not committed.
