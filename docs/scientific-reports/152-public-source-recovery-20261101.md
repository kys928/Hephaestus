# Original public baseline bytes recovered — 2026-11-01

The public base/tokenizer dependency previously listed as unavailable is now
resolved. A bounded read-only CPU acquisition in this public repository fetched
the exact official snapshot without inference, training, secrets or a GPU:

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`.
- Revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Acquisition source: `b5bda91e2990065c527797bfda6bd8a250283756`.
- [Actions run 41270671169](https://github.com/kys928/Hephaestus/actions/runs/41270671169), artifact `13069106451`.
- Downloaded original ZIP: 271,179,798 bytes, SHA-256
  `99218fd987991cbf9a39eb4e889e34712355daccdd995c9b6e95668ccc0e26d2`.

The downloaded ZIP was checked against GitHub's retained artifact digest,
validated for safe extraction and every extracted file independently rehashed
locally against its acquisition index. “Locally rehashed” describes a separate
byte check by the same engineering agent, not independent human custody.

| Actual file | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 269,060,552 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| `tokenizer.json` | 2,104,556 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `config.json` | 861 | `8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a` |
| `tokenizer_config.json` | 3,764 | `4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df` |
| `special_tokens_map.json` | 655 | `2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52` |
| `generation_config.json` | 132 | `87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013` |
| Official pinned model card | 6,772 | `4f97533ad95b1b2fea15fbc075c01b94578ebdd7c8138888fa43fa3abd530dc4` |

The actual chat-template string SHA-256 is
`872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788`.
The first two file identities exactly match the historical CPU experiment's
retained commitments. The pinned official model card records Apache-2.0.
The source-recovery script refuses different model/tokenizer hashes, a missing
template or inconsistent license source. No downloaded model bytes enter Git.

The one-off acquisition workflow was deliberately retired before integration.
Its initial intermediate tree failed the active-workflow allowlist check;
the allowlist was preserved, and final CI runs on the retired-workflow tree.
The script remains reproducible from a fresh output directory.

Together with [the original CPU adapter and 96/24 source recovery](152-source-recovery-20261031.md),
the public original model, tokenizer, chat template, adapter and authored
training/development bytes can now be checked directly. Historical optimizer
counters are retained evidence, not newly executed original training.

This does not recover the separate private R10 target checkpoint/archive,
original private role-training corpus or historical Granite comparisons. It
does not supply independent entrusted holdout custody, the blinded power/EOS
pilot, final design acceptance or fresh paid-resource authorization. #152 and
#102 remain open. No scientific improvement or external certificate follows
from a successful byte recovery.
