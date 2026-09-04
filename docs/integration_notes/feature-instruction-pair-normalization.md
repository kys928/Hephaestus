# Integration note: instruction-pair normalization

## Scope

This data-owned amendment extends the existing record normalizer so common instruction-pair schemas can enter the existing preprocessing pipeline without changing shared contracts.

Supported pair aliases now normalize into the canonical `prompt` / `target` shape:

- `prompt` / `target`
- `input` / `output`
- `instruction` / `response`
- `question` / `answer`

The downstream wrapper construction, chunking, manifest generation, preprocessing report, and `TrainableDataContract` logic are unchanged.

## Why this is required

The first autonomously selected post-failure dataset, `sail/symbolic-instruction-tuning`, uses `input` and `output` columns. The prior normalizer would reject those records as unsupported, so acquisition could succeed while preprocessing produced no trainable records.

This amendment is intentionally narrow. It does not add arbitrary schema inference, chat-message flattening, remote acquisition behavior, approval behavior, or training behavior.

## Safety behavior

A partially present recognized pair is rejected as malformed rather than silently falling through to a text schema. Existing `prompt` / `target` and plain `text` behavior remains unchanged.

## Tests

`tests/test_instruction_pair_normalization.py` covers all added aliases, malformed partial pairs, and preservation of existing behavior.
