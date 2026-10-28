# Governed original verifiable data

Depends on #118. Original arithmetic, JSON aggregation, string-reversal and graph shortest-path examples are generated with bounded difficulty, explicit seeds and balanced family quotas. Independent validators parse original prompts and use Fraction arithmetic, an independent string construction and Floyd-Warshall (rather than the generator's BFS). Incorrect values, duplicate JSON keys and numeric/boolean type substitutions are rejected.

Canonical semantic task keys partition train/development/evaluation globally by hash independently of generation seed. Equivalent commutative operations, JSON list permutations and reordered graph edges cannot cross partitions. Near-text flags for supported grammars are resolved only when the independent validator establishes distinct canonical tasks; unresolved matches block admission. This measures task isolation, not independent scientific certification or general language coverage.

Generator/validator code hashes, specifications, rejected duplicates, balanced distributions, source hashes and sample commitments are immutable. Prospective evaluation is frozen before training exposure. Tier A admission is limited to specifically preauthorized generator code hashes and exact processing/tokenizer/experiment/resource scope. Token artifacts retain exact IDs and mask the complete prompt; worker loading reproduces source, authorization, tokenizer and every mask. No EOS insertion or truncation, and no generic text fallback.

The explicit `original-task-token-ids.v1` format is separate from PR #116's `trainable-token-ids.v1` continuation contract. No WikiText recipe or frozen CUDA identity is changed. Original generated task data cannot satisfy natural-language pretraining requirements.

`run_internal_data_intelligence.py` performs actual public metadata/card triage on WikiText and GSM8K, rejects them without accepted full-source evidence, continues to the original generated source, executes full-source screening with the pinned pretrained CPU model's real tokenizer and selects/materializes the eligible source under the internal policy. It does not train a model or claim campaign progress.
