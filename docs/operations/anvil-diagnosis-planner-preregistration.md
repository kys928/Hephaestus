# Diagnosis–Planner preregistration candidate

## Objective and evidence

Select one evaluation-only measurement to distinguish incomplete source coverage from inconsistent replay or indexing of the migrated evidence. The specialists may conclude evidence is insufficient or propose incompatible measurements. Record those outcomes.

Use only these operational observations as the evidence bundle. Their exact wording is fixed below, with independent audit provenance in the [pinned mission ledger](https://github.com/kys928/Hephaestus/blob/05cd0d8dfbc44d0d11f4a5d22b36299babd2b65b/docs/operations/anvil-mission-state.md). Keep protected model evaluation data outside the prompts.

- Initial observation. The frozen inventory contains 8,224 candidate source references. The audit could not resolve or safely follow 111 of them.
- Second-round observation. The independently mounted Czech and Global inventories matched size and SHA-256 for all 8,113 readable candidates. The comparison does not measure which files are actually returned by runtime retrieval.

Before scheduling, independently confirm that the approved artifact writer has published each exact observation and objective as canonical, digest-addressed material in the chosen investigation, run and lineage. Record their returned SHA-256 references. The first two inference slots share the empty advisory frontier and initial observation. The second two slots share the post-round-one frontier, both earlier recommendations and both observations. Admit earlier raw receipts before capturing that frontier. Each specialist sees the same frontier for its round regardless of job completion order.

## Frozen configuration and acceptance

| Control | Value |
|---|---|
| Original registry SHA-256 | `b48e3bb7cdc7f397cbdfb99384786c0d4b9c81e5ef0d404df6575f5a34a01b7d` |
| Frozen role contract SHA-256 | `17e057b23e7bbb4abec39137269c6c0a0a501cf2ee73313692bcdc99e473da14` |
| Original role certificate SHA-256 | `903d4fd2ba5a244f16a3f2f9f03c860668eb074317b37dc5340d0bdc67afbb69` |
| Diagnosis | `mistralai/Ministral-3-3B-Reasoning-2512@4a36357c811bf511a7b625d132e12f22408aac91`, adapter `534d1929d76fc64d07d8d36e85115d5660c113e63b022206bd500d288bdff32c` |
| Planner | `mistralai/Ministral-3-8B-Reasoning-2512@81eaece1948f3875421d9a45bc55487d10e2d894`, adapter `1d4315b588c31262435cef1ad7df5c63edecd948b7992eca6e972a63bc18df85` |
| Inference | Existing certified reasoner and canonical Anvil context, CUDA device 0, offline snapshots, sequential loads, seed 271828, 512 new tokens per slot, four slots, zero retries |
| Compute | One 48 GiB L40S candidate with independent exact-Pod teardown and 60-minute total wall ceiling including cleanup |
| Rate reference | [RunPod public L40S rate](https://www.runpod.io/gpu-models/l40s) lists Secure Cloud at USD 1.09/hour. Availability, placement, storage charges and total provider quote require authenticated readback before a grant. |

An independent reviewer rechecks exact identities, all receipts and reconstructed contexts. Score semantic validity and evidence grounding as pass/fail. Score useful disagreement and revision quality as absent, unsupported or grounded. Classify the terminal result as supported agreement, justified disagreement or unresolved. Missing or rejected outputs retain their receipt and remain missing or rejected in the scoring table. Agreement, revision and Coder admission are optional outcomes. Approval covers inference only, without training or automatic capability execution.

Before requesting GPU spend, attach the measured provider quote, exact region and volume placement, immutable runtime image digest and environment lock, model custody and offline-load receipts, evidence artifact references, cleanup callback receipt, and independent attestation policy. Keep the compute grant at zero until those prerequisites are established.

## Priority acquisition requests

1. **Diagnosis.** Target the two canonical shard files at the [pinned revision](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512/tree/4a36357c811bf511a7b625d132e12f22408aac91). The upstream SHA-256 values for `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors` are respectively `2fe8ca14c36eb7f76f70d9f4a3196532e622683f6c2ea74a33e122f771e362e7` and `68c1cb14d59a8013f7b6ba511dfddffcec7a9c971a8bb31af3889c1e4d5a3124`. Confirm exact pointer lengths before an authorization request. Propose a ceiling of 8,000,000,000 bytes including auxiliaries and reject the request if exact lengths exceed it.
2. **Planner.** Reuse the four Czech originals only after the separate streaming receipt qualifies them. Resolve the four canonical names from the [pinned index](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512/blob/81eaece1948f3875421d9a45bc55487d10e2d894/model.safetensors.index.json). Acquire auxiliary files only if missing after independent alias and digest validation.
3. **Both models.** For the frozen Transformers loader, review `config.json`, `generation_config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`, `tekken.json`, `processor_config.json` and `params.json` against the pinned trees. Freeze per-file lengths and digests for the needed set. Exclude the duplicate consolidated weight export. Stage only absent or mismatched files in a qualified immutable destination, independently read back and hash them, then rehearse offline loading. Restore from independent credentials before calling backup custody complete.
