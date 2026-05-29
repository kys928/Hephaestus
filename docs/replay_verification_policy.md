# Replay Verification Policy

Hephaestus replay verification is a read-only trust layer for deciding whether a stored run's decision context can be reconstructed from persisted evidence. It does not launch training, mutate state, edit artifacts, or infer absent evidence.

## Policy

- Verification loads existing run, lineage, decision, manifest, report, artifact-index, and memory stores.
- A missing run returns `missing`.
- Missing critical decision evidence returns `insufficient`. Critical evidence includes replay metadata, the referenced evaluation report, the judge-exit decision, promotion-gate metadata, action-boundary metadata, required manifest evidence, required checkpoint content hashes, and checkpoint-reference consistency across persisted records.
- Complete reference-level evidence without checkpoint content hashes may return `partial`; this means the decision context is useful but byte-identical model-output replay is not proven.
- `reproducible` is reserved for runs whose critical evidence is present and whose checkpoint integrity metadata can support the recorded replay scope.
- The verifier is deterministic: its report is derived only from persisted records and uses persisted run timestamps rather than wall-clock verification time.

## CLI

```bash
python -m hephaestus.cli.verify_replay --state-root state --run-id <run-id> --format json
python -m hephaestus.cli.verify_replay --state-root state --run-id <run-id> --format text
```
