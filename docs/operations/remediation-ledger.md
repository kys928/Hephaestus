# Implementation campaign ledger

Baseline: `bc79bf021c4f357a709594d4083befa18f39a82e`, audited 2026-10-31.
Original audit probes reproduced before implementation. Status requires integrated
and postmerge-verified behavior; an open PR is not a resolved finding.

| Finding | Priority | Root cause / repair | Reproduction and regression | Status / integration |
| --- | --- | --- | --- | --- |
| F01 | P1 | Self-consistent checkpoint replacement; retain producer digest outside checkpoint | `test_checkpoint_action_recovery_integrity.py`: same-path weights + regenerated manifest, missing legacy identity | implementing |
| F02 | P1 | Prepared automatic continuation skipped preconditions; share validation before mutation | Same module: crash before mutation + corrupt checkpoint; already-committed recovery | implementing |
| F03 | P1 | Concurrent staged/production controllers lack exclusive ownership | Original concurrent same-operation probe | open |
| F04 | P1 | CPU/CUDA evaluators disagree with strict scorer on duplicate keys | Original CPU/CUDA scoring probes | open |
| F05 | P2 | Decoded numeric overflow accepted | Original `1e400` probe | open |
| F06 | P2 | Final checkpoint omits cumulative counters | Original off-cadence save probe | open |
| F07 | P2 | Full accumulation wraps past epoch cap | Original 3-batch / accumulation-2 probe | open |
| F08 | P1 | Whitespace chunking loses prompt/target identity | Original long-prompt masking probe | open |
| F09 | P2 | Composed services fall back to relative configs | Original unrelated-CWD build probe | open |
| F10 | P1 | Healthy pending polls counted against failure budget | Original sixth-resume assertion probe | open |
| F11 | P1 | Legacy coordinator records denial then launches | Original sentinel backend probe; no subprocess | open |

## Larger capabilities (separate from defects)

- Milestone 1: execution integrity — in progress; not accepted yet.
- Milestone 2: training/scoring correctness — pending.
- Milestone 3 / #152: prospective protocol amendment, execution/custody pipeline
  and CPU rehearsal pending. Confirmatory science still requires authentic source
  recovery, independently entrusted holdout custody and any fresh paid approval.
- Milestone 4: genuine failed-run recovery, executable bounded transitions,
  three authentic cycles and matched fixed/heuristic/adaptive comparison pending.
- #102 external historical certification stays open; #151's historical scoped
  source acceptance stays closed. Documentation follows #159 without duplicating it.

No original R10 evidence, historical source identities, protected data, approval
records or production activation settings are changed by the campaign.
