# Hephaestus documentation

**Start here.** This index separates how the system is built, how to operate it safely, and what its experiments have actually demonstrated. Updated 2026-11-01; evidence links retain their original provenance.

| I want to… | Read |
| --- | --- |
| Understand the system and the eight control phases | [Architecture overview](architecture/README.md) and [control/governance](architecture/control-and-governance.md) |
| Understand data, evidence custody and lineage | [Data and state architecture](architecture/data-and-state.md) |
| Install, run CPU-only tests, inspect a demo | [Operations quickstart](operations/developer-quickstart.md) |
| Avoid unsafe launch, approval or checkpoint actions | [Safe operation policy](operations/run-safety.md) |
| Prepare a reviewed integration/release | [Release and rollback checklist](operations/release-checklist.md) |
| Inspect proven CUDA45 science and limitations | [Scientific reports](scientific-reports/README.md) |
| Understand what counts as scientific evidence | [Claims and certification standard](scientific-reports/scientific-claims.md) |
| Check supported actions, real CPU research and remaining scaffolds | [Supported execution](operations/supported-execution.md) |
| Inspect all eleven repaired audit findings | [Implementation ledger](operations/remediation-ledger.md) |
| Inspect the complete implementation/research campaign | [Campaign report](scientific-reports/remediation-campaign-20261101.md) |
| Operate within the verified persistence and recovery boundary | [Restricted storage acceptance](scientific-reports/179-restricted-storage-resolution-20261101.md) and [snapshot runbook](operations/campaign-snapshots.md) |
| Inspect the authentic public Smol pilot and no-launch decision | [Pilot results](scientific-reports/152-public-pilot-results-20261101.md), [v3 review record](scientific-reports/152-prospective-protocol-v3.md) and [custodian handoff](operations/152-independent-custodian-handoff.md) |

## Status as of 2026-11-01

The reviewed R10 source and original commit lineage were integrated into `main` by regular Git merges (#153–#157). The published original CUDA45 internal experiment completed **45 native operations, five accepted frozen CUDA role inferences, and 96 real target-model optimizer updates**. A separate cold verifier passed. Those facts do **not** establish an improved model, external historical certification, automatically authorized GPU launches, unrestricted role dispatch, or a promoted checkpoint.

The eleven audit defects are repaired in focused ordinary merges. The staged loop now handles verified zero-update training failures through bounded new attempts, while retaining failed history. Actual CPU research and one-use prospective custody have separate, explicit claim scopes; neither supplies an external certificate or general proof that autonomous research chooses better experiments.

[Integration issue #151](https://github.com/kys928/Hephaestus/issues/151) remains closed for its accepted scoped review, including the documented owner protection waiver. [External certification #102](https://github.com/kys928/Hephaestus/issues/102) and [behavioral-improvement #152](https://github.com/kys928/Hephaestus/issues/152) remain open. [Documentation #159](https://github.com/kys928/Hephaestus/issues/159) tracks supported walkthroughs and remaining incomplete surfaces. Do not treat a dated snapshot as live operational state.

## Stable reference and historical material

- [Original frozen scientific evidence](evidence/original-cuda45-r10/README.md) is a **historical evidence index**, not a document to overwrite during documentation maintenance.
- [Integration notes](integration_notes/151-r10-full-history-and-runtime.md), [source ancestry ledger](integration_notes/151-historical-stacked-pr-ancestry.md) and [workflow retirement ledger](integration_notes/151-active-workflow-retirement.md) document the exact development history.
- The existing [policy references](evaluation_policy.md), [promotion gates](promotion_gate_policy.md), [replay verification](replay_verification_policy.md) and [bounded recovery](autonomous_recovery.md) remain at their established paths so old PRs and links continue to work.
- Other `docs/integration_notes/`, `docs/reviews/` and `docs/evidence/` material records particular historical decisions, not necessarily the currently supported production interface.

**Maintenance rule:** change current operational advice in these canonical guides; preserve scientific receipts, source hashes, raw findings and historical docs as records. Never convert a completed GPU run into a claim of demonstrated model improvement.
