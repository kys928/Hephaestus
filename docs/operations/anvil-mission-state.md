# Project Anvil mission state

| Item | Status on 24 November 2026 |
| --- | --- |
| Current phase | 0, source baseline and physical audit open |
| Baseline parent | `cf31eb4eddfc67983110ce5570a852d48ca2bf7e` |
| Integration ref | `feature/anvil-integration` with the Phase 0 audit and this ledger |
| Completed | Pinned main and draft heads, mapped source entry points, recorded the local test baseline attempt, verified historical triad receipts and observer outcome, ran the core CI suite with 1,205 passed and 147 skipped on both supported Python versions |
| In progress | Czech read-only S3 list from [the isolated read-only workflow](https://github.com/kys928/Hephaestus/actions/runs/45392348582), with encrypted per-object output |
| Blocked | Fresh Global volume audit requires an approved already-mounted reader or a new bounded CPU allocation. A mounted reader must stay within the admitted Global volume and approved custody. |
| Financial state | This audit started no RunPod resources. Historical run cleanup applies to its exact Pod. |
| Scientific artifacts | Original R10, Prometheus and CUDA triad source and evidence references are in the [baseline audit](../reviews/anvil-baseline-audit.md). Hold all candidate adapters and evidence at their current custody locations. |
| Approvals | Fresh scoped compute approval is required before a paid Pod. Storage deletion, promotion, production activation and independent scientific acceptance retain separate gates. |

## Resume from verified state

1. Fetch `feature/anvil-integration` and record its exact tip with `git rev-parse HEAD`. Read the [baseline audit](../reviews/anvil-baseline-audit.md) and recheck main before any code extraction.
2. Read the Czech inventory action outcome and record a redacted summary and an independently inspectable custody reference. Classify paths by custody and lineage, hash identity-critical material subject to access rights, and record duplicates and reproducibility separately from provider ETags.
3. Complete the Global inventory, verify the five original adapter archives and base snapshots, and reconcile the 8,224 candidate keys against the receipt exclusions and curated HF manifest. Verify backup and authoritative storage behavior.
4. Establish a comparable CPU baseline and review the imports and dependencies behind the proposed resolver, capability bindings and durable queue integration before changing production composition.
5. Execute subsequent milestones after their acceptance gates. Phase 1 consolidation, Phase 2 persistent knowledge and semantic protocol, Phase 3 orchestration, Phase 4 collaborative adapters and comparative results, and Phase 5 end-to-end certification remain open.

## Recovery

The integration draft contains only Phase 0 documentation. Recreate it from the recorded baseline and Git history if interrupted. Recover operations outputs through the exact Actions run and canonical custody references. Terminate only resources identified by the exact scoped grant and separate provider confirmation. Preserve the native staged lineage, both physical volumes and all certified baselines through subsequent reviews.
