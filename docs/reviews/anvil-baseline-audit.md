# Project Anvil baseline audit

**Source snapshot, 24 November 2026.** GitHub default branch `main` and the locally fetched ref both resolve to `cf31eb4eddfc67983110ce5570a852d48ca2bf7e`. The baseline is the standalone diagnosis coverage repair. The two named merge bases below are this commit and its immediate predecessor, `f27882fc08aa6d7c162a4b12c5dc522e9cdb69e2`.

## Dependency census

Parsed 416 package modules, 78 scripts, and 171 test files.
Found 1359 unique directed package-module import edges and 2 dynamic import call sites across scanned files.

| Package area | Modules | Reachable from named entry points | Direct script targets | Direct test targets |
| --- | ---: | ---: | ---: | ---: |
| `(package root)` | 2 | 1 | 0 | 1 |
| `app` | 2 | 1 | 0 | 1 |
| `backends` | 15 | 13 | 1 | 7 |
| `cli` | 5 | 4 | 0 | 4 |
| `control` | 20 | 15 | 5 | 13 |
| `data` | 28 | 27 | 13 | 19 |
| `development` | 13 | 0 | 3 | 12 |
| `diagnosis` | 7 | 4 | 2 | 4 |
| `evaluation` | 25 | 6 | 5 | 18 |
| `evaluators` | 2 | 2 | 0 | 0 |
| `generation` | 5 | 4 | 2 | 4 |
| `infrastructure` | 7 | 6 | 1 | 4 |
| `interfaces` | 3 | 2 | 0 | 2 |
| `jobs` | 6 | 6 | 0 | 2 |
| `llm` | 12 | 4 | 2 | 6 |
| `planning` | 3 | 1 | 2 | 2 |
| `policy` | 24 | 20 | 2 | 19 |
| `production` | 35 | 21 | 11 | 27 |
| `providers` | 19 | 10 | 9 | 10 |
| `recovery` | 10 | 8 | 0 | 1 |
| `roles` | 21 | 10 | 0 | 10 |
| `runtime` | 8 | 6 | 0 | 0 |
| `safety` | 8 | 2 | 0 | 1 |
| `schemas` | 71 | 56 | 12 | 40 |
| `scoring` | 7 | 3 | 0 | 1 |
| `state` | 31 | 20 | 3 | 27 |
| `storage` | 10 | 5 | 4 | 5 |
| `training` | 11 | 4 | 4 | 7 |
| `utils` | 6 | 2 | 1 | 1 |

Entry-point roots include the infrastructure CLI, console, diagnostic and replay CLIs, the production, staged and original CUDA compositions, the RunPod smoke command and the role probe. Static edges include optional and type-only imports and omit dynamic dispatch, configuration, shell wrappers and implicit package initialization. A zero is an investigation lead and provides no basis for deletion.

| Pull request or reference | Head | Merge base family | Main-only / head-only commits |
| --- | --- | --- | ---: |
| [#201](https://github.com/kys928/Hephaestus/pull/201) | [`dd966f59684c110f9b4a53ba5225aa85302d1794`](https://github.com/kys928/Hephaestus/commit/dd966f59684c110f9b4a53ba5225aa85302d1794) | `f27882f` | 1 / 75 |
| [#203](https://github.com/kys928/Hephaestus/pull/203) | [`7068caa19322c4f5068baa7b2f48318880e9810e`](https://github.com/kys928/Hephaestus/commit/7068caa19322c4f5068baa7b2f48318880e9810e) | `f27882f` | 1 / 6 |
| [#204](https://github.com/kys928/Hephaestus/pull/204) | [`da953f575a119a97f330b63cd9430d0c88931d24`](https://github.com/kys928/Hephaestus/commit/da953f575a119a97f330b63cd9430d0c88931d24) | `f27882f` | 1 / 75 |
| [#205](https://github.com/kys928/Hephaestus/pull/205) | [`7d7b7bf0fcfd5537b7e67843b1e8b630240a1d2c`](https://github.com/kys928/Hephaestus/commit/7d7b7bf0fcfd5537b7e67843b1e8b630240a1d2c) | `f27882f` | 1 / 81 |
| [#206](https://github.com/kys928/Hephaestus/pull/206) | [`4794d9c76424351e0841c414fdc8d44e711f0194`](https://github.com/kys928/Hephaestus/commit/4794d9c76424351e0841c414fdc8d44e711f0194) | `f27882f` | 1 / 107 |
| [#207](https://github.com/kys928/Hephaestus/pull/207) | [`3127d152e8b2d2e7d8b64c9bb25fa05bc3d0f7a7`](https://github.com/kys928/Hephaestus/commit/3127d152e8b2d2e7d8b64c9bb25fa05bc3d0f7a7) | `f27882f` | 1 / 114 |
| [#208](https://github.com/kys928/Hephaestus/pull/208) | [`ce0b847bc498918ccb5614843ed627067674e2f0`](https://github.com/kys928/Hephaestus/commit/ce0b847bc498918ccb5614843ed627067674e2f0) | `f27882f` | 1 / 125 |
| [#209](https://github.com/kys928/Hephaestus/pull/209) | [`02b39fe2b08c2b9a454ada1846fadfb46167fea9`](https://github.com/kys928/Hephaestus/commit/02b39fe2b08c2b9a454ada1846fadfb46167fea9) | `f27882f` | 1 / 132 |
| [#212](https://github.com/kys928/Hephaestus/pull/212) | [`88c51ba30efafa02d07e3a488fcdf2c20a35f16e`](https://github.com/kys928/Hephaestus/commit/88c51ba30efafa02d07e3a488fcdf2c20a35f16e) | `cf31eb4` | 0 / 31 |
| [#214](https://github.com/kys928/Hephaestus/pull/214) | [`89f8b234f5b3c3b845d8cfce03138bbf1a407a61`](https://github.com/kys928/Hephaestus/commit/89f8b234f5b3c3b845d8cfce03138bbf1a407a61) | `cf31eb4` | 0 / 69 |
| [#216](https://github.com/kys928/Hephaestus/pull/216) | [`5787e6db2c8a1423cf4b25f3ecc7121d64eb2e3a`](https://github.com/kys928/Hephaestus/commit/5787e6db2c8a1423cf4b25f3ecc7121d64eb2e3a) | `cf31eb4` | 0 / 78 |
| Triad frozen run source | [`6cc805b6ed91d991d1ed9dbc2a078c813d721434`](https://github.com/kys928/Hephaestus/commit/6cc805b6ed91d991d1ed9dbc2a078c813d721434) | `cf31eb4` | 0 / 105 |
| Triad recovery branch | [`018975b7e92614d6183c469f8b0531d2395a8c36`](https://github.com/kys928/Hephaestus/commit/018975b7e92614d6183c469f8b0531d2395a8c36) | `cf31eb4` | 0 / 107 |
| Triad independent observer | [`ad1cb4effe5ba8480a309a210e61f08f341f3f16`](https://github.com/kys928/Hephaestus/commit/ad1cb4effe5ba8480a309a210e61f08f341f3f16) | `cf31eb4` | 0 / 104 |
| Standalone diagnosis coverage | [`360e34229f4ac718e5453669c3170e872446372b`](https://github.com/kys928/Hephaestus/commit/360e34229f4ac718e5453669c3170e872446372b) | `f27882f` | 1 / 138 |

## Current execution and ownership

| Surface | Source and entry point | Disposition |
| --- | --- | --- |
| Production composition | `src/hephaestus/production/composition.py` builds `ProductionRuntime`. The governed handoff script and CPU development benchmark instantiate it. Original CUDA wiring and staged composition add explicit service bindings. | Keep the root. Use these seams to admit additional services. |
| Staged training | `control/staged_autonomous.py`, `control/staged_state.py`, `production/staged_driver.py` and `production/loop.py` preserve the eight boundaries and native replay. | Keep. Expose the workflow as an admitted capability with independent verification. |
| Execution queue | `jobs/sqlite.py` and `jobs/durable_worker.py` implement leases and persistent job handling. `jobs/queue.py` and `jobs/worker.py` also supply a transient execution path. The draft research CLI instantiates a durable queue against its selected root. | Keep durable primitives. Bind research handlers to the authoritative runtime and verify ownership, job registration and restart behavior. Retain transient paths for existing callers. |
| State and evidence | `storage/sqlite.py`, `storage/filesystem.py`, `storage/verified.py`, `state/single_writer.py`, `state/evidence_store.py`, `state/experiment_memory.py` and `production/state.py`. Storage admission distinguishes local rehearsal and persistent research. | Keep. Extend transactional state and verified evidence references with investigation scope. Verify physical durability before admitting a new authoritative writable location. |
| Artifact and state adapters | Filesystem and SQLite are used by production. `storage/s3.py` handles object storage and `storage/postgres.py` contains optional PostgreSQL plumbing. | Keep existing interfaces. Require operational qualification for any different state backend. |
| Registry and inference | `providers/models/role_stack.py` loads the frozen role stack config. `llm/certified_runtime.py`, `llm/role_contract.py`, `llm/inference_receipts.py` and `production/model_roles.py` implement the typed runtime path. The execution backend registry and candidate discovery catalog serve different callers. | Keep the certified source of role identities. Route candidate Coder inference through the shared interface with provisional status. |
| Role artifact resolution | Main has `production/mounted_original_role_adapter.py`. PR #205 and the triad family add `development/prometheus_global_adapter_source.py` and `research/global_role_archive_source.py`. PR #203 adds Global placement and artifact verification components. | Consolidate transport behind one resolver after parity tests cover mount identity, bytes, hash, archive extraction and confinement. |
| RunPod | `providers/runpod/execution.py` is the main provider transport. `providers/runpod/storage.py` addresses regional object access. #206 and the triad branches add Global worker controllers and an external watchdog. | Extend the canonical provider. Preserve independent watchdog behavior. Retire one-shot controllers from production consideration only after equivalent tested fencing and cleanup. |
| Governance | `policy/action_registry.py`, `policy/action_authorization.py`, `policy/approval_policy.py`, `control/deterministic_role_boundary.py` and `production/actions.py` retain execution authority. | Keep. Extend capability descriptors through these services. |
| Backend alternatives | `backends/registry.py` wires dry-run, local process, HF causal and Ardor paths. Development and test roots exercise distinct historical and fixture paths. | Keep existing behavior while checking callers. The fixture paths provide contract coverage and the offline verifier provides bounded independent repair verification. |
| Operator entry points | Installed `hephaestus-infrastructure` CLI, module console under `app/console.py`, CLI inspection/replay commands and governed handoff scripts. The `hephaestus-research` CLI exists on the draft research stack. | Preserve inspection and support commands. Reuse the production composition to admit research capabilities. |

Five workflow files are present in pinned main. Their trigger definitions cover autonomous integration, the certified role stack, the integration #151 CPU foundation, the integration #151 R10 CPU runtime and production autonomy integration. The paid research workflows remain on historical operations refs. Counts in this report describe tracked files, so the dependency evidence for script entry points on external deployments and dynamically loaded services remains open.

## Branch integration decisions

| Branches | Decision and dependency |
| --- | --- |
| #201 → #205 → #206 → #207 → #208 → #209 | This is the Prometheus development family. The first three supply experimental model-backed decisions, a verified Global role transport and supervised lifecycle logic. The last three preserve evidence calibration, experimental contrasts and the authentic coverage blind spot. Select behavior by path and test. Main already carries the standalone coverage repair. |
| #203 and #204 | #203 supplies reusable Global artifact publication and placement primitives. #204 contains the migration and custody history. Preserve the history and select reusable verification directly. |
| #212 → #214 → #216 | Reuse reviewed intake and durable handlers from #212, signed frozen-role research and admitted branch/rollback/restart handoffs from #214, and Coder contracts and isolated verification from #216. The draft CLI and ownership of research state need reconciliation. |
| Triad recovery and observer | Frozen source is 105 commits ahead, with 44 changed paths and roughly 6,000 additions. The recovery head adds one paid workflow commit on top. The independent observer is a separate head. Preserve both and extract source changes with exact provenance after testing. |
| Standalone diagnosis fix | The separately verified coverage branch and its CI witness are useful provenance. Treat the current main implementation as the integration target. |

## Physical storage and historical evidence

| Evidence | What is verified | Outstanding |
| --- | --- | --- |
| Czech volume [inventory run](https://github.com/kys928/Hephaestus/actions/runs/43256848721) | Historical list-only audit job passed. | Fresh listing, approved custody for names, required hashes and associations. S3 ETags are metadata and can differ from SHA-256. |
| Global recovery [#204](https://github.com/kys928/Hephaestus/pull/204) | The verifier source requires 8,224 source candidates to reconcile exactly as readable files plus accounted exclusions. Original independent verification run #43257715054 did not enter its rehash job. The later receipt-only run #43305182543 failed its verification step and its cleanup succeeded. | Confirm the later receipt-aware run against retained immutable receipts. Check the 111 alias targets independently and compare exact path, size and hash against the HF manifest. Count equality alone cannot establish parity. |
| Frozen triad [GPU job](https://github.com/kys928/Hephaestus/actions/runs/45375181356), [independent observation](https://github.com/kys928/Hephaestus/actions/runs/45376969755) | The CUDA job failed at campaign completion. The independent observer recovered two CUDA role receipts and confirmed exact Pod absence. Semantic disagreement prevented Coder inference. | Replay raw role outputs through protocol v2 once available. Offline identities of all five full base-model snapshots and original adapter archives still require independent verification. |
| Backup and authoritative state | Historical receipts and archives provide scoped evidence. The storage policy requires identified Linux mount semantics, and a source recovery controller exposed a Global/SQLite incompatibility. | Verify regional transaction, WAL, crash recovery and backup procedures. Keep Global artifacts immutable. |

The registry pins the following base revisions at the source snapshot.

| Role | Model | Revision |
| --- | --- | --- |
| Controller | Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Diagnostician | mistralai/Ministral-3-3B-Reasoning-2512 | `4a36357c811bf511a7b625d132e12f22408aac91` |
| Planner | mistralai/Ministral-3-8B-Reasoning-2512 | `81eaece1948f3875421d9a45bc55487d10e2d894` |
| Evaluator | ibm-granite/granite-3.3-8b-instruct | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |
| Judge | microsoft/Phi-4-mini-instruct | `cfbefacb99257ffa30c83adab238a50856ac3083` |

## Test baseline

The first local run used the core package, pytest and independent-certification dependencies. It reported 968 passed, 85 skipped, 287 failed and 24 errors. The first reproduced failure was an expected refusal to open SQLite on the workspace's default `/tmp` tmpfs. Numerous other cases use the same storage boundary. The local result needs a rerun on an admitted mount with the pinned R10 CPU dependencies before it can serve as a comparable full regression baseline. [The historical published R10 result](https://github.com/kys928/Hephaestus/pull/157) describes an earlier revision and environment.

Twelve focused tests passed for the certified role stack, object storage and secrets, and instruction-pair normalization.
