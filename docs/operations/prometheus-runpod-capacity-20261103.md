# Prometheus GPU placement — authenticated live blocker (2026-11-03)

Issue #195 / PR #201. **This is authenticated public provider capacity evidence, not Granite inference or scientific autonomy evidence.**

## Independently reproduced from live RunPod REST v2

[Read-only authenticated EU datacenter catalog run 41688420657](https://github.com/kys928/Hephaestus/actions/runs/41688420657), verified successful, `2026-11-03T17:52:37Z`.

The request was `GET /v2/catalog/datacenters?include=GPU_AVAILABILITY` using an existing GitHub Actions credential and made **no Pod allocations, S3 writes, training updates or provider resource mutations**. It returned 32 datacenters globally. Filtered EU results:

| Datacenter | gpuAvailability | Reported in-stock GPU |
|---|---|---|
| EU-CZ-1 | omitted / null | none returned |
| EU-FR-1 | omitted / null | none returned |
| EU-NL-1 | omitted / null | none returned |
| EU-RO-1 | one row | NVIDIA RTX PRO 4500 Blackwell (LOW) |

The original network volume `cviwpryzao` is pinned to **EU-CZ-1**, while the existing deployment adapter requires its pod in the same datacenter with a verified network mount. **No compatible EU-CZ-1 GPU is currently listed in the live authenticated catalog.** The positive stock of EU-RO-1 does not by itself prove the original volume can be mounted there or that the quoted GPU can safely load the whole role stack.

The prior original-parent and historical-D source artifacts passed their exact SHA-256 checks during both one-time worker attempts. The real read-only CPU preflight returned `inconclusive` and exactly `collect_more_evidence` as the admissible scientific action. Those deterministic results do **not** prove genuine model-generated Diagnostician or Planner decisions.

The previous launch attempts failed **before GPU allocation**. The independent watchdog for [run 41681514197](https://github.com/kys928/Hephaestus/actions/runs/41681514197) ultimately printed `PROMETHEUS_WATCHDOG_NO_ALLOCATION=true`. Any more recent watchdog is separately monitored by its source-scoped identity and must likewise complete safe cleanup.

## Adapter schema correction

The original `RunPodExecutionAdapter.select_gpu` parser accepted only `id`/`availability` inside `gpuAvailability`; the current documented v2 datacenter response may provide `gpuId`/`stockStatus`. The compatibility parser now accepts either form **only for recognised positive stock levels**, and refuses missing or malformed lists. New tests cover both current and older forms. This was an independent correctness bug worth fixing, but the authenticated run above confirms **EU-CZ-1 actually omitted the GPU list**; the final blocker there is not merely a parser issue.

## Next safe route

Do **not** claim that a model was loaded or trained. Do **not** weaken the S3/network-volume locality checks, fabricate available hardware, or remount the EU-CZ-1 volume in a different region without verified provider support. To continue actual model-backed research, first review one of two explicit deployment options:

1. A genuinely compatible EU-CZ-1 GPU becoming available, checked against live provider stock and VRAM; or
2. An independently reviewed **cross-datacenter S3-only worker** or carefully copied, integrity-verified new network volume alongside an available suitable GPU elsewhere. This requires a new provider execution contract, source/artifact SHA-256 checks, bounded billing and independent cleanup.

The first frozen-parent **development** diagnostic remains proposed, not accepted or consumed. No protected holdout, training or production promotion occurred.

All one-time credential-bearing workflows from the proposed integration branch must be retired before merge. Temporary workflows are not part of permanent source integration.
