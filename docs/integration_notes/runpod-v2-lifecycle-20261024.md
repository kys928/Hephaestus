# RunPod API investigation and v2 migration

Inspected 2026-10-24 against PRs #81–#86, combined tree
`0a9c85bee084de9b16805f03c92b3edfae25c108` / GitHub commit
`f3e4ddb0dbe069316294c574912726abd879f40a`. All six PRs were draft/unmerged;
all three workflows at that head passed. Baseline locally: 563 passed, 6 skipped.

## API ownership and callers

| Path | Before | Use |
|---|---|---|
| `providers/runpod/execution.py` | Configured REST base, flat v1 Pod body | Historical scientific training/evaluation launchers and opt-in live smoke |
| `scripts/launch_first_bounded_scientific_training.py` | Provider above | `first-bounded-scientific-training.yml` |
| `scripts/launch_first_semantic_evaluation.py` | Provider above | `first-semantic-evaluation.yml` |
| `scripts/launch_second_controlled_{training,semantic_evaluation}.py` and training v3 | Provider above | `second-controlled-experiment.yml` |
| `scripts/runpod_capacity_selection.py` | Historical v1 GPU preferences; comment about blocked GraphQL catalog | Imported by second controlled training v3; no GraphQL HTTP call |
| `providers/runpod/storage.py`, certified adapter retrieval | S3, independent credentials/protocol | Network-volume objects; unaffected by REST migration |
| `ProductionCompositionRoot` | Local backend and HF subprocess lifecycle | Did not compose RunPod deployment automatically |

No executable GraphQL caller or pre-existing REST v2 lifecycle was found in the
checked source/workflows. Historical evidence mentioning GraphQL is retained.
This does not establish what external deployment code or an external environment
currently uses. GitHub repository variable values were not retrieved.

## Official contract evidence

Read the official [v1 overview](https://docs.runpod.io/api-reference/overview),
[migration guide](https://docs.runpod.io/api-reference-v2/migrate-from-v1),
[create](https://docs.runpod.io/api-reference-v2/pods/create-a-pod),
[list](https://docs.runpod.io/api-reference-v2/pods/list-pods), and
[terminate](https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod) documentation,
and downloaded `https://api.runpod.io/v2/openapi.json` for schema inspection.
The overview and migration guide explicitly state retirement on November 15, 2026.

The adapter now requires `RUNPOD_API_BASE_URL=https://api.runpod.io/v2` and fails
before credential resolution on another endpoint. Existing workflows pass this
through the repository variable; update that deployment variable before launching
an authorized future workload. This change does not modify any repository secret,
variable, production routing flag, or live Pod.

## Implemented lifecycle

- Nested `gpu`/`cpu`, `image`, `disk`, `cloud`, `mounts.network`, and explicit
  `entrypoint`/`cmd`. Clearing ENTRYPOINT preserves the historical complete command
  semantics rather than passing `bash` as an argument to an image entrypoint.
- v2 accepts one GPU ID. Multiple preferences are resolved from the pinned
  datacenter's `GPU_AVAILABILITY` catalog before allocation intent. Recovery uses
  the original allowlist/hash and never reruns placement or POSTs again.
- v2 exposes no interruptible create field. Such a request is rejected rather
  than silently changed. Datacenter and network volume stay fixed.
- Cursor-paginated Pod discovery scans all pages, rejects invalid pagination and
  duplicate identity matches, and checks immutable request/resource identities.
- Stable intent before POST, no allocation retry after timeout or receipt loss.
  An existing v1 intent blocks a v2 allocation under the same identity and needs
  explicit migration reconciliation; it cannot be silently abandoned.
- Termination intent persists before DELETE. A retry after lost acceptance can
  repeat DELETE safely; 204 or 404 records termination. Failed requests retain
  the resource identity. A terminated allocation cannot be resubmitted.
- `PROVISIONING` and `STARTING` are accepted allocation states, never readiness;
  `is_ready` requires actual `status=RUNNING`. Historical polling reads v2 status.

## Validation and limits

571 passed, 6 skipped (CPU-only complete suite). Includes eight new v2 lifecycle
tests, plus the existing acceptance-window/restart/timeout tests now using v2
response envelopes. Provider doubles are explicit; no RunPod request with account
credentials, allocation, termination, inference, or training was performed.
Live capacity, environment configuration and end-to-end GPU operation remain
unverified. Old v1 ledger migration deliberately fails closed pending review.
