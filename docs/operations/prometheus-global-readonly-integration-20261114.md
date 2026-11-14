# Prometheus Unbound — Global Volume integration gate (2026-11-14)

## Scope of this stacked change

This is **only a read-only source adapter** for the original certified role archives. It is stacked on the research branch behind PR #201; it does not merge or authorize PR #201, launch a Pod, train Granite, access a protected holdout, modify a role, or promote an adapter.

The old PR #201 provider remains **intentionally pinned** to EU-CZ-1 and the original regional volume. Do not run `scripts/launch_prometheus_gpu_once.py` as if it supported the Global Volume. That launcher and independent watchdog require a separate reviewed cross-region implementation.

## Verified inputs and distinct stores

- Global Volume: externally verified provider ID `cmw51pjsk000007l9a0mv1jga`, mount path `/workspace-global` on the 2026-11-13 successful Controller CUDA Pod.
- The five original certified adapter archives were present under `/workspace-global/hephaestus/runtime-models/role-mastery-v1/<role>/selected-adapter-<digest-prefix>.tar.gz`, with archived role SHA/bytes from `configs/models/hephaestus_role_model_stack_v1.json`.
- Original Czech volume `cviwpryzao` remains a distinct S3-compatible **regional** evidence/backup source. The Global Volume itself is **not** S3 bucket `cviwpryzao`.
- The historical original-parent and different D-source reports remain separately SHA-pinned in Prometheus research code. Their ontologies are not interchangeable.

## Read-only adapter activation

The `prometheus_gpu_campaign` module now accepts exactly one adapter transport:

1. **Regional S3 (legacy):** `--s3-endpoint`, `--s3-region`, `--s3-bucket`.
2. **Verified Global mount (opt-in):** `--global-volume-root /workspace-global --global-volume-id <externally-verified-ID>` with no S3 adapter flags.

The Global client implements the same read-only `get_object(Bucket, Key)` interface used by `VerifiedRoleAdapters`. Its registry maps keys to the **only five** allowed Global paths. It refuses unexpected keys, mismatched volume IDs, wrong byte sizes, wrong SHA-256, missing mounts and symlink roots. The existing downstream adapter materializer independently validates archive bytes and extracts only into the explicit local cache.

No model registry, training recipe, selected intervention or protected data is modified. Output, adapter cache and Hugging Face cache are refused if placed inside the Global mount. The mount identity must be confirmed by a separate authenticated RunPod controller before invoking the worker.

## Cross-region launch requirements (NOT yet implemented or executed here)

1. Read-only authenticated RunPod account, exact Global mount and current GPU stock check; verify recorded source/receipt hashes.
2. Preserve exact, independently verified pre-existing historical evidence bytes; do not treat prior exposed development scores as independent holdout.
3. Generate a fresh one-shot source-scoped, financed lease, immutable pinned checkout/code bundle and separate *already live* off-Pod watchdog. Do not broaden the repository's maintained active workflow allowlist.
4. Allocate one exact supported Secure Cloud GPU in any eligible center with **only the Global archive mount**. Keep temporary model downloads, local SQLite/WAL, inference receipts, locks and processing on local POSIX disk. Preserve results through a separately verified durable private evidence writer, not by writing mutable state to Global storage.
5. Execute **real** frozen Diagnostician and Planner inference via `prometheus_gpu_campaign`, authenticate source/report hash, generated token IDs, GPU name, model revision and adapter hashes. Never silently replace rejected model decisions with heuristic choices.
6. Only if admitted decision selects `collect_more_evidence`, perform the approved **public development** Granite probe. Any training, repair, protected assessment, certification or promotion requires a separate research admission.

## Causal autonomy acceptance (not achieved by an ordinary run)

- Precommit actual observed evidence, available actions, and model/adapter identities.
- Compare true specialist choice to (a) a frozen fixed-action control and (b) an independently specified heuristic control on the **same admissible evidence**.
- Repeat with materially changed **authentic** evidence to establish counterfactual dependence, and an irrelevant-metadata perturbation to check invariance.
- Retain *unmodified* raw model generations, deterministic admissions/rejections, all measured parent/candidate results, and each decision owner. Do not claim decision causality from one coincidental right choice.
- If real independent Granite weaknesses aren't established, `collect_more_evidence` or `stop` can be scientifically correct. An unauthorized improvement claim is a failure of governance.

## Current acceptance state

- [x] Global role archive read adapter implemented on separate stacked branch.
- [x] Offline tests added for exact registry archives, key and volume admission, checksum failure and mount/symlink refusal.
- [ ] Independent CI status reviewed after GitHub run completion.
- [ ] New cross-region provider and off-Pod watchdog reviewed.
- [ ] Source report retrieval and actual live S3 custody verified in this campaign.
- [ ] Real GPU Diagnostician and Planner receipts obtained.
- [ ] Scientific action and counterfactual evidence-dependence independently scored.
- [ ] Controlled Granite measurement or admitted repair executed.
- [ ] Independent candidate evaluation and causal attribution accepted.

**No claim of Prometheus scientific completion is made by this integration note.**
