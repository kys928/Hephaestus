# Hephaestus fresh-Pod boot and original role inference smoke — 2026-11-13

**Scope:** Application/runtime readiness on pinned GitHub `main`, read-only migrated Global artifact resolution, original five LoRA adapter bytes and one real pinned CUDA Controller inference. This is *not* new training, model promotion, unrestricted agent autonomy, external scientific certification, or proof that all five original base models loaded on a new CUDA Pod.

## Observed results

- **CPU application boot:** [GitHub Actions #43426990148](https://github.com/kys928/Hephaestus/actions/runs/43426990148), `boot` job passed. Single CPU Pod `gx4da4iap8bo3j`, secure EU-CZ-1, observed price **$0.06/hour**. Live recheck #43427033714 attempt 2 confirmed Pod `RUNNING` and private S3 boot record `phase=READY`.
- **Code:** immutable GitHub main revision `f27882fc08aa6d7c162a4b12c5dc522e9cdb69e2` executed from container-local checkout; all eight demo orchestration phases completed and doctor/replay CLIs returned successfully. Synthetic demonstration, not a new model training campaign.
- **Global Volume:** `cmw51pjsk000007l9a0mv1jga` mounted at `/workspace-global`, historical original R9 marker visible; selected role adapters from Czech source volume `cviwpryzao` copied into `/workspace-global/hephaestus/runtime-models/role-mastery-v1/<role>/selected-adapter-<hash>.tar.gz` and SHA-256 verified against the frozen `configs/models/hephaestus_role_model_stack_v1.json` registry.
- **Five adapter tests:** controller, diagnosis, planner, evaluator and judge original LoRA archive checks passed; `safetensors` was opened and an actual stored tensor loaded for each adapter on CPU. Does *not* prove the five base-models all generated tokens on GPU.
- **Runtime state:** active SQLite on Pod-local POSIX storage; initialized infrastructure CLI and health checks passed. Independent cold SQLite backup stored at `/workspace-state/hephaestus/runtime/boot-43426990148/infrastructure-cold-backup.sqlite3` on original Czech regional Network Volume; reported SHA-256 `c44a7b52a667326ddff2e7c543360130b4d34e04020e3583e6f43a339b8bd262`. Demo state additionally copied to regional cold storage; historical runs not imported or rewritten.
- **Operator console:** startup and `GET /healthz` verified at `http://127.0.0.1:8765` **inside the Pod only**, not bound to 0.0.0.0 or publicly routed. This console is for bounded inspection and narrow governed actions, not generic autonomous training.
- **CPU lifecycle:** 3-hour post-boot GitHub Actions lease with exact Pod termination expected after lease expires. The branch's one-off workflow should not be merged into `main` unchanged.

## Real CUDA test

- **GitHub Actions [#43427869731](https://github.com/kys928/Hephaestus/actions/runs/43427869731): success (GPU inference and independent cleanup jobs both passed)**.
- GPU Pod `uxcwzuepisvip0`, NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), secure cloud; observed rate **$2.69/hour**, temporarily mounted Global at `/workspace-global`.
- Original Controller pinned base `Qwen/Qwen3-4B-Instruct-2507` revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Loaded registered original Controller adapter archive SHA-256 `62ae757aad46191486c09acf87e164972434b19834f4401f9dd7f3e8bb65972e` from Global; applied PEFT LoRA; generated **8 actual new CUDA tokens** without sampling. Token-sequence SHA-256 `98c0395a22a78c32e82fafd26323eac6f2db1bbbd3b419d681b48b04d98e8e6e`.
- Worker reported `base_and_lora_loaded_on_cuda=true` and `status=COMPLETE` in private S3 status `s3://cviwpryzao/hephaestus/gpu-smoke-status/43427869731.json`. Controller accepted it; exact GPU Pod deletion requested and a separate cleanup job confirmed it **already absent**.

## Deliberate limitations and follow-up

1. CPU Pod may be terminated automatically three hours after its boot job; no permanent production deployment or public console URL is claimed.
2. Cold backup is a technical operational backup, **not** fully admitted persistent-research campaign storage with independent witness/fencing. Active durable research SQLite must meet the project local-storage admission policy and separate external custody.
3. No current run tested live CUDA token generation for the other four specialist base models, active role routing through the full eight-phase pipeline, full old campaign restore, protected eval migration, or new training.
4. No unrelated Pod, historical Czech volume, or existing evidence archive was deleted. Avoid merging branch-local paid one-off CI until reviewed.
5. CPU Pod remains available for the documented lease as of the last provider status check; verify current status before relying on it.

**Conclusion:** A real Hephaestus application boot on fresh RunPod storage passed, all five original PEFT adapters were loaded and hashed, and a separate actual original Controller base+LoRA CUDA inference passed. Additional production integration remains a separate scope.
