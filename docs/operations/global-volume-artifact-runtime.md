# Global Volume artifact runtime — opt-in, review required

**Status:** additive groundwork. Not integrated into production finalization or live Pod allocation. No current Hephaestus run changes storage by default.

## Boundaries

- Regional/local volume remains the source of truth for active training, optimizer state, mutable checkpoints, SQLite/WAL and locks. Preserve the existing RunPod regional S3 adapter.
- The Global Volume is object-backed and suitable for sealed read-mostly model weights, LoRA, tokenizers, datasets and evidence. It has no dependable atomic rename, POSIX locks or multi-writer safety.
- A genuinely independent backup uses a separately admitted S3/HF/other storage provider. A second copy on the same Global Volume is **not** independent.

## Configuration

Set `RUNPOD_GLOBAL_VOLUME_ID` and optionally `RUNPOD_GLOBAL_MOUNT_PATH`. GlobalVolumeConfig is separate from RunPodConfig and generates a REST v2 `mounts.global` request fragment. Confirm the actual Pod mount path: Console-created global-only Pods can mount at `/workspace`; the API may default to `/workspace-global`. Do not assume any particular path.

## Publisher and resolver

`GlobalArtifactLibrary.publish` requires a sealed file, exact SHA-256, byte size, kind and approval reference from the trusted deterministic lineage owner. The application must inject an **independent admission_verifier** callback tied to persisted policy state; merely providing an approval hash never authorizes publication. It also requires an independent backup provider with `put_file(expected_hash=...)` and `verify(ref)` (the existing S3ArtifactStore implements these methods).

Publication uses a single externally coordinated writer, writes the hash-addressed artifact, **reads it back**, writes and verifies the independent backup, checks the source again, and writes a content-hashed immutable receipt **last**. A failed write may leave an incomplete unreceipted file. The code never silently overwrites it. Do not treat `open('xb')` as a distributed lock on this filesystem. There is no auto-deletion.

`GlobalArtifactLibrary.resolve` validates the receipt and re-hashes the complete artifact. **Receipt SHA is not a signature or scientific certification.** Production worker activation must separately authenticate its expected receipt and approval from the deterministic state owner. Old files on the Global Volume, including the earlier r9 TAR archive, are not automatically admitted.

## Worker placement

`build_global_gpu_request` is pure, does not POST. It checks exact approved GPU SKU, datacenter, minimum VRAM, price ceiling, wall-clock lease, current authenticated prepaid credit and spending, approval verification and off-Pod watchdog readiness. It emits a single exact GPU/datacenter REST v2 Pod body using `mounts.global`; `verify_global_pod_mount` compares the live provider result. Deployment still requires the existing independent cost/lease governance and durable external operation journal, including ambiguity reconciliation and termination. Do not create a Pod just because a request body exists.

## Cross-region smoke and backup gate — pending live evidence

1. Wire the publisher's admission callback to real deterministic state; configure an approved separate backup target and verify its read-back.
2. Seal a frozen model/adapter/tokenizer bundle with pinned revision/manifest and publish it; verify the destination from a newly mounted Pod.
3. With fresh provider availability and a new paid-run approval, create a bounded worker in a **different data center** from EU-CZ-1. Verify the actual Pod mount, GPU and datacenter, and run `python scripts/verify_global_worker_artifact.py --receipt sha256:<receipt> --kind adapter` on it.
4. Optionally provide `--base-model`, `--adapter-dir` and `--tokenizer` paths for a bounded eight-token, **offline** PEFT inference. A bare .safetensors file is not a complete adapter directory. Do not automatically download or change selected model weights.
5. Preserve the source/receipt/result hashes and provider Pod identity as external evidence, verify backup availability after Pod teardown, and independently confirm billing cleanup. A smoke inference does not certify scientific quality.

RunPod Global Volumes may not immediately show changes made after worker start; use fresh mounts for tests. Cross-region verification and external backup have **not** yet been performed.

Track rollout in https://github.com/kys928/Hephaestus/issues/202.
