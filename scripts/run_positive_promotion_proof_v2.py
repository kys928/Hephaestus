#!/usr/bin/env python3
"""Second real-model promotion wave using stronger 7B Apache-2.0 candidates.

The first wave durably rejected Qwen2.5-0.5B-Instruct and 1.5B-Instruct on
unchanged hard semantic gates. This adapter reuses the exact production-loop
proof implementation while changing only the governed model-candidate set and
making snapshot acquisition weight-format selective.
"""
from __future__ import annotations

import re
from pathlib import Path

import run_positive_promotion_proof as proof

PRIOR_REJECTION_EVIDENCE = (
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-33957215257/cycles/cycle-01/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-33957215257/cycles/cycle-02/cycle_summary.json",
)

proof.CANDIDATES = (
    {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "ddb6b63a3f61ac6c557eb55619b0a5e125129302",
        "license": "apache-2.0",
        "judge_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "judge_revision": "582efe62d7cfafd242bffca71ecbde1bcecc1bcc",
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_REJECTION_EVIDENCE),
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "license": "apache-2.0",
        "judge_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "judge_revision": "582efe62d7cfafd242bffca71ecbde1bcecc1bcc",
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_REJECTION_EVIDENCE),
    },
)


def materialize_model_minimal(
    *,
    proof_root: Path,
    model_id: str,
    revision: str,
    expected_license: str,
) -> dict[str, object]:
    """Download only the Transformers safetensors/tokenizer runtime surface."""
    from huggingface_hub import HfApi, snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"model revision is not immutable: {model_id}@{revision}")
    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    observed_revision = str(info.sha or "")
    if observed_revision != revision:
        raise RuntimeError(f"Hub revision drift for {model_id}: {observed_revision} != {revision}")
    observed_license = str(getattr(getattr(info, "card_data", None), "license", "") or "").lower()
    if observed_license != expected_license.lower():
        raise RuntimeError(
            f"model license drift for {model_id}@{revision}: {observed_license!r} != {expected_license!r}"
        )

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=proof_root / "hf_cache",
            local_files_only=False,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "model-*.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
            ],
        )
    ).resolve()
    executable = [
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".sh", ".exe", ".dll", ".so", ".dylib"}
    ]
    if executable:
        raise RuntimeError(f"remote executable/code files are not admitted: {executable}")

    components: dict[str, str] = {}
    byte_size = 0
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        components[relative] = proof.sha_file(path)
        byte_size += path.stat().st_size
    if not components or not any(name.endswith(".safetensors") for name in components):
        raise RuntimeError(f"model snapshot has no admitted safetensors weights: {model_id}@{revision}")

    manifest_payload: dict[str, object] = {
        "manifest_version": "external-model-snapshot.v1",
        "acquisition_profile": "transformers_safetensors_minimal_v1",
        "model_id": model_id,
        "provider": "huggingface",
        "requested_revision": revision,
        "resolved_revision": observed_revision,
        "license": observed_license,
        "trust_remote_code": False,
        "snapshot_path": str(snapshot),
        "component_count": len(components),
        "byte_size": byte_size,
        "components": components,
    }
    manifest_payload["manifest_hash"] = proof.canonical_hash(components)
    manifest_path = proof_root / "model_manifests" / proof.slug(model_id) / revision / "snapshot_manifest.json"
    proof.atomic_json(manifest_path, manifest_payload)
    manifest_payload["manifest_ref"] = str(manifest_path)
    return manifest_payload


proof.materialize_model = materialize_model_minimal

if __name__ == "__main__":
    raise SystemExit(proof.main())
