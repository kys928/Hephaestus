"""Metadata admission for immutable model waves; GPU proof remains separate."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from hephaestus.schemas.discovery_contract import ModelCandidate

METADATA_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "tekken.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "chat_templates/*.jinja", "processor_config.json",
    "README.md", "LICENSE", "LICENSE.txt", "NOTICE", "SYSTEM_PROMPT.txt",
]
RUNTIME_PATTERNS = METADATA_PATTERNS + ["model.safetensors", "model-*.safetensors"]
PERMISSIVE_LICENSES = frozenset({"apache-2.0", "mit"})


def validate_identity(spec: dict[str, Any]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(spec.get("model_id", ""))):
        raise ValueError("model identity must be an explicit owner/repository")
    if not re.fullmatch(r"[0-9a-f]{40}", str(spec.get("revision", ""))):
        raise ValueError("model revision must be an immutable 40-character Hub commit")
    if spec.get("license") not in PERMISSIVE_LICENSES:
        raise ValueError("model license is outside the permissive admission allowlist")


def validate_hub_metadata(spec: dict[str, Any], *, revision: str, license_id: str,
                          config: dict[str, Any]) -> None:
    validate_identity(spec)
    if revision != spec["revision"]:
        raise ValueError("resolved model revision differs from admitted immutable identity")
    if license_id.lower() != spec["license"]:
        raise ValueError("immutable model license disagrees with the admission record")
    if config.get("architectures") != [spec["architecture"]] or config.get("model_type") != spec["model_type"]:
        raise ValueError("model architecture disagrees with the admission record")
    if config.get("quantization_config") or config.get("text_config", {}).get("quantization_config"):
        raise ValueError("quantized source weights are forbidden in this model wave")
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype != spec["source_dtype"]:
        raise ValueError(f"source weight dtype drift: {dtype!r}")


def inspect_immutable_model(spec: dict[str, Any], cache_root: Path) -> ModelCandidate:
    """Read pinned metadata and exercise the native tokenizer, without model weights.

    The resulting record admits a candidate to GPU preflight, not to certification.
    It deliberately leaves smoke_test unknown until real execution supplies evidence.
    """
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    import transformers

    validate_identity(spec)
    info = HfApi().model_info(spec["model_id"], revision=spec["revision"], files_metadata=True)
    license_id = str(getattr(getattr(info, "card_data", None), "license", "") or "").lower()
    snapshot = Path(snapshot_download(
        repo_id=spec["model_id"], revision=spec["revision"], cache_dir=cache_root,
        allow_patterns=METADATA_PATTERNS,
    ))
    raw_config = json.loads((snapshot / "config.json").read_text())
    validate_hub_metadata(spec, revision=str(info.sha), license_id=license_id, config=raw_config)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    loaders = {"AutoModelForCausalLM": AutoModelForCausalLM,
               "AutoModelForImageTextToText": AutoModelForImageTextToText}
    if spec["loader"] not in loaders or type(config) not in loaders[spec["loader"]]._model_mapping:
        raise ValueError("architecture has no native loader in the installed Transformers release")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if not getattr(tokenizer, "chat_template", None) or tokenizer.eos_token_id is None:
        raise ValueError("native chat template and EOS token are required")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Describe a paper clip."}],
        tokenize=False, add_generation_prompt=True,
    )
    if not rendered or not tokenizer(rendered)["input_ids"]:
        raise ValueError("native chat-template/tokenizer preflight returned empty input")
    index = snapshot / "model.safetensors.index.json"
    parameters = None
    if index.is_file():
        index_payload = json.loads(index.read_text())
        size = index_payload.get("metadata", {}).get("total_size")
        if isinstance(size, int) and size > 0 and size % 2 == 0:
            parameters = size // 2
    if parameters is None:
        tensor_info = getattr(info, "safetensors", None)
        parameters = (getattr(tensor_info, "total", None) if not isinstance(tensor_info, dict)
                      else tensor_info.get("total"))
    if not isinstance(parameters, int) or not 0 < parameters <= spec["parameter_ceiling"]:
        raise ValueError(f"unverified or excessive parameter count: {parameters!r}")
    weight_gib = parameters * 2 / (1024 ** 3)
    if not weight_gib + 4 < spec["estimated_runtime_memory_gib"] <= 44:
        raise ValueError("FP16 memory estimate has insufficient headroom or exceeds the GPU budget")
    hashes = {p.relative_to(snapshot).as_posix(): "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob("*")) if p.is_file()}
    text_config = raw_config.get("text_config", raw_config)
    return ModelCandidate(
        candidate_id=f"huggingface:{spec['model_id']}@{spec['revision']}",
        provider_id="huggingface", model_id=spec["model_id"], revision=spec["revision"],
        license=license_id, architecture_family=spec["architecture"], parameter_count=parameters,
        context_length=int(text_config["max_position_embeddings"]),
        tokenizer_ref=f"hf://models/{spec['model_id']}@{spec['revision']}",
        capabilities=["causal_lm", "chat"],
        runtime_requirements={
            "supported_backends": ["pinned_chat_template_transformers"],
            "memory_gb": spec["estimated_runtime_memory_gib"] * (1024 ** 3) / 1e9,
            "estimated_memory_gib": spec["estimated_runtime_memory_gib"],
            "weight_memory_gib": weight_gib, "dtype": "float16",
            "quantization": False, "model_offload": False, "loader": spec["loader"],
        },
        compatibility={"immutable_revision": True, "remote_code_required": False,
                       "tokenizer_preflight": True, "smoke_test": None},
        artifact_ref=f"hf://models/{spec['model_id']}@{spec['revision']}",
        evidence_refs=list(spec["evidence_refs"]),
        metadata={
            "admission_state": "metadata_admitted_runtime_pending",
            "source_dtype": spec["source_dtype"], "transformers_version": transformers.__version__,
            "metadata_component_hashes": hashes, "eos_token_id": tokenizer.eos_token_id,
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256(
                json.dumps(tokenizer.chat_template, sort_keys=True).encode()).hexdigest(),
            "selection_reason": spec["selection_reason"], "research_rank": spec["rank"],
            "runtime_generation_tested": False, "certification_state": "not_evaluated",
            "promotion_state": "not_promoted",
        },
    )
