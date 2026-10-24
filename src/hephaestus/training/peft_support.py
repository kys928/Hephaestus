"""LoRA identity checks shared by training and generation; no ML imports."""
import json
import math
from pathlib import Path

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.schemas.peft_training import PeftTrainingSpec


def validate_recipe(payload: dict, *, model_id: str, revision: str, mode: str) -> PeftTrainingSpec:
    spec = PeftTrainingSpec.from_dict(payload)
    if (type(spec.rank) is not int or spec.rank < 1 or
            not math.isfinite(spec.alpha) or spec.alpha <= 0 or
            not math.isfinite(spec.dropout) or not 0 <= spec.dropout < 1 or
            spec.bias != "none" or not isinstance(spec.target_modules, list) or not spec.target_modules or
            len(set(spec.target_modules)) != len(spec.target_modules) or
            not all(isinstance(x, str) and x.strip() for x in spec.target_modules)):
        raise ValueError("invalid LoRA rank, alpha, dropout, targets or bias")
    if mode == "lora_finetune":
        if spec.continuation_kind != "fresh" or spec.parent_adapter_ref or spec.parent_adapter_hash:
            raise ValueError("fresh LoRA recipe cannot include a continuation parent")
    elif mode == "lora_continuation":
        if spec.continuation_kind != "adapter_warm_start":
            raise ValueError("cross-run LoRA continuation must declare adapter_warm_start; optimizer is reset")
        parent = Path(spec.parent_adapter_ref or "")
        if not spec.parent_adapter_hash or directory_content_identity(parent) != spec.parent_adapter_hash:
            raise ValueError("parent adapter SHA-256 mismatch")
        validate_adapter(parent, model_id=model_id, revision=revision, spec=spec)
    else:
        raise ValueError("unsupported PEFT mode")
    return spec


def validate_adapter(path: Path, *, model_id: str, revision: str, spec: PeftTrainingSpec | None = None) -> dict:
    config = json.loads((path / "adapter_config.json").read_text())
    provenance = json.loads((path / "adapter_provenance.json").read_text())
    if (provenance.get("base_model_id") != model_id or provenance.get("base_revision") != revision or
            config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM" or
            config.get("revision") not in {None, "", revision}):
        raise ValueError("adapter base identity or type mismatch")
    base = str(config.get("base_model_name_or_path", "")).replace("\\", "/").rstrip("/")
    snapshot = "models--" + model_id.replace("/", "--") + "/snapshots/" + revision
    if base != model_id.rstrip("/") and not base.endswith("/" + snapshot):
        raise ValueError("adapter configuration names a different base model")
    if spec is not None and (config.get("r") != spec.rank or config.get("lora_alpha") != spec.alpha or
            config.get("lora_dropout") != spec.dropout or config.get("bias") != spec.bias or
            set(config.get("target_modules", [])) != set(spec.target_modules)):
        raise ValueError("continuation LoRA recipe differs from parent adapter")
    if not (path / "adapter_model.safetensors").is_file():
        raise ValueError("final adapter weights are missing")
    return provenance
