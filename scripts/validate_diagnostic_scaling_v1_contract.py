#!/usr/bin/env python3
"""Validate the frozen Diagnostic Scaling V1 experiment contract without GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json"
OUT_PATH = ROOT / "diagnostic_scaling_contract_validation.json"
EXPECTED_CANDIDATES = [
    ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", "qwen3", "apache-2.0"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "1df8507178afcc1bef68cd8c393f61a886323761", "qwen2", "mit"),
    ("zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67", "glm4_moe_lite", "mit"),
    ("Qwen/Qwen3-30B-A3B-Thinking-2507", "144afc2f379b542fdd4e85a1fcd5e1f79112d95d", "qwen3_moe", "apache-2.0"),
]
FIXED_TRAINING_KEYS = [
    "method", "peft_version", "precision", "rank", "alpha", "dropout", "bias",
    "learning_rate", "weight_decay", "micro_batch_size", "gradient_accumulation_steps",
    "max_seq_length", "warmup_ratio", "max_grad_norm", "optimizer", "shuffle_seed",
    "model_seed", "gradient_checkpointing", "train_only_assistant_tokens",
    "examples_per_role", "max_epochs", "expected_optimizer_steps_per_epoch", "dose_optimizer_steps",
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(contract: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    if contract.get("protocol_id") != "hephaestus_diagnostic_scaling_v1" or contract.get("protocol_version") != 1:
        raise ValueError("unexpected Diagnostic Scaling protocol identity")
    if contract.get("scientific_variable") != "candidate_model_identity_and_reasoning_family_under_fixed_lora_dose_geometry":
        raise ValueError("scientific variable drifted")

    source_path = root / contract["sources"]["base_distance_protocol_path"]
    topology_path = root / contract["sources"]["frozen_topology_path"]
    semantic_path = root / contract["sources"]["frozen_semantic_pack_path"]
    source = _load(source_path)
    topology = _load(topology_path)
    semantic = _load(semantic_path)

    if contract["roles"] != {"primary": ["diagnosis"], "calibration": ["controller"]}:
        raise ValueError("role scope must remain diagnosis primary plus controller calibration")
    if not {"diagnosis", "controller"}.issubset(set(topology.get("roles", []))):
        raise ValueError("frozen topology no longer contains required roles")
    if semantic.get("content_hash") != contract["sources"]["source_semantic_content_hash"]:
        raise ValueError("semantic pack identity drifted")
    if source["sources"]["training_dataset_sha256"] != contract["sources"]["source_training_dataset_sha256"]:
        raise ValueError("source training dataset identity drifted")

    train = contract["training"]
    source_train = source["training"]
    drift = {key: (source_train.get(key), train.get(key)) for key in FIXED_TRAINING_KEYS if source_train.get(key) != train.get(key)}
    if drift:
        raise ValueError(f"fixed Test-3 training geometry drifted: {drift}")
    if train.get("dose_optimizer_steps") != [3, 6, 12, 24, 36, 48]:
        raise ValueError("dose ladder drifted")
    source_targets = source_train["target_module_suffixes"]
    for family in ("qwen3", "qwen2"):
        if train["target_policy"].get(family) != source_targets:
            raise ValueError(f"{family} target surface must match Test-3 exactly")

    candidates = contract.get("candidates", [])
    observed = [(c.get("model_id"), c.get("revision"), c.get("model_type"), str(c.get("license", "")).lower()) for c in candidates]
    if observed != EXPECTED_CANDIDATES:
        raise ValueError(f"candidate identities/revisions drifted: {observed}")
    if len({row[1] for row in observed}) != len(observed) or any(not re.fullmatch(r"[0-9a-f]{40}", row[1]) for row in observed):
        raise ValueError("every candidate must use a unique immutable 40-hex revision")
    for candidate in candidates:
        targets = train["target_policy"].get(candidate["model_type"])
        if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets):
            raise ValueError(f"invalid target policy for {candidate['model_id']}")

    projection = contract["evaluation"]["reasoning_projection"]
    if projection != {
        "preserve_raw_generation": True,
        "score_projection": "text_after_last_reasoning_close_delimiter_else_raw",
        "reasoning_close_delimiters": ["</think>"],
        "projection_applies_identically_to_baseline_and_all_doses": True,
    }:
        raise ValueError("reasoning projection contract drifted")

    execution = contract["execution"]
    if execution.get("network_volume_attached") is not False or execution.get("persistent_model_cache_allowed") is not False:
        raise ValueError("Diagnostic Scaling must remain ephemeral with no persistent model cache")
    if execution.get("one_model_per_pod") is not True or int(execution.get("max_parallel_pods", 0)) != 1:
        raise ValueError("bounded one-model-per-pod execution drifted")
    if int(execution.get("container_disk_gb", 0)) < 250:
        raise ValueError("ephemeral disk budget is too small for the frozen cohort")

    governance = contract["governance"]
    forbidden_true = ["promotion_allowed", "lineage_mutation_allowed", "frozen_eval_mutation_allowed", "source_dataset_mutation_allowed", "model_revision_substitution_allowed"]
    if governance.get("training_is_experimentally_approved") is not True or any(governance.get(key) is not False for key in forbidden_true):
        raise ValueError("governance boundary drifted")

    return {
        "status": "local_contract_valid",
        "protocol_id": contract["protocol_id"],
        "contract_sha256": _sha256(CONTRACT_PATH) if CONTRACT_PATH.exists() else None,
        "base_distance_protocol_sha256": _sha256(source_path),
        "candidate_count": len(candidates),
        "roles": ["diagnosis", "controller"],
        "dose_optimizer_steps": train["dose_optimizer_steps"],
        "ephemeral_only": True,
        "promotion_allowed": False,
    }


def validate_remote_models(contract: dict[str, Any]) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    rows: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        model_id = candidate["model_id"]
        revision = candidate["revision"]
        info = api.model_info(model_id, revision=revision, files_metadata=False)
        if info.sha != revision:
            raise RuntimeError(f"remote immutable revision mismatch for {model_id}: {info.sha}")
        config_path = Path(hf_hub_download(model_id, "config.json", revision=revision))
        config = _load(config_path)
        if config.get("model_type") != candidate["model_type"]:
            raise RuntimeError(f"remote model_type mismatch for {model_id}: {config.get('model_type')}")
        card = getattr(info, "card_data", None)
        remote_license = str(getattr(card, "license", "") or "").lower()
        if remote_license and remote_license != candidate["license"].lower():
            raise RuntimeError(f"remote license mismatch for {model_id}: {remote_license}")
        rows.append({
            "model_id": model_id,
            "revision": revision,
            "model_type": config.get("model_type"),
            "license": remote_license or candidate["license"].lower(),
            "remote_revision_verified": True,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="also resolve immutable Hugging Face revisions/configs")
    args = parser.parse_args()
    contract = _load(CONTRACT_PATH)
    evidence = validate_contract(contract)
    if args.remote:
        evidence["remote_models"] = validate_remote_models(contract)
        evidence["status"] = "contract_validated"
    OUT_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DIAGNOSTIC_SCALING_CONTRACT_JSON " + json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
