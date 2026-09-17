#!/usr/bin/env python3
"""Validate Diagnostic Scaling Recovery V1 without creating any paid compute."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json"
ORIGINAL_PATH = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_v1.json"
LAUNCH_MARKER_REL = Path("configs/experiments/diagnostic_scaling_recovery_v1.launch.json")
OUT_PATH = ROOT / "diagnostic_scaling_recovery_validation.json"

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
    "target_policy", "router_trainable",
]
EXPECTED_RUNTIME = {
    "transformers": "5.17.0",
    "accelerate": "1.15.0",
    "safetensors": "0.8.0",
    "huggingface_hub": "1.31.0",
    "peft": "0.20.0",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(contract: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    if contract.get("protocol_id") != "hephaestus_diagnostic_scaling_recovery_v1" or contract.get("protocol_version") != 1:
        raise ValueError("unexpected recovery protocol identity")
    recovery = contract.get("recovery_of", {})
    if recovery.get("protocol_id") != "hephaestus_diagnostic_scaling_v1" or recovery.get("original_evidence_is_immutable") is not True:
        raise ValueError("recovery provenance must preserve original evidence")
    if recovery.get("reinterpret_original_scores_as_recovery_trigger_not_scaling_result") is not True:
        raise ValueError("invalid original-result interpretation policy")

    original = load(root / contract["sources"]["original_scaling_protocol_path"])
    topology = load(root / contract["sources"]["frozen_topology_path"])
    semantic = load(root / contract["sources"]["frozen_semantic_pack_path"])
    if topology.get("generation", {}).get("max_new_tokens") != 256:
        raise ValueError("frozen topology budget drifted")
    if semantic.get("decoding_config", {}).get("max_new_tokens") != 96:
        raise ValueError("frozen semantic budget drifted")
    if semantic.get("content_hash") != contract["sources"]["source_semantic_content_hash"]:
        raise ValueError("frozen semantic pack identity drifted")
    if contract["sources"]["source_training_dataset_sha256"] != original["sources"]["source_training_dataset_sha256"]:
        raise ValueError("training dataset identity drifted")

    if contract.get("roles") != {"primary": ["diagnosis"], "calibration": ["controller"]}:
        raise ValueError("role scope drifted")
    drift = {
        key: (original["training"].get(key), contract["training"].get(key))
        for key in FIXED_TRAINING_KEYS
        if original["training"].get(key) != contract["training"].get(key)
    }
    if drift:
        raise ValueError(f"recovery training geometry drifted: {drift}")

    observed = [
        (row.get("model_id"), row.get("revision"), row.get("model_type"), str(row.get("license", "")).lower())
        for row in contract.get("candidates", [])
    ]
    if observed != EXPECTED_CANDIDATES:
        raise ValueError("candidate identities/revisions drifted")
    if any(not re.fullmatch(r"[0-9a-f]{40}", row[1] or "") for row in observed):
        raise ValueError("candidate revisions must remain immutable 40-hex SHAs")

    lanes = contract.get("evaluation_lanes", {})
    fixed = lanes.get("fixed_non_thinking", {})
    reasoning = lanes.get("reasoning_aware", {})
    if fixed.get("topology_max_new_tokens") != 256 or fixed.get("semantic_max_new_tokens") != 96:
        raise ValueError("fixed non-thinking lane must preserve frozen output budgets")
    if fixed.get("do_sample") is not False or fixed.get("adaptive_budget") is not False:
        raise ValueError("fixed non-thinking lane must remain deterministic and fixed-budget")
    if fixed.get("score_incomplete_final_answer") is not True:
        raise ValueError("fixed lane must count truncation as a fixed-budget failure")
    if fixed.get("incomplete_interpretation") != "fixed_budget_efficiency_failure_not_general_capability_failure":
        raise ValueError("fixed lane truncation interpretation drifted")
    if fixed.get("ineligible_policy") != "not_applicable_not_zero":
        raise ValueError("fixed non-thinking ineligible policy drifted")
    if reasoning.get("adaptive_budget") is not True or reasoning.get("score_incomplete_final_answer") is not False:
        raise ValueError("reasoning-aware lane must retry incomplete answers rather than score them")
    if reasoning.get("budget_exhaustion_policy") != "inconclusive_reasoning_budget_exhausted_not_floor_score":
        raise ValueError("reasoning budget exhaustion policy drifted")
    if reasoning.get("require_complete_schema_before_scoring_topology") is not True:
        raise ValueError("reasoning topology samples require complete schema before scoring")
    if fixed.get("cross_lane_ranking_allowed") is not False or reasoning.get("cross_lane_ranking_allowed") is not False:
        raise ValueError("cross-lane ranking is forbidden")

    candidates = {row["model_id"]: row for row in contract["candidates"]}
    expected_fixed = {
        "Qwen/Qwen3-14B": True,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": False,
        "zai-org/GLM-4.7-Flash": True,
        "Qwen/Qwen3-30B-A3B-Thinking-2507": False,
    }
    for model_id, eligible in expected_fixed.items():
        row = candidates[model_id]
        fixed_cfg = row.get("fixed_non_thinking", {})
        if fixed_cfg.get("eligible") is not eligible:
            raise ValueError(f"fixed lane eligibility drifted for {model_id}")
        if eligible and fixed_cfg.get("chat_template_kwargs") != {"enable_thinking": False}:
            raise ValueError(f"fixed lane must use documented hard non-thinking switch for {model_id}")
        if not eligible and not fixed_cfg.get("reason"):
            raise ValueError(f"fixed lane ineligibility must be explicit for {model_id}")

        aware = row.get("reasoning_aware", {})
        if aware.get("eligible") is not True:
            raise ValueError(f"reasoning-aware lane missing for {model_id}")
        ladder = aware.get("topology_token_ladder")
        semantic_ladder = aware.get("semantic_token_ladder")
        if not isinstance(ladder, list) or not ladder or ladder != sorted(set(ladder)):
            raise ValueError(f"invalid reasoning token ladder for {model_id}")
        if not isinstance(semantic_ladder, list) or not semantic_ladder or semantic_ladder != sorted(set(semantic_ladder)):
            raise ValueError(f"invalid semantic token ladder for {model_id}")
        if ladder[0] <= 256 or semantic_ladder[0] <= 96:
            raise ValueError(f"reasoning-aware lane does not exceed frozen truncating budget for {model_id}")
        decoding = aware.get("decoding", {})
        if decoding.get("do_sample") is not True or float(decoding.get("temperature", 0)) <= 0:
            raise ValueError(f"reasoning-aware decoding must sample for {model_id}")

    if candidates["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]["reasoning_aware"].get("assistant_prefill") != "<think>\n":
        raise ValueError("DeepSeek reasoning prefill drifted")
    for model_id, row in candidates.items():
        aware = row["reasoning_aware"]
        if aware["topology_token_ladder"][-1] < 32768:
            raise ValueError(f"reasoning-aware topology lane lacks 32K terminal budget for {model_id}")
        if aware["semantic_token_ladder"][-1] < 8192:
            raise ValueError(f"reasoning-aware semantic lane lacks 8K terminal budget for {model_id}")

    execution = contract.get("execution", {})
    if execution.get("runtime_dependencies") != EXPECTED_RUNTIME:
        raise ValueError("runtime dependency lock drifted")
    if execution.get("network_volume_attached") is not False or execution.get("persistent_model_cache_allowed") is not False:
        raise ValueError("recovery must remain ephemeral")
    if execution.get("pod_exit_without_terminal_policy") != "fail_immediately_preserve_pod_status_and_log_snapshot":
        raise ValueError("silent pod-exit policy drifted")
    silent_timeout = int(execution.get("silent_container_start_timeout_seconds", 0))
    if not 300 <= silent_timeout <= 1800:
        raise ValueError("silent container-start watchdog must be bounded between 5 and 30 minutes")

    governance = contract.get("governance", {})
    paid_allowed = governance.get("paid_launch_allowed_now")
    if paid_allowed not in (True, False):
        raise ValueError("paid launch state must be an explicit boolean")
    if governance.get("paid_launch_requires_new_explicit_user_go") is not True:
        raise ValueError("paid launch must always require explicit user go")
    authorization_state = "blocked_pre_user_go"
    if paid_allowed:
        marker_path = root / LAUNCH_MARKER_REL
        if not marker_path.is_file():
            raise ValueError("paid launch requires committed authorization marker")
        marker = load(marker_path)
        if marker.get("authorized") is not True:
            raise ValueError("paid launch authorization marker is not authorized")
        if marker.get("protocol_id") != "hephaestus_diagnostic_scaling_recovery_v1":
            raise ValueError("paid launch authorization marker protocol mismatch")
        if marker.get("authorization_source") != "user_directive":
            raise ValueError("paid launch authorization source drifted")
        if not str(marker.get("authorization_text", "")).strip():
            raise ValueError("paid launch authorization text is missing")
        if marker.get("promotion_allowed") is not False or marker.get("lineage_mutation_allowed") is not False:
            raise ValueError("paid launch marker may not authorize promotion or lineage mutation")
        if governance.get("approval_source") != "user_directive_2026-09-17_paid_recovery_launch":
            raise ValueError("paid launch approval source drifted")
        authorization_state = "authorized_by_user"
    for key in ("promotion_allowed", "lineage_mutation_allowed", "frozen_eval_mutation_allowed", "source_dataset_mutation_allowed", "model_revision_substitution_allowed"):
        if governance.get(key) is not False:
            raise ValueError(f"governance boundary drifted: {key}")

    return {
        "status": "recovery_contract_valid",
        "protocol_id": contract["protocol_id"],
        "contract_sha256": sha256(CONTRACT_PATH) if CONTRACT_PATH.exists() else None,
        "original_protocol_sha256": sha256(ORIGINAL_PATH) if ORIGINAL_PATH.exists() else None,
        "candidate_count": len(observed),
        "fixed_non_thinking_eligible": [model_id for model_id, eligible in expected_fixed.items() if eligible],
        "reasoning_aware_eligible": [row[0] for row in observed],
        "paid_launch_allowed_now": bool(paid_allowed),
        "authorization_state": authorization_state,
        "frozen_eval_mutated": False,
    }


def validate_remote(contract: dict[str, Any]) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    rows: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        model_id = candidate["model_id"]
        revision = candidate["revision"]
        info = api.model_info(model_id, revision=revision, files_metadata=False)
        if str(info.sha) != revision:
            raise RuntimeError(f"remote revision mismatch for {model_id}")
        config_path = Path(hf_hub_download(model_id, "config.json", revision=revision))
        config = load(config_path)
        if config.get("model_type") != candidate["model_type"]:
            raise RuntimeError(f"remote model_type mismatch for {model_id}: {config.get('model_type')}")

        fixed_cfg = candidate["fixed_non_thinking"]
        template_support = None
        if fixed_cfg["eligible"]:
            template_text = ""
            try:
                template_text = Path(hf_hub_download(model_id, "chat_template.jinja", revision=revision)).read_text(encoding="utf-8")
            except Exception:
                try:
                    tokenizer_cfg = load(Path(hf_hub_download(model_id, "tokenizer_config.json", revision=revision)))
                    template_text = str(tokenizer_cfg.get("chat_template") or "")
                except Exception:
                    template_text = ""
            template_support = "enable_thinking" in template_text
            if not template_support:
                raise RuntimeError(f"documented fixed non-thinking candidate lacks enable_thinking template switch: {model_id}")

        rows.append({
            "model_id": model_id,
            "revision": revision,
            "model_type": candidate["model_type"],
            "remote_revision_verified": True,
            "fixed_non_thinking_eligible": bool(fixed_cfg["eligible"]),
            "enable_thinking_template_support": template_support,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    contract = load(CONTRACT_PATH)
    evidence = validate_contract(contract)
    if args.remote:
        evidence["remote_models"] = validate_remote(contract)
        evidence["status"] = "recovery_contract_remote_validated"
    OUT_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DIAGNOSTIC_SCALING_RECOVERY_VALIDATION_JSON " + json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
