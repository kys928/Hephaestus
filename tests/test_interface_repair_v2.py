from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs/experiments/hephaestus_interface_repair_v1.json"
V2 = ROOT / "configs/experiments/hephaestus_interface_repair_v2.json"
MARKER = ROOT / "configs/experiments/interface_repair_v2.launch.json"
LAUNCHER = ROOT / "scripts/launch_interface_repair_v2.py"
RUNNER = ROOT / "scripts/run_interface_repair_v2.py"

EXPECTED_GPU_FALLBACK = [
    "NVIDIA A40",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_preserves_frozen_science() -> None:
    v1 = load(V1)
    v2 = load(V2)
    assert v2["protocol_id"] == v1["protocol_id"]
    assert v2["execution_path_version"] == 2
    assert v2["pack"] == v1["pack"]
    assert v2["source_phase_i_stack"] == v1["source_phase_i_stack"]
    assert v2["source_phase_ii_run"] == v1["source_phase_ii_run"]
    assert v2["frozen_diagnosis"] is True
    assert v2["trainable_roles"] == v1["trainable_roles"]
    for key in (
        "method", "precision", "seed", "lora_rank", "lora_alpha", "lora_dropout",
        "optimizer", "learning_rate", "weight_decay", "micro_batch_size",
        "gradient_accumulation_steps", "max_sequence_length", "gradient_checkpointing",
        "max_grad_norm", "assistant_only_loss", "optimizer_steps",
    ):
        assert v2["training"][key] == v1["training"][key]
    assert v2["certification"] == v1["certification"]
    assert v2["live_certification"] == v1["live_certification"]
    assert v2["roles"] == v1["roles"]


def test_v2_compute_efficiency_and_governance_are_frozen() -> None:
    cfg = load(V2)
    marker = load(MARKER)
    ex = cfg["execution"]
    assert cfg["training"]["dynamic_padding"] is True
    assert ex["preferred_gpu_type_id"] == "NVIDIA A40"
    assert ex["gpu_type_ids"] == EXPECTED_GPU_FALLBACK
    assert ex["compatible_fallback_gpu_type_ids"] == EXPECTED_GPU_FALLBACK[1:]
    assert ex["capacity_fallback_policy"] == "preferred_then_cheapest_compatible"
    assert ex["parallel_training_roles"] == 1
    assert ex["minimum_gpu_memory_gib"] == 44
    assert ex["structured_json_early_stop"] is True
    assert ex["evaluation_shard_size"] == 32
    assert ex["baseline_cache_enabled"] is True
    assert ex["hard_wall_seconds_per_role"] == 3600
    assert ex["hard_wall_seconds_preflight"] == 1200
    assert ex["hard_wall_seconds_preflight_benchmark"] == 600
    assert ex["hard_wall_seconds_preflight_bootstrap"] == 600
    assert ex["hard_wall_seconds_preflight_total"] == 1200
    assert ex["hard_wall_seconds_stack"] == 14400
    assert ex["max_hourly_usd_per_pod"] == 2.20
    assert ex["max_estimated_total_usd_preflight"] == 0.74
    assert ex["max_estimated_total_usd_stack"] == 2.50
    assert ex["capacity_retry_attempts"] == 8
    assert ex["capacity_retry_seconds"] == 15
    assert ex["persistent_venv_cache"].startswith("/workspace/")
    assert cfg["preflight"]["maximum_projected_role_seconds"] == 3600
    assert cfg["preflight"]["maximum_projected_role_seconds"] <= ex["hard_wall_seconds_per_role"]
    assert cfg["governance"]["paid_preflight_allowed"] is False
    assert cfg["governance"]["paid_full_launch_allowed"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert marker["preflight_authorized"] is False
    assert marker["full_launch_authorized"] is False


def test_v2_launcher_mounts_volume_reuses_one_pod_and_caches_environment() -> None:
    launcher = import_file(LAUNCHER, "interface_repair_v2_launcher")
    cfg = load(V2)
    rendered = launcher.request_body(cfg, "deadbeef", "dry-run", "full", real_env=False)
    assert rendered["gpuTypeIds"] == EXPECTED_GPU_FALLBACK
    assert rendered["gpuCount"] == 1
    assert rendered["networkVolumeId"] == "<volume>"
    assert rendered["volumeMountPath"] == "/workspace"
    shell = rendered["dockerStartCmd"][2]
    assert "planner evaluator judge controller" in shell
    assert "run_interface_repair_live_v2.py" in shell
    assert "/workspace/hephaestus-cache/huggingface" in shell
    assert cfg["execution"]["persistent_venv_cache"] in shell
    assert ".hephaestus-ready" in shell


def test_v2_launcher_is_stage_aware_and_reports_safe_pod_metadata() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "PRELIGHT_BENCHMARK_STAGES" in text
    assert "preflight_evaluation_started" in text
    assert "hard_wall_seconds_preflight_benchmark" in text
    assert "hard_wall_seconds_preflight_bootstrap" in text
    assert "INTERFACE_REPAIR_V2_PROGRESS_JSON" in text
    assert "INTERFACE_REPAIR_V2_POD_METADATA_JSON" in text
    assert "safe_pod_metadata" in text


def test_v2_runner_contains_contract_stop_shards_and_dynamic_lengths() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "class ContractComplete" in text
    assert "contract_early_stop" in text
    assert "put_jsonl" in text
    assert "baseline_cache_hit" in text
    assert '"input_ids": torch.tensor([f])' in text
    assert '"token_slots": len(f)' in text
    assert "padded_training_token_slots" not in text


def test_preflight_is_small_and_cannot_auto_launch_full_run() -> None:
    cfg = load(V2)
    assert cfg["preflight"]["role"] == "planner"
    assert cfg["preflight"]["evaluation_cases"] == 8
    assert cfg["preflight"]["optimizer_steps"] == 5
    assert cfg["preflight"]["optimizer_steps"] * cfg["training"]["gradient_accumulation_steps"] == 40
    assert cfg["preflight"]["maximum_projected_role_seconds"] == 3600
    assert cfg["preflight"]["maximum_projected_role_seconds"] <= cfg["execution"]["hard_wall_seconds_per_role"]
    assert cfg["execution"]["hard_wall_seconds_preflight_benchmark"] == 600
    assert cfg["governance"]["paid_full_launch_allowed"] is False
