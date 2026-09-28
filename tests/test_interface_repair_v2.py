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
    # Keep original decoding envelopes; V2 exits early only after a complete contract JSON exists.
    assert v2["roles"] == v1["roles"]


def test_v2_compute_efficiency_and_governance_are_frozen() -> None:
    cfg = load(V2)
    marker = load(MARKER)
    ex = cfg["execution"]
    assert cfg["training"]["dynamic_padding"] is True
    assert ex["gpu_type_ids"] == ["NVIDIA A40"]
    assert ex["parallel_training_roles"] == 1
    assert ex["minimum_gpu_memory_gib"] == 44
    assert ex["structured_json_early_stop"] is True
    assert ex["evaluation_shard_size"] == 32
    assert ex["baseline_cache_enabled"] is True
    assert ex["hard_wall_seconds_per_role"] == 3600
    assert ex["hard_wall_seconds_preflight"] == 600
    assert ex["hard_wall_seconds_stack"] == 14400
    assert ex["max_estimated_total_usd_preflight"] == 0.10
    assert ex["max_estimated_total_usd_stack"] == 2.50
    assert cfg["preflight"]["maximum_projected_role_seconds"] == 3600
    assert cfg["preflight"]["maximum_projected_role_seconds"] <= ex["hard_wall_seconds_per_role"]
    assert cfg["governance"]["paid_preflight_allowed"] is False
    assert cfg["governance"]["paid_full_launch_allowed"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert marker["preflight_authorized"] is False
    assert marker["full_launch_authorized"] is False


def test_v2_launcher_mounts_volume_and_reuses_one_pod() -> None:
    launcher = import_file(LAUNCHER, "interface_repair_v2_launcher")
    cfg = load(V2)
    rendered = launcher.request_body(cfg, "deadbeef", "dry-run", "full", real_env=False)
    assert rendered["gpuTypeIds"] == ["NVIDIA A40"]
    assert rendered["gpuCount"] == 1
    assert rendered["networkVolumeId"] == "<volume>"
    assert rendered["volumeMountPath"] == "/workspace"
    shell = rendered["dockerStartCmd"][2]
    assert "planner evaluator judge controller" in shell
    assert "run_interface_repair_live_v2.py" in shell
    assert "/workspace/hephaestus-cache/huggingface" in shell


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
    assert cfg["governance"]["paid_full_launch_allowed"] is False
