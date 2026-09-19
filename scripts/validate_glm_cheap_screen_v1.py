#!/usr/bin/env python3
"""Validate the bounded GLM cheap-screen protocol without launching paid compute."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCREEN_PATH = ROOT / "configs/experiments/hephaestus_glm_cheap_screen_v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate() -> dict[str, Any]:
    screen = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    recovery = json.loads((ROOT / screen["sources"]["recovery_contract_path"]).read_text(encoding="utf-8"))
    semantic = json.loads((ROOT / screen["sources"]["semantic_pack_path"]).read_text(encoding="utf-8"))
    topology = json.loads((ROOT / screen["sources"]["topology_pack_path"]).read_text(encoding="utf-8"))

    require(screen["protocol_id"] == "hephaestus_glm_cheap_screen_v1", "protocol id drifted")
    require(screen["scientific_scope"]["screening_only"] is True, "screen must remain screening-only")
    require(screen["scientific_scope"]["certification_claim_allowed"] is False, "screen may not certify")
    require(screen["scientific_scope"]["training_performed"] is False, "screen may not train")
    require(screen["scientific_scope"]["promotion_allowed"] is False, "screen may not promote")
    require(screen["scientific_scope"]["lineage_mutation_allowed"] is False, "screen may not mutate lineage")

    model = screen["model"]
    require(model["model_id"] == "zai-org/GLM-4.7-Flash", "model id drifted")
    require(model["revision"] == "7dd20894a642a0aa287e9827cb1a1f7f91386b67", "GLM revision drifted")
    require(model["precision"] == "bfloat16", "GLM screen must remain BF16")
    require(model["quantization"] is False, "quantization is forbidden")
    require(model["cpu_or_disk_offload"] is False, "CPU/disk offload is forbidden")
    require(model["gpu_count"] == 1, "screen must use one GPU")

    candidate = next((row for row in recovery["candidates"] if row["model_id"] == model["model_id"]), None)
    require(candidate is not None, "GLM candidate missing from recovery protocol")
    for key in ("revision", "model_type", "loader", "license"):
        require(candidate[key] == model[key], f"model identity mismatch: {key}")

    probes = screen["probes"]
    require(len(probes) == 5, "cheap screen must contain exactly five probes")
    require(int(screen["generation"]["seed"]) == 11, "cheap screen must use one frozen seed=11")
    ids = [row["probe_id"] for row in probes]
    require(len(ids) == len(set(ids)), "probe ids must be unique")
    require(sum(int(row["max_new_tokens"]) for row in probes) <= 3000, "total generated-token ceiling exceeded")

    semantic_ids = {
        row["task_id"]
        for key in ("generation_probes", "continuation_prompts", "structure_tests", "repetition_checks", "length_termination_checks")
        for row in semantic.get(key, [])
    }
    topology_by_id = {row["case_id"]: row for row in topology["cases"]}
    expected = {
        "semantic_instruction_triplet": ("semantic", "instruction_triplet", "fixed_non_thinking", 96),
        "semantic_structured_planet": ("semantic", "structured_planet_answer", "fixed_non_thinking", 128),
        "semantic_anti_repetition": ("semantic", "anti_repetition", "fixed_non_thinking", 128),
        "topology_diagnosis_d1": ("topology", "D1", "reasoning_aware", 1536),
        "topology_controller_c3": ("topology", "C3", "reasoning_aware", 1024),
    }
    for probe in probes:
        probe_id = probe["probe_id"]
        require(probe_id in expected, f"unexpected probe: {probe_id}")
        source, source_id, lane, budget = expected[probe_id]
        require(probe["source"] == source, f"{probe_id} source drifted")
        require(probe["lane"] == lane, f"{probe_id} lane drifted")
        require(int(probe["max_new_tokens"]) == budget, f"{probe_id} budget drifted")
        require(probe.get("required_for_advance") is True, f"{probe_id} must remain required")
        if source == "semantic":
            require(probe["task_id"] == source_id and source_id in semantic_ids, f"{probe_id} semantic source drifted")
        else:
            require(probe["case_id"] == source_id and source_id in topology_by_id, f"{probe_id} topology source drifted")
            require(topology_by_id[source_id]["role"] == probe["role"], f"{probe_id} topology role drifted")
            require(float(probe["minimum_quality_100"]) >= 70.0, f"{probe_id} quality floor too low")

    execution = screen["execution"]
    require(0 < int(execution["hard_wall_seconds"]) <= 1500, "hard wall must be <=25 minutes")
    require(0 < float(execution["max_hourly_usd"]) <= 2.25, "hourly cost ceiling drifted")
    require(0 < float(execution["max_estimated_total_usd"]) <= 0.75, "total estimated cost ceiling drifted")
    require(execution["network_volume_attached"] is False, "cheap screen must not attach Network Volume")
    require(execution["persistent_model_cache_allowed"] is False, "cheap screen must remain ephemeral")
    gpu_ids = execution["gpu_type_ids"]
    require(len(gpu_ids) == 2, "cheap-screen GPU allowlist drifted")
    require(all("RTX PRO 6000 Blackwell" in item for item in gpu_ids), "cheap screen must use only RTX PRO 6000 Blackwell")
    require(not any("H200" in item or "B200" in item for item in gpu_ids), "expensive fallback is forbidden")

    require(screen["observability"]["write_sample_immediately"] is True, "per-sample S3 writes are required")
    require(screen["observability"]["write_heartbeat_after_every_sample"] is True, "per-sample heartbeat is required")
    require(screen["observability"]["write_terminal_result"] is True, "terminal result is required")
    require(screen["advance_gate"]["on_pass"] == "eligible_for_tiny_adaptation_experiment_only", "advance gate drifted")
    require(screen["advance_gate"]["on_nonpass"] == "do_not_start_full_lora_recovery", "full LoRA block drifted")

    summary = {
        "status": "valid",
        "protocol_id": screen["protocol_id"],
        "model_id": model["model_id"],
        "revision": model["revision"],
        "probe_count": len(probes),
        "seed_count": 1,
        "total_max_new_tokens": sum(int(row["max_new_tokens"]) for row in probes),
        "training_performed": False,
        "gpu_type_ids": gpu_ids,
        "hard_wall_seconds": execution["hard_wall_seconds"],
        "max_hourly_usd": execution["max_hourly_usd"],
        "max_estimated_total_usd": execution["max_estimated_total_usd"],
        "paid_launch_allowed_now": screen["governance"]["paid_launch_allowed_now"],
    }
    return summary


def main() -> int:
    print(json.dumps(validate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())