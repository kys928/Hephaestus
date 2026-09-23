from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_interface_mastery_v1b.json"
MARKER=ROOT/"configs/experiments/interface_mastery_v1b.launch.json"

def test_phase_iib_runtime_is_repair_gated_and_eval_only() -> None:
    cfg=json.loads(CFG.read_text());marker=json.loads(MARKER.read_text())
    assert cfg["pack"]["canonical_sha256"]=="e71f02a1965e53d1c157e5c6f1d3adba23216560569325c25b18661967b1a9ed"
    assert cfg["repair_gate"]["required_roles"]==["controller","evaluator","judge"]
    assert cfg["governance"]["training_use_allowed"] is False
    assert cfg["governance"]["weights_mutation_allowed"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    if cfg["repair_gate"]["required_repair_run_id"] is None:
        assert marker["authorized"] is False
