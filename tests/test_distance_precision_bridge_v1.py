from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_distance_to_role_v1_bridge as bridge


def test_distance_precision_bridge_allows_only_frozen_fp16_to_bf16_transition() -> None:
    assert bridge.classify_precision_bridge(["torch.float16"], ["torch.bfloat16"]) == "fp16_topology_to_bf16_training"
    assert bridge.classify_precision_bridge(["torch.float16"], ["torch.float16"]) == "same_precision"
    assert bridge.classify_precision_bridge(["torch.bfloat16"], ["torch.bfloat16"]) == "same_precision"
    assert bridge.classify_precision_bridge(["torch.float32"], ["torch.bfloat16"]) == "unsupported_precision_bridge"
    assert bridge.classify_precision_bridge(["torch.float16", "torch.float32"], ["torch.bfloat16"]) == "unsupported_precision_bridge"
