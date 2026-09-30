from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neutral_holdout_is_paired_identifier_only_transform() -> None:
    mod = _module("neutral_pack", "scripts/build_interface_repair_v3_1_neutral_holdout.py")
    pack = mod.build_pack()
    mod.validate(pack)
    assert mod.canonical_sha256(pack) == "0bae520eda2b59f9317baacfd9d77a24f51f0b7c4a2b5e7e057e0a8cee2d7492"
    part = pack["partitions"]["evaluator"]
    source = part["regression"]
    neutral = part["neutral_holdout"]
    assert len(source) == len(neutral) == 128
    for before, after in zip(source, neutral):
        assert after["source_case_id"] == before["case_id"]
        assert after["expected"] == before["expected"]
        assert after["verified_facts"] == before["verified_facts"]
        assert after["upstream_output"] == before["upstream_output"]
        assert [x["fact"] for x in after["evidence"]] == [x["fact"] for x in before["evidence"]]
        assert after["context"] != before["context"]
        assert after["allowed_evidence_refs"] != before["allowed_evidence_refs"]


def test_protocol_freezes_exact_v31_adapter_and_forbids_training() -> None:
    cfg = json.loads((ROOT / "configs/experiments/hephaestus_interface_repair_v3_1_neutral_holdout.json").read_text())
    marker = json.loads((ROOT / "configs/experiments/interface_repair_v3_1_neutral_holdout.launch.json").read_text())
    assert cfg["pack"]["canonical_sha256"] == "0bae520eda2b59f9317baacfd9d77a24f51f0b7c4a2b5e7e057e0a8cee2d7492"
    assert cfg["source_v31_adapter"]["sha256"] == "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"
    assert cfg["source_v31_adapter"]["source_run_id"] == "interface-repair-v3-evaluator-full-36738914957"
    assert cfg["diagnostic_only"] is True
    assert cfg["training_performed"] is False
    assert cfg["governance"]["training_allowed"] is False
    assert cfg["governance"]["adapter_mutation_allowed"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert cfg["governance"]["certification_claim_allowed"] is False
    assert cfg["governance"]["paid_launch_allowed"] is False
    assert marker["launch_authorized"] is False


def test_runner_contains_no_training_or_adapter_persistence_path() -> None:
    text = (ROOT / "scripts/run_interface_repair_v3_1_neutral_holdout.py").read_text()
    forbidden = (
        "v2.train(", ".backward(", "optimizer.step(", "save_pretrained(",
        "selected-adapter.tar.gz", "adapter_persist_seconds",
    )
    for token in forbidden:
        assert token not in text
    assert '"training_performed": False' in text
    assert '"adapter_mutated": False' in text
    assert "requires_grad_(False)" in text


def test_launcher_renders_exact_gpu_and_neutral_runner_without_spend() -> None:
    launcher = _module("neutral_launcher", "scripts/launch_interface_repair_v3_1_neutral_holdout.py")
    cfg = launcher.load_json(launcher.CFG_PATH)
    body = launcher.request_body(cfg, "deadbeef", "neutral-dry-run", real_env=False)
    assert body["gpuTypeIds"] == ["NVIDIA RTX PRO 6000 Blackwell Server Edition"]
    shell = body["dockerStartCmd"][2]
    assert "build_interface_repair_v3_1_neutral_holdout.py" in shell
    assert "run_interface_repair_v3_1_neutral_holdout.py" in shell
    assert "run_interface_repair_v2_cached.py --preflight" not in shell


def test_predeclared_diagnostic_gates_match_decision_rule() -> None:
    cfg = json.loads((ROOT / "configs/experiments/hephaestus_interface_repair_v3_1_neutral_holdout.json").read_text())
    gates = cfg["diagnostic_gates"]
    assert gates["minimum_semantic_quality_100"] == 95
    assert gates["minimum_semantic_exact_pass_rate"] == 0.95
    assert gates["minimum_decision_exact_pass_rate"] == 0.95
    assert gates["system_action_exact_pass_rate_required"] == 1.0
    assert gates["maximum_system_semantic_escalation_rate"] == 0.0
