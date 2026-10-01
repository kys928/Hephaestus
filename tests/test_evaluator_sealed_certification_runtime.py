from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/experiments/hephaestus_evaluator_sealed_certification_v1.json"
MARKER = ROOT / "configs/experiments/hephaestus_evaluator_sealed_certification_v1.launch.json"
RUNNER = ROOT / "scripts/run_evaluator_sealed_certification_v1.py"
LAUNCHER = ROOT / "scripts/launch_evaluator_sealed_certification_v1.py"
EXPECTED_PACK_SHA = "1314d28c978538ca8bb5527c24e4200d5463f7ad90b084aaae1e656d0b530d25"
EXPECTED_ADAPTER_SHA = "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_certification_protocol_is_frozen_inference_only_and_launch_locked() -> None:
    cfg = _load(CFG)
    marker = _load(MARKER)
    assert cfg["pack"]["canonical_sha256"] == EXPECTED_PACK_SHA
    assert cfg["pack"]["preflight_cases"] == 8
    assert cfg["pack"]["certification_cases"] == 128
    assert cfg["pack"]["regression_cases"] == 128
    assert cfg["model_protocol"]["weights_frozen_for_certification"] is True
    assert cfg["training"]["optimizer_steps"] == 0
    assert cfg["governance"]["training_allowed"] is False
    assert cfg["governance"]["adapter_mutation_allowed"] is False
    assert cfg["governance"]["candidate_weights_mutation_allowed"] is False
    assert cfg["governance"]["paid_launch_allowed"] is False
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert cfg["governance"]["identical_rerun_after_scientific_failure_allowed"] is False
    assert marker["launch_authorized"] is False


def test_certification_runner_contains_no_training_or_promotion_path() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "optimizer" not in text.lower()
    assert ".backward(" not in text
    assert "save_pretrained" not in text
    assert '"training_performed": False' in text
    assert '"adapter_mutated": False' in text
    assert '"production_promotion_performed": False' in text
    assert "sealed_certification_partitions_touched" in text
    assert "preflight_rejected" in text
    assert "scientifically_rejected" in text


def test_certification_runner_binds_candidate_pack_and_code_blobs() -> None:
    runner = _import(RUNNER, "sealed_cert_runner_test")
    cfg = runner.load_cfg()
    candidate = runner.load_candidate(cfg)
    assert candidate["model"]["adapter_sha256"] == EXPECTED_ADAPTER_SHA
    assert candidate["sealed_evidence"]["canonical_sha256"] == EXPECTED_PACK_SHA
    observed = runner.verify_code_blobs(candidate)
    assert observed == candidate["architecture"]["code_blobs"]
    pack = runner.load_pack(cfg, candidate)
    assert len(pack["partitions"]["evaluator"]["preflight"]) == 8
    assert len(pack["partitions"]["evaluator"]["certification"]) == 128
    assert len(pack["partitions"]["evaluator"]["regression"]) == 128


def test_certification_gates_remain_strict() -> None:
    runner = _import(RUNNER, "sealed_cert_gate_test")
    cfg = runner.load_cfg()
    passing = {
        "semantic_quality_100": 98.0,
        "semantic_exact_pass_rate": 0.97,
        "schema_compliance": 1.0,
        "evidence_grounding": 1.0,
        "hallucination_rate": 0.0,
        "primary_variable_projection_rate": 1.0,
        "boundary_projection_rate": 1.0,
        "system_action_exact_pass_rate": 1.0,
        "system_semantic_escalation_rate": 0.0,
    }
    result = runner.certify(cfg, passing, dict(passing))
    assert result["certified"] is True

    one_action_miss = dict(passing)
    one_action_miss["system_action_exact_pass_rate"] = 127 / 128
    result = runner.certify(cfg, passing, one_action_miss)
    assert result["certified"] is False
    assert result["system_certification"]["checks"]["regression_system_action_exact"] is False

    weak_model = dict(passing)
    weak_model["semantic_exact_pass_rate"] = 0.949
    result = runner.certify(cfg, weak_model, passing)
    assert result["certified"] is False
    assert result["model_certification"]["checks"]["certification_semantic_exact"] is False


def test_launcher_dry_request_is_single_gpu_and_no_fallback() -> None:
    launcher = _import(LAUNCHER, "sealed_cert_launcher_test")
    cfg = launcher.load_json(CFG)
    body = launcher.request_body(cfg, "a" * 40, "dry-run", real_env=False)
    assert body["gpuCount"] == 1
    assert body["gpuTypeIds"] == ["NVIDIA RTX PRO 6000 Blackwell Server Edition"]
    assert body["interruptible"] is False
    assert body["env"]["HEPHAESTUS_EVALUATOR_CERTIFICATION_RUN_ID"] == "dry-run"
    command = body["dockerStartCmd"][-1]
    assert "run_evaluator_sealed_certification_v1.py" in command
    assert "build_evaluator_certification_sealed_v1.py" in command
