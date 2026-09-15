from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiments/hephaestus_adaptation_elasticity_v1.json"
TOPOLOGY_PATH = ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"


def _load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_freezes_low_dose_and_baseline_hash() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    topology_raw = TOPOLOGY_PATH.read_bytes()
    assert protocol["protocol_id"] == "hephaestus_adaptation_elasticity_v1"
    assert protocol["baseline"]["protocol_sha256"] == hashlib.sha256(topology_raw).hexdigest()
    assert protocol["scientific_variable"] == "base_model_identity_under_fixed_role_lora_dose"
    assert protocol["roles"] == ["diagnosis", "planner", "evaluator", "judge", "controller"]
    train = protocol["training"]
    assert train["method"] == "lora"
    assert train["peft_version"] == "0.20.0"
    assert train["rank"] == 8 and train["alpha"] == 16 and train["dropout"] == 0.0
    assert train["learning_rate"] == 5e-5
    assert train["epochs"] == 2 and train["dose_checkpoints"] == [1, 2]
    assert train["micro_batch_size"] == 1 and train["gradient_accumulation_steps"] == 4
    assert protocol["governance"]["promotion_allowed"] is False
    assert protocol["governance"]["lineage_mutation_allowed"] is False
    assert protocol["governance"]["training_is_experimentally_approved"] is True


def test_dataset_is_balanced_and_does_not_contaminate_frozen_topology() -> None:
    builder = _load_script("elasticity_builder", "scripts/build_adaptation_elasticity_dataset_v1.py")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    topology = json.loads(TOPOLOGY_PATH.read_text())
    rows = builder.build_dataset()
    assert len(rows) == 240
    for role in protocol["roles"]:
        assert sum(row["role"] == role for row in rows) == 48
    report = builder.contamination_report(rows, topology, protocol)
    assert report["passed"] is True
    assert report["violations"] == []
    assert report["observed_maxima"]["jaccard"] <= protocol["dataset"]["max_normalized_token_jaccard"]
    required = {"decision", "action", "primary_variable", "confidence", "evidence_refs", "uncertainties", "rationale"}
    for row in rows:
        assert set(row["target"]) == required
        assert set(row["target"]["evidence_refs"]).issubset(set(row["allowed_evidence_refs"]))
        assert 0.0 <= float(row["target"]["confidence"]) <= 1.0


def test_controller_training_examples_follow_registry_semantics() -> None:
    builder = _load_script("elasticity_builder_controller", "scripts/build_adaptation_elasticity_dataset_v1.py")
    from hephaestus.policy.action_registry import evaluate_action_boundary

    for row in builder.build_dataset():
        if row["role"] != "controller":
            continue
        action = row["target"]["action"]
        facts = " ".join(item["fact"] for item in row["evidence"])
        approval = ""
        if "override_approved" in facts:
            approval = "override_approved"
        elif "approved" in facts:
            approval = "approved"
        boundary = evaluate_action_boundary(action, {"approval_status": approval})
        expected = "allowed" if boundary["allowed"] else "blocked"
        assert row["target"]["decision"] == expected
        assert row["target"]["primary_variable"] == "action_boundary"


def test_lora_target_selector_excludes_visual_modules() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    runner = _load_script("elasticity_runner", "scripts/run_adaptation_elasticity_v1.py")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.q_proj = torch.nn.Linear(4, 4)
            self.language_model.k_proj = torch.nn.Linear(4, 4)
            self.vision_model = torch.nn.Module()
            self.vision_model.q_proj = torch.nn.Linear(4, 4)
            self.other = torch.nn.Linear(4, 4)

    targets = runner._target_modules(Tiny(), ["q_proj", "k_proj"])
    assert targets == ["language_model.k_proj", "language_model.q_proj"]


def test_elasticity_metric_uses_signed_gain_and_training_cost() -> None:
    runner = _load_script("elasticity_runner_metric", "scripts/run_adaptation_elasticity_v1.py")
    baseline = {"quality_100": 50.0, "schema_compliance": 0.8, "evidence_grounding": 0.7, "confidence_calibration": 0.5, "hallucination_rate": 0.1}
    adapted = {"quality_100": 60.0, "schema_compliance": 1.0, "evidence_grounding": 0.8, "confidence_calibration": 0.7, "hallucination_rate": 0.05}
    record = runner._elasticity_record(baseline=baseline, adapted=adapted, dose=1, training_seconds=1800.0, training_tokens=500_000, trainable_parameters=10_000_000, adapter_manifest={"adapter_bytes": 1234}, training_peak_vram_bytes=10, mean_loss=0.4)
    assert record["delta_quality_points"] == 10.0
    assert record["delta_per_gpu_hour"] == 20.0
    assert record["delta_per_million_trainable_parameters"] == 1.0
    assert record["delta_per_million_training_tokens"] == 20.0


def test_launcher_shell_uses_adaptation_admission_and_driver_when_s3_available() -> None:
    if importlib.util.find_spec("boto3") is None:
        return
    launcher = _load_script("elasticity_launcher", "scripts/launch_adaptation_elasticity_v1.py")
    shell = launcher.pod_shell_elasticity()
    assert "/adaptation_elasticity/model_admission/$HEPHAESTUS_REPO_SHA" in shell
    assert '"$PY" scripts/run_adaptation_elasticity_v1.py' in shell
    assert "run_positive_promotion_proof_v5.py" not in shell
