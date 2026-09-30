from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_interface_repair_v3_evaluator.py"
RUNNER = ROOT / "scripts/run_interface_repair_v3_evaluator.py"
CFG = ROOT / "configs/experiments/hephaestus_interface_repair_v3_evaluator.json"
CANDIDATES = ROOT / "configs/models/hephaestus_phase_ii_repair_candidates_v1.json"
EXPECTED_PACK_SHA = "35f26b93527cc919ca65e5b3e5a83e614dd601afb40ce598fc9345aacc7d7098"
V2_EVALUATOR_SHA = "cd675321340f831351710f9af122cd2a7e3741dc087955a43671bd6d0eff724e"
V2_PLANNER_SHA = "6f6b8d9d50917cda52b3554d849d71d9cb295435b480d8d25e6a06d2784e0c71"


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v3_pack_is_deterministic_crossed_and_fresh() -> None:
    builder = import_file(BUILDER, "interface_repair_v3_builder")
    pack = builder.build_pack()
    builder.validate(pack)
    assert builder.canonical_sha256(pack) == EXPECTED_PACK_SHA

    part = pack["partitions"]["evaluator"]
    train = part["train"]
    cert = part["certification"]
    regression = part["regression"]
    assert len(train) == 1536
    assert len(cert) == 128
    assert len(regression) == 128

    roots = set(pack["contract_vocabulary"]["evaluator"]["decision"])
    for context in builder.TRAIN_CONTEXTS:
        observed = {
            row["semantic_root"]
            for row in train
            if row["context"] == context and not row["focused_boundary"]
        }
        assert observed == roots

    counts = {root: sum(row["semantic_root"] == root for row in train) for root in roots}
    assert counts["improved"] > counts["regressed"]
    assert counts["equivalent"] > counts["regressed"]
    assert any(row["upstream_output"] is not None for row in train)

    train_contexts = {row["context"] for row in train}
    cert_contexts = {row["context"] for row in cert}
    regression_contexts = {row["context"] for row in regression}
    assert train_contexts.isdisjoint(cert_contexts)
    assert train_contexts.isdisjoint(regression_contexts)
    assert cert_contexts.isdisjoint(regression_contexts)
    assert all(not row["case_id"].startswith("IR-EVALUATOR-CERTIFICATION-") for row in cert + regression)
    assert pack["diagnostic_burned_sets"]["reused_in_v3_certification"] is False
    assert pack["diagnostic_burned_sets"]["reused_in_v3_regression"] is False


def test_v3_protocol_is_evaluator_only_and_preserves_planner() -> None:
    cfg = load(CFG)
    candidates = load(CANDIDATES)
    assert cfg["protocol_id"] == "hephaestus_interface_repair_v3_evaluator_boundary"
    assert cfg["execution_path_version"] == 3
    assert cfg["trainable_roles"] == ["evaluator"]
    assert cfg["pack"]["canonical_sha256"] == EXPECTED_PACK_SHA
    assert cfg["pack"]["v2_burned_cases_reused_for_certification"] is False
    assert cfg["pack"]["v2_burned_cases_reused_for_regression"] is False
    assert cfg["source_evaluator_adapter"]["sha256"] == V2_EVALUATOR_SHA
    assert cfg["source_evaluator_adapter"]["status"] == "diagnostic_uncertified_v2_candidate"
    assert cfg["preserved_planner_candidate"]["sha256"] == V2_PLANNER_SHA
    assert cfg["preserved_planner_candidate"]["retrain_in_v3"] is False
    assert candidates["roles"]["planner"]["adapter"]["sha256"] == V2_PLANNER_SHA
    assert candidates["roles"]["planner"]["certification"]["passed"] is True
    assert candidates["roles"]["evaluator"]["adapter"]["sha256"] == V2_EVALUATOR_SHA
    assert candidates["roles"]["evaluator"]["certification"]["passed"] is False
    assert candidates["production_certified"] is False
    assert candidates["automatic_role_dispatch_enabled"] is False


def test_v3_training_geometry_matches_crossed_pack() -> None:
    cfg = load(CFG)
    training = cfg["training"]
    assert training["method"] == "continue_existing_lora"
    assert training["lora_rank"] == 16
    assert training["lora_alpha"] == 32
    assert training["optimizer_steps"]["evaluator"] * training["gradient_accumulation_steps"] == 1536
    assert cfg["preflight"]["role"] == "evaluator"
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert cfg["governance"]["preserved_planner_mutation_allowed"] is False


def test_v3_runner_separates_model_and_system_certification() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "project_evaluator_action" in text
    assert '"semantic_exact_pass_rate"' in text
    assert '"model_action_agreement_rate"' in text
    assert '"system_action_exact_pass_rate"' in text
    assert '"system_semantic_escalation_rate"' in text
    assert '"model_certification"' in text
    assert '"system_certification"' in text
    assert "diagnostic_uncertified_v2_candidate" in text
    assert "materialize_parent_cached" in text


def test_v3_system_boundary_is_not_gated_on_model_action_agreement() -> None:
    runner = import_file(RUNNER, "interface_repair_v3_runner")
    cfg = load(CFG)
    post = {
        "semantic_quality_100": 100.0,
        "semantic_exact_pass_rate": 1.0,
        "schema_compliance": 1.0,
        "evidence_grounding": 1.0,
        "hallucination_rate": 0.0,
        "model_action_agreement_rate": 0.0,
        "model_semantic_escalation_rate": 1.0,
        "upstream_copy_violation_rate": 0.0,
        "boundary_projection_rate": 1.0,
        "system_action_exact_pass_rate": 1.0,
        "system_semantic_escalation_rate": 0.0,
    }
    result = runner.certify_v3(post, post, post, post, cfg)
    assert result["model_certification"]["passed"] is True
    assert result["system_certification"]["passed"] is True
    assert result["certified"] is True
    assert result["report_only"]["interface_model_action_agreement_rate"] == 0.0
    assert result["report_only"]["interface_model_semantic_escalation_rate"] == 1.0
