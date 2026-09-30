from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_interface_repair_v3_1_evaluator.py"
RUNNER = ROOT / "scripts/run_interface_repair_v3_1_evaluator.py"
CFG = ROOT / "configs/experiments/hephaestus_interface_repair_v3_evaluator.json"
CANDIDATES = ROOT / "configs/models/hephaestus_phase_ii_repair_candidates_v1.json"
EXPECTED_PACK_SHA = "8e1d1e689c1497b1ee57d0fca834dd0e486518b2e9b8da5c12ff5ab23f686407"
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


def test_v31_pack_is_deterministic_crossed_fresh_and_preflight_isolated() -> None:
    builder = import_file(BUILDER, "interface_repair_v31_builder")
    pack = builder.build_pack()
    builder.validate(pack)
    assert builder.canonical_sha256(pack) == EXPECTED_PACK_SHA

    part = pack["partitions"]["evaluator"]
    train = part["train"]
    preflight = part["preflight"]
    cert = part["certification"]
    regression = part["regression"]
    assert len(train) == 1536
    assert len(preflight) == 8
    assert len(cert) == 128
    assert len(regression) == 128

    roots = set(pack["contract_vocabulary"]["evaluator"]["decision"])
    base = builder.base
    for context in base.TRAIN_CONTEXTS:
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
    assert {row["semantic_root"] for row in preflight} == roots

    groups = {
        "train": {row["context"] for row in train},
        "preflight": {row["context"] for row in preflight},
        "certification": {row["context"] for row in cert},
        "regression": {row["context"] for row in regression},
    }
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert groups[left].isdisjoint(groups[right])

    assert all(not row["case_id"].startswith("IR-EVALUATOR-CERTIFICATION-") for row in cert + regression)
    assert pack["diagnostic_burned_sets"]["reused_in_v3_certification"] is False
    assert pack["diagnostic_burned_sets"]["reused_in_v3_regression"] is False
    assert pack["diagnostic_burned_sets"]["v3_sampled_certification_cases_reused_in_v31"] is False
    assert pack["diagnostic_burned_sets"]["v31_preflight_cases_reused_in_certification"] is False
    assert pack["diagnostic_burned_sets"]["v31_preflight_cases_reused_in_regression"] is False


def test_v31_protocol_is_evaluator_only_and_preserves_planner() -> None:
    cfg = load(CFG)
    candidates = load(CANDIDATES)
    assert cfg["protocol_id"] == "hephaestus_interface_repair_v3_evaluator_boundary"
    assert cfg["protocol_revision"] == "3.1"
    assert cfg["execution_path_version"] == 4
    assert cfg["trainable_roles"] == ["evaluator"]
    assert cfg["pack"]["builder_path"] == "scripts/build_interface_repair_v3_1_evaluator.py"
    assert cfg["pack"]["canonical_sha256"] == EXPECTED_PACK_SHA
    assert cfg["pack"]["v2_burned_cases_reused_for_certification"] is False
    assert cfg["pack"]["v2_burned_cases_reused_for_regression"] is False
    assert cfg["pack"]["v3_sampled_certification_cases_reused_in_v31"] is False
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


def test_v31_training_geometry_and_format_retry_are_bounded() -> None:
    cfg = load(CFG)
    training = cfg["training"]
    assert training["method"] == "continue_existing_lora"
    assert training["lora_rank"] == 16
    assert training["lora_alpha"] == 32
    assert training["optimizer_steps"]["evaluator"] * training["gradient_accumulation_steps"] == 1536
    assert cfg["preflight"]["role"] == "evaluator"
    assert cfg["preflight"]["partition"] == "preflight"
    assert cfg["execution"]["format_retry_max_attempts"] == 1
    assert cfg["execution"]["format_retry_target_free"] is True
    assert cfg["governance"]["production_promotion_allowed"] is False
    assert cfg["governance"]["automatic_role_dispatch_allowed"] is False
    assert cfg["governance"]["preserved_planner_mutation_allowed"] is False


def test_v31_runner_separates_model_and_system_certification_and_reports_retry() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    v3_text = (ROOT / "scripts/run_interface_repair_v3_evaluator.py").read_text(encoding="utf-8")
    assert "project_evaluator_action" in v3_text
    assert '"semantic_exact_pass_rate"' in v3_text
    assert '"model_action_agreement_rate"' in v3_text
    assert '"system_action_exact_pass_rate"' in v3_text
    assert '"system_semantic_escalation_rate"' in v3_text
    assert '"model_certification"' in v3_text
    assert '"system_certification"' in v3_text
    assert "diagnostic_uncertified_v2_candidate" in v3_text
    assert "materialize_parent_cached" in v3_text
    assert "FORMAT-RETRY NOTICE" in text
    assert '"format_retry_attempt_rate"' in text
    assert '"format_retry_success_rate"' in text
    assert '"preflight_partition": "dedicated_preflight"' in text


def test_v3_system_boundary_is_not_gated_on_model_action_agreement() -> None:
    runner = import_file(ROOT / "scripts/run_interface_repair_v3_evaluator.py", "interface_repair_v3_runner")
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
