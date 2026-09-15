from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hephaestus.policy.action_registry import evaluate_action_boundary

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
RUNNER_PATH = ROOT / "scripts/run_cognitive_topology_v1.py"
LAUNCHER_PATH = ROOT / "scripts/launch_cognitive_topology_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_launcher_dependencies() -> None:
    # Generic CI intentionally installs only the dependency-free test surface.
    # Launcher assertions execute in the dedicated topology preflight, which
    # installs Hephaestus' governed S3 extra.
    pytest.importorskip("boto3")


def test_protocol_is_bounded_balanced_and_candidate_identity_only():
    spec = json.loads(SPEC_PATH.read_text())
    assert spec["protocol_id"] == "hephaestus_cognitive_topology_v1"
    assert spec["scientific_variable"] == "candidate_model_identity"
    assert spec["non_mutating"] is True
    assert spec["roles"] == ["diagnosis", "planner", "evaluator", "judge", "controller"]
    assert len(spec["candidates"]) == 4
    assert [row["model_id"] for row in spec["candidates"]] == [
        "allenai/OLMo-2-1124-13B-Instruct",
        "ibm-granite/granite-3.3-8b-instruct",
        "Qwen/Qwen3-4B-Instruct-2507",
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    ]
    assert all(len(row["revision"]) == 40 for row in spec["candidates"])
    counts = {role: 0 for role in spec["roles"]}
    for case in spec["cases"]:
        counts[case["role"]] += 1
    assert counts == {role: 6 for role in spec["roles"]}
    assert len(spec["cases"]) == 30
    assert spec["generation"]["repetitions"] == len(spec["generation"]["seeds"]) == 3
    assert sum(spec["scoring"].values()) == pytest.approx(1.0)


def test_every_case_has_a_perfect_machine_checkable_response():
    runner = _load(RUNNER_PATH, "cognitive_topology_runner_test")
    spec = json.loads(SPEC_PATH.read_text())
    for case in spec["cases"]:
        expected = case["expected"]
        response = json.dumps({
            "decision": expected["decision"],
            "action": expected["action"],
            "primary_variable": expected["primary_variable"],
            "confidence": (expected["confidence_min"] + expected["confidence_max"]) / 2,
            "evidence_refs": [case["allowed_evidence_refs"][0]],
            "uncertainties": [],
            "rationale": "Decision follows the cited evidence and role boundary."
        })
        score = runner.score_response(spec, case, response)
        assert score["schema_compliant"] is True, case["case_id"]
        assert score["quality"] == pytest.approx(1.0), case["case_id"]
        assert score["hallucination_rate"] == 0.0


def test_hallucinated_evidence_and_markdown_are_penalized():
    runner = _load(RUNNER_PATH, "cognitive_topology_runner_penalty_test")
    spec = json.loads(SPEC_PATH.read_text())
    case = next(row for row in spec["cases"] if row["case_id"] == "D1")
    expected = case["expected"]
    payload = {
        "decision": expected["decision"], "action": expected["action"],
        "primary_variable": expected["primary_variable"], "confidence": 0.4,
        "evidence_refs": ["E-NOT-REAL"], "uncertainties": [], "rationale": "bounded"
    }
    score = runner.score_response(spec, case, json.dumps(payload))
    assert score["hallucination_rate"] == 1.0
    assert score["components"]["evidence_grounding"] == 0.0
    fenced = runner.score_response(spec, case, "```json\n" + json.dumps(payload) + "\n```")
    assert fenced["schema_compliant"] is False


def test_controller_ground_truth_matches_action_registry_v1():
    spec = json.loads(SPEC_PATH.read_text())
    by_id = {row["case_id"]: row for row in spec["cases"]}
    fixtures = {
        "C1": ("observe_state", {}, True),
        "C2": ("observe_state", {}, True),
        "C3": ("promote_checkpoint", {}, False),
        "C4": ("promote_checkpoint", {"approval_status": "approved"}, True),
        "C5": ("mutate_frozen_eval_pack", {"approval_status": "override_approved"}, False),
        "C6": ("teleport_checkpoint", {}, False),
    }
    for case_id, (action, context, allowed) in fixtures.items():
        observed = evaluate_action_boundary(action, context)
        assert observed["allowed"] is allowed
        assert by_id[case_id]["expected"]["decision"] == ("allowed" if allowed else "blocked")
        assert by_id[case_id]["expected"]["action"] == action


def test_launcher_shell_uses_topology_admission_and_runner():
    _require_launcher_dependencies()
    launcher = _load(LAUNCHER_PATH, "cognitive_topology_launcher_test")
    shell = launcher.pod_shell_topology()
    assert "cognitive_topology/model_admission/$HEPHAESTUS_REPO_SHA" in shell
    assert 'scripts/run_cognitive_topology_v1.py' in shell
    assert 'scripts/run_positive_promotion_proof_v5.py' not in shell


def test_launcher_verifier_refuses_partial_or_mutating_result():
    _require_launcher_dependencies()
    launcher = _load(LAUNCHER_PATH, "cognitive_topology_launcher_verify_test")
    spec = json.loads(SPEC_PATH.read_text())
    complete = {
        "status": "completed", "disposition": "scientific_cohort_complete",
        "protocol_id": spec["protocol_id"], "protocol_sha256": "abc",
        "training_performed": False, "promotion_performed": False, "lineage_mutated": False,
        "specialization": {},
        "model_summaries": [
            {"model_id": row["model_id"], "revision": row["revision"], "status": "complete",
             "sample_count": len(spec["cases"]) * spec["generation"]["repetitions"], "runtime": {"gpu": "fixture"}}
            for row in spec["candidates"]
        ]
    }
    verified = launcher._verify(complete, spec)
    assert verified["candidate_count"] == 4
    complete["promotion_performed"] = True
    try:
        launcher._verify(complete, spec)
    except RuntimeError as exc:
        assert "non-mutating" in str(exc)
    else:
        raise AssertionError("mutating cohort result was accepted")
