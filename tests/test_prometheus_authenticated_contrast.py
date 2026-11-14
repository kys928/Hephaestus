"""Regression tests of the authenticated, retrospective Prometheus source inventory.

This test NEVER substitutes synthetic metadata for actual S3 readback. The live
full-byte verification is externally evidenced in Actions run 43637529964.
"""
from hashlib import sha256
import json
from pathlib import Path

from hephaestus.diagnosis.probes import analyze_dataset_task_coverage

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/evidence/prometheus-authenticated-contrast-20261114.json"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _record() -> dict:
    return _json(RECORD)


def test_authenticated_s3_source_contrast_matches_archived_scientific_records():
    record = _record()
    orig, candidate = record["source_inputs"]
    assert record["classification"] == "retrospective_development_only"
    assert record["source_validation"]["run_state"] == "both_verifier_jobs_succeeded"
    assert record["source_validation"]["third_party_independent_human_adjudication"] is False
    assert record["source_validation"]["protected_final_holdout_accessed"] is False
    assert orig["sha256"] != candidate["sha256"]
    prior = _json(ROOT / "docs/evidence/first-targeted-post-failure-diagnostics-001-33874832481/diagnostics.json")
    source = prior["s3_evidence"]
    assert orig["sha256"] == source["dataset_sha256"].removeprefix("sha256:")
    assert orig["bytes"] == source["dataset_bytes"]
    assert orig["rows"] == source["dataset_rows"]
    latest = _json(ROOT / "docs/evidence/first-selected-dataset-preparation-001-33881924136/verification.json")
    assert candidate["sha256"] == latest["processed"]["sha256"].removeprefix("sha256:")
    assert candidate["bytes"] == latest["processed"]["byte_size"]
    assert orig["prompt_target_fields"] == 0
    assert orig["text_prompt_target_wrapped"] == 0
    assert candidate["prompt_target_fields"] == 0
    assert candidate["text_prompt_target_wrapped"] == 35774
    assert candidate["wrapped_empty_targets"] == 206
    assert candidate["wrapped_structured_json_object_targets"] == 0


def test_checksum_archival_scientific_provenance_is_still_intact():
    for dirname in (
        "first-targeted-post-failure-diagnostics-001-33874832481",
        "first-selected-dataset-preparation-001-33881924136",
        "second-controlled-experiment-001-33889959923",
    ):
        directory = ROOT / "docs/evidence" / dirname
        lines = (directory / "SHA256SUMS").read_text().splitlines()
        verified = 0
        for line in lines:
            digest, name = line.split(maxsplit=1)
            path = directory / name
            if not path.is_file():
                assert name == "original-github-actions-artifact.zip"
                continue
            assert sha256(path.read_bytes()).hexdigest() == digest
            verified += 1
        assert verified >= 4


def test_actual_probe_legacy_field_only_predicate_is_an_explained_measurement_gap():
    # Synthetic shape reproduction for implementation correctness only;
    # the actual 171295-row GPU-free S3 audit is in the immutable CI receipt.
    pack = _json(ROOT / "configs/eval_packs/semantic_behavior_v1.yaml")
    evidence, counts = analyze_dataset_task_coverage(
        [{"text":"<|prompt|>\\nreturn JSON\\n<|target|>\\n{\"ok\":true}"}],
        pack, source_ref="synthetic-fixture-not-an-authentic-source",
    )
    assert counts["prompt_target_rows"] == 0
    assert counts["structured_target_rows"] == 0
    assert evidence["data_coverage_gap"] is True
    real = _record()["source_inputs"][1]
    assert real["text_prompt_target_wrapped"] > 0
    assert real["structured_target_fields"] == 0
    # Under-count of wrapper markers does NOT establish overall complete coverage.


def test_diagnosis_gate_and_original_parent_precommit_are_not_bypassed():
    data = _record()
    assert data["science_authorization"]["full_original_prometheus_preregistered_v1_matrix"] is False
    for key in ("frozen_role_gpu_inference", "finetune", "granite_repair", "planner", "final_holdout"):
        assert data["science_authorization"][key] is False
    precommit = _json(ROOT / "configs/research/prometheus_diagnosis_evidence_contrast_v1.json")
    assert precommit["status"] == "FROZEN_PROTOCOL_NO_GPU_AUTHORIZATION"
    assert all(case["status"] != "authentic_source_admitted"
               for case in precommit["case_slots"])
    comparison = _json(ROOT / "docs/evidence/second-controlled-experiment-001-33889959923/second_controlled_semantic_evaluation_verification.json")["comparison"]
    assert comparison["primary_outcome"] == "regressed"
    assert comparison["deterministic_gate_status"] == "failed"



def test_new_predicate_gold_fixes_actual_observations_but_not_unknown_granite_cause():
    truth = _json(ROOT / "configs/research/prometheus_source_predicate_contrasts_v1.json")
    assert truth["status"] == (
        "FROZEN_RETROSPECTIVE_DEVELOPMENT_PREDICATE_LABELS_NOT_ORIGINAL_GRANITE_GOLD"
    )
    by_id = {p["id"]:p for p in truth["predicates"]}
    assert by_id["P_ORIG_EMBEDDED_PROMPT_TARGET"]["gold"] is False
    assert by_id["P_LATER_EMBEDDED_PROMPT_TARGET"]["gold"] is True
    assert by_id["P_PROBE_COUNTS_EMBEDDED_WRAPPERS"]["gold"] is False
    assert by_id["P_LATER_EMBEDDED_JSON_OBJECT_TARGET"]["gold"] is False
    assert by_id["P_ORIGINAL_GRANITE_EVAL_BROKEN"]["gold"] is None
    assert by_id["P_LATER_ALL_STRUCTURED_TASK_COVERAGE_VERIFIED"]["gold"] is None
    assert truth["control_policy"]["gold_domain_label_adjudication_complete"] is False
    assert truth["control_policy"]["permitted_for_original_granite_certification"] is False
    assert truth["control_policy"]["permitted_for_planner"] is False
