import json
from pathlib import Path
import pytest
from scripts.analyze_cpu_prospective_replications import analyze_reports


def sample(seed):
    return {
        "workflow_id": f"wf-{seed}", "status": "completed", "completed_operations": 45,
        "evidence": {"verified_operations": 45, "verified_domain_operations": 40,
            "claim_scope": "internal_prospective_generated_task_research",
            "independent_scientific_certification": False, "five_frozen_cuda_roles_executed": False,
            "cpu_seed": seed, "dataset_seed": 77, "experiment_id": f"exp-{seed}",
            "actual_training": {"completed_optimizer_steps": 8, "weights_changed": True},
            "prospective": {"count": 24, "target_nll_only_diagnostic": True,
                "evaluation_item_commitment": "sha256:" + "a"*64,
                "correct_before": 0, "correct_after": 0, "json_before": 0, "json_after": 0,
                "mean_target_nll_before": 3.0, "mean_target_nll_after": 2.5}}}


def files(tmp_path, data):
    paths = []
    for i, d in enumerate(data):
        p = tmp_path / f"{i}.json"
        p.write_text(json.dumps(d))
        paths.append(p)
    return paths


def test_no_answer_improvement_is_not_misreported_as_correctness(tmp_path):
    r = analyze_reports(files(tmp_path, [sample(x) for x in (1,2,3)]))
    assert r["training_seed_count"] == 3
    assert r["nll_reduction_on_all_seeds"]
    assert not r["task_correctness_improvement_on_all_seeds"]
    assert r["model_promoted"] is False and r["external_certification"] is False


def test_different_holdout_or_reused_seed_must_fail(tmp_path):
    rows = [sample(x) for x in (1,2,3)]
    rows[2]["evidence"]["prospective"]["evaluation_item_commitment"] = "sha256:" + "b"*64
    with pytest.raises(ValueError, match="frozen"):
        analyze_reports(files(tmp_path, rows))
    rows = [sample(1), sample(1), sample(3)]
    with pytest.raises(ValueError, match="seed"):
        analyze_reports(files(tmp_path, rows))


def test_forged_or_incomplete_cpu_run_must_fail(tmp_path):
    rows = [sample(x) for x in (1,2,3)]
    rows[0]["evidence"]["actual_training"]["completed_optimizer_steps"] = 2
    with pytest.raises(ValueError, match="eight-step"):
        analyze_reports(files(tmp_path, rows))
