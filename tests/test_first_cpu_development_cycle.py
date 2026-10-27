"""Evidence-based tests for first end-to-end *experimental* CPU cycle."""
import copy
import json

import pytest

from hephaestus.development.cpu_cycle import (
    GENERATED_LICENSE_BASIS, isolate_data, plan_evaluation, source_data,
    verify_cycle, run_cycle, make_training_batch,
)
from hephaestus.evaluation.pack_loader import load_eval_pack


def test_created_corpus_has_no_third_party_data_and_source_is_reconstructible():
    original = source_data()
    assert len(original) == 100
    assert len({x["id"] for x in original}) == 100
    assert "no external" in GENERATED_LICENSE_BASIS.casefold()
    assert all(
        json.loads(x["response"]) == {"sum": x["left"] + x["right"]}
        for x in original
    )


def test_nonleaking_split_frozen_evaluation_plan_and_heldout_ground_truth():
    original = source_data()
    train, holdout, checks = isolate_data(original)
    assert (len(train), len(holdout)) == (80, 20)
    assert checks["train_id_overlap"] == 0
    assert checks["exact_prompt_target_overlap"] == 0
    frozen = load_eval_pack("semantic_behavior_v1")
    plan = plan_evaluation(original, holdout, frozen)
    assert plan["created_before_training"]
    assert not plan["holdout_claimed"]
    assert plan["regression_pack_unchanged"]
    assert plan["regression_pack_canonical_sha256"] == frozen["content_hash"]
    assert all(t["expected_sum"] in range(19) for t in plan["tasks"])
    tamper = copy.deepcopy(original)
    tamper[0]["response"] = '{"sum":999}'
    with pytest.raises(ValueError, match="ground-truth"):
        plan_evaluation(tamper, holdout, frozen)


def test_training_labels_mask_instruction_tokens_and_keep_answer():
    torch = pytest.importorskip("torch")
    x, y = make_training_batch(source_data()[:2])
    assert x.shape == y.shape
    assert int((y[0] != -100).sum()) > 4
    assert int((y[0] == -100).sum()) > 4
    assert torch.all((y[y != -100] >= 0) & (y[y != -100] < 256))


@pytest.mark.real_cpu
def test_full_real_optimizer_judge_and_replay_no_campaign_counter_move(tmp_path):
    pytest.importorskip("torch")
    dest = tmp_path / "cpu-development"
    outcome = run_cycle(dest, steps=4)
    assert outcome["optimizer_steps"] == 4
    assert outcome["weight_delta"] > 0
    assert outcome["checkpoint_sha256"].startswith("sha256:")
    assert outcome["original_campaign_advanced"] is False
    assert verify_cycle(dest)["valid"]
    judge = json.loads((dest / "judge_verdict.json").read_text())
    assert judge["judge_policy_action"] in ("reject_checkpoint", "continue_lineage_best")
    assert judge["training_approved_for_production"] is False
    assert judge["scientific_certification"] is False
    assert judge["independent_holdout_examined"] is False
    with pytest.raises(FileExistsError):
        run_cycle(dest, steps=4)
    candidate = dest / "candidate_evaluation.json"
    candidate.write_bytes(candidate.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="replay content mismatch"):
        verify_cycle(dest)


def test_training_settings_refuse_unbounded_optimization(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="cap"):
        run_cycle(tmp_path / "no-run", steps=9999)
