"""Actual existing PEFT/generation adapter, strict approval boundaries, replay."""
import json
from pathlib import Path
import pytest

from hephaestus.development.existing_peft_bridge import (
    _lexicon, prepare_experiment, run_existing_peft_cycle, verify_peft_cycle,
)
from hephaestus.development.cpu_cycle import source_data


def test_reviewed_development_source_is_original_not_hf_acquisition():
    source=source_data()
    assert len(source)==100
    assert len({row["id"] for row in source})==100


def test_cannot_launch_unbounded_real_peft_or_overwrite_evidence(tmp_path):
    with pytest.raises(ValueError,match="1..12"):
        prepare_experiment(tmp_path/"refused",steps=1000)


def test_development_scope_never_claims_model_role_or_data_acquisition_approval(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    root=tmp_path/"development"
    staged=prepare_experiment(root,steps=2)
    contract=staged["proposal"].training_constraints
    assert staged["plan"]["created_before_training"]
    assert staged["plan"]["multimodel_frozen_cuda_roles_executed"] is False
    assert staged["proposal"].metadata["production_staged_acquisition_approval"] is False
    assert staged["proposal"].required_approvals==[]
    assert contract["local_files_only"]
    evidence=json.loads(Path(contract["processing_evidence_ref"]).read_text())
    assert evidence["production_approval_ref"] is None
    assert evidence["tokenizer_compatibility"]["compatible"] is True
    assert evidence["tokenizer_compatibility"]["unknown_token_count"]==0


@pytest.mark.real_cpu
def test_actual_existing_peft_training_and_generation_judge_replay(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("peft")
    root=tmp_path/"peft-cycle"
    answer=run_existing_peft_cycle(root,steps=2,timeout_seconds=150)
    assert answer["status"]=="completed_noncertified_development"
    assert answer["training_steps"]==2
    assert answer["replay"]["valid"]
    assert answer["verdict"]["judge_policy_action"] in ("continue_lineage_best","reject_checkpoint")
    assert answer["verdict"]["scientific_certified"] is False
    report=json.loads((root/"peft_cycle_manifest.json").read_text())
    assert report["actual_training_adapter"]=="TransformersTrainingLifecycleService"
    assert report["actual_generation_adapter"]=="TransformersCausalLMGenerationBackend"
    assert report["frozen_five_role_production_models_not_loaded"] is True
    assert report["original_governed_45_stage_progress"]=="11/45_unchanged"
    assert "lora_B" in " ".join(report["peft_weight_names"])
    assert verify_peft_cycle(root)["real_checkpoint_manifest_verified"]
    f=root/"candidate_generation.json"
    f.write_bytes(f.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="output modified"):
        verify_peft_cycle(root)
