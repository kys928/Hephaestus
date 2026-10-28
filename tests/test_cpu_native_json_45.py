"""Proof that 96-step JSON 45-stage CPU research cannot downgrade original policy."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from hephaestus.schemas.cpu_campaign_config import CpuCampaignConfig
from hephaestus.schemas.cpu_research import CpuResearchExecutionProfile


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/cpu_native_json_45_v2.json"


def test_exact_v2_recipe_commitment_and_distinct_identity():
    spec = CpuCampaignConfig.from_file(CONFIG)
    assert spec.schema == "cpu-campaign-config.v2"
    assert spec.maximum_training_steps == 96
    assert spec.training_recipe == "chat_json_r8_full_epoch_96_v1"
    assert spec.include_public is False
    assert spec.stage_name == "targeted_repair"
    assert spec.recipe_sha256.startswith("sha256:")
    assert CpuCampaignConfig.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize("field,value", [
    ("recipe_sha256", "sha256:"+"a"*64),
    ("training_recipe", "legacy_raw_r2"),
    ("maximum_training_steps", 8),
    ("maximum_training_steps", 97),
    ("include_public", True),
    ("dataset_seed", 5),
    ("schema", "cpu-campaign-config.v1"),
])
def test_v2_recipe_or_original_dataset_cannot_change(field,value):
    with pytest.raises((ValueError, PermissionError)):
        replace(CpuCampaignConfig.from_file(CONFIG), **{field:value}).validate()


def test_legacy_cpu_profile_refuses_96_unless_explicit_versioned_recipe():
    dummy_hashes={k:"sha256:"+"a"*64 for k in
                  ("config.json","model.safetensors","tokenizer.json","tokenizer_config.json")}
    valid=CpuResearchExecutionProfile("p", "sha256:"+"b"*64,
       "HuggingFaceTB/SmolLM2-135M-Instruct",
       "12fd25f77366fa6b3b4b768ec3050bf629380bac",
       dummy_hashes, "sha256:"+"c"*64, 96, 3600, seed=1117,
       training_recipe="chat_json_r8_full_epoch_96_v1")
    valid.validate()
    with pytest.raises(PermissionError):
        replace(valid, training_recipe="legacy_raw_r2").validate()


def test_original_v1_demo_still_loads_unchanged():
    legacy=CpuCampaignConfig.from_file(ROOT/"configs/research/cpu_original_demo_v1.json")
    assert legacy.schema=="cpu-campaign-config.v1"
    assert legacy.training_recipe=="legacy_raw_r2"
    assert legacy.recipe_sha256 is None
    assert CpuCampaignConfig.from_dict(legacy.to_dict())==legacy
