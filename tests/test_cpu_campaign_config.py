"""No GPU, network, or model download needed to verify campaign identities."""
from dataclasses import replace
import json
import pytest
from hephaestus.schemas.cpu_campaign_config import CpuCampaignConfig


def campaign():
    return CpuCampaignConfig("wf-a", "run-a", "lineage-a", "targeted_repair",
                             "repair-a", 17, 42, 4, 600, False).validate()


def test_multiple_distinct_campaign_identities_are_valid():
    first = campaign()
    second = replace(first, workflow_id="wf-b", run_id="run-b",
                     lineage_id="lineage-b", experiment_id="repair-b", cpu_seed=23)
    assert second.validate().identity != first.identity
    assert CpuCampaignConfig.from_dict(first.to_dict()) == first


@pytest.mark.parametrize("field,value", [
    ("workflow_id", "../escape"), ("run_id", ""), ("lineage_id", "a/b"),
    ("stage_name", "smoke test"), ("experiment_id", " spaces "),
    ("cpu_seed", -1), ("dataset_seed", True), ("maximum_training_steps", 0),
    ("maximum_training_steps", 9), ("maximum_cpu_seconds", 3601),
    ("include_public", 1), ("schema", "untrusted.v2")
])
def test_bad_identity_or_resource_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(campaign(), **{field:value}).validate()


def test_unknown_keys_and_oversized_spec_are_rejected(tmp_path):
    d = campaign().to_dict()
    with pytest.raises(ValueError):
        CpuCampaignConfig.from_dict({**d, "approval_granted": True})
    p = tmp_path / "request.json"
    p.write_text(json.dumps(d))
    assert CpuCampaignConfig.from_file(p) == campaign()
    p.write_bytes(b" " * 20000)
    with pytest.raises(ValueError):
        CpuCampaignConfig.from_file(p)
