"""F09: composition must use one explicit configuration root from any CWD."""
import json
from pathlib import Path
import shutil

from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings


def test_supplied_absolute_config_root_is_shared_by_planner_and_evaluator(tmp_path, monkeypatch):
    config = tmp_path / "custom-config"
    shutil.copytree("configs", config)
    policy_path = config / "policies/approval_policy.yaml"
    payload = json.loads(policy_path.read_text())
    payload["action_rules"]["branch_new_experiment"] = "approval_required_high_risk"
    policy_path.write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    runtime = ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=tmp_path / "state", artifact_root=tmp_path / "artifacts", config_dir=config)).build()
    assert runtime.planner.policy.approval_policy is runtime.approval_policy
    assert runtime.recovery.policy.approval_policy is runtime.approval_policy
    assert runtime.recovery.stage_policy.config_dir == config
    assert runtime.approval_policy.config_dir == config
    assert runtime.evaluator.config_dir == config
    assert runtime.coordinator.evaluation_service is runtime.evaluator
    assert runtime.planner.policy.approval_policy.action_rules["branch_new_experiment"] == "approval_required_high_risk"
