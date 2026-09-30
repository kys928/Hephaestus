from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LAUNCHER = SCRIPTS / "launch_interface_repair_v3_evaluator.py"
CFG = ROOT / "configs/experiments/hephaestus_interface_repair_v3_evaluator.json"
MARKER = ROOT / "configs/experiments/interface_repair_v3_evaluator.launch.json"


def import_launcher():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("interface_repair_v3_launcher", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v3_launcher_is_evaluator_only_and_reuses_safe_bootstrap() -> None:
    launcher = import_launcher()
    cfg = load(CFG)
    rendered = launcher.request_body(cfg, "deadbeef", "dry-run", "full", real_env=False)
    assert rendered["gpuCount"] == 1
    assert rendered["gpuTypeIds"] == cfg["execution"]["gpu_type_ids"]
    assert rendered["networkVolumeId"] == "<volume>"
    assert rendered["volumeMountPath"] == "/workspace"
    shell = rendered["dockerStartCmd"][2]
    assert "export HEPHAESTUS_REPAIR_ROLE=evaluator" in shell
    assert "run_interface_repair_v3_evaluator.py" in shell
    assert "build_interface_repair_v3_evaluator.py" in shell
    assert "for ROLE in planner evaluator judge controller" not in shell
    assert "run_interface_repair_live_v2.py" not in shell
    assert cfg["execution"]["persistent_venv_cache"] in shell
    assert cfg["execution"]["persistent_adapter_cache"] in shell


def test_v3_preflight_shell_is_small_and_evaluator_pinned() -> None:
    launcher = import_launcher()
    cfg = load(CFG)
    rendered = launcher.request_body(cfg, "deadbeef", "dry-preflight", "preflight", real_env=False)
    shell = rendered["dockerStartCmd"][2]
    assert "export HEPHAESTUS_REPAIR_ROLE=evaluator" in shell
    assert "run_interface_repair_v3_evaluator.py --preflight" in shell
    assert cfg["preflight"]["optimizer_steps"] == 5
    assert cfg["preflight"]["evaluation_cases"] == 8


def test_v3_paid_execution_starts_locked() -> None:
    cfg = load(CFG)
    marker = load(MARKER)
    assert cfg["governance"]["paid_preflight_allowed"] is False
    assert cfg["governance"]["paid_full_launch_allowed"] is False
    assert marker["preflight_authorized"] is False
    assert marker["full_launch_authorized"] is False
    assert marker["preflight_evidence_key"] is None
    assert marker["preflight_repo_sha"] is None


def test_v3_full_binding_allows_only_authorization_files() -> None:
    launcher = import_launcher()
    assert launcher.ALLOWED_AUTH_CHANGED_FILES == {
        "configs/experiments/hephaestus_interface_repair_v3_evaluator.json",
        "configs/experiments/interface_repair_v3_evaluator.launch.json",
    }
    cfg = load(CFG)
    normalized = launcher._normalized_science_config(cfg)
    assert "paid_preflight_allowed" not in normalized["governance"]
    assert "paid_full_launch_allowed" not in normalized["governance"]
    assert normalized["training"] == cfg["training"]
    assert normalized["certification"] == cfg["certification"]
    assert normalized["pack"] == cfg["pack"]
