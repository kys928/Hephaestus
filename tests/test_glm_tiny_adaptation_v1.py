from __future__ import annotations
import importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path); assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
validator=load_module("tiny_validator","scripts/validate_glm_tiny_adaptation_v1.py")
launcher=load_module("tiny_launcher","scripts/launch_glm_tiny_adaptation_v1.py")

def protocol(): return json.loads((ROOT/"configs/experiments/hephaestus_glm_tiny_adaptation_v1.json").read_text())

def test_protocol_is_bounded_and_disabled_now():
    s=validator.validate(); assert s["status"]=="valid"; assert s["optimizer_steps"]==3; assert s["training_examples"]==12; assert s["post_probe_count"]==5; assert s["post_max_new_tokens"]<=3000; assert s["paid_launch_allowed_now"] is False

def test_launcher_uses_only_cheap_gpu_and_no_volume():
    p=protocol(); body=launcher.build_pod_request(p,name="tiny",env=launcher.placeholder_environment("1"*40,"tiny","screen/result.json")); launcher.validate_pod_request(p,body)
    assert all("RTX PRO 6000 Blackwell" in x for x in body["gpuTypeIds"]); assert not any("H200" in x or "B200" in x for x in body["gpuTypeIds"]); assert "networkVolumeId" not in body

def test_tiny_runner_is_three_steps_not_full_recovery():
    src=(ROOT/"scripts/run_glm_tiny_adaptation_v1.py").read_text()
    assert "optimizer_steps" in src; assert "run_diagnostic_scaling_recovery_v1.py" not in src; assert "advance_to_full_recovery" in src; assert "save_pretrained" in src

def test_prerequisite_requires_passed_screen():
    runner=load_module("tiny_runner_test","scripts/run_glm_tiny_adaptation_v1.py"); p=protocol()
    good={"result_version":"hephaestus-glm-cheap-screen.v1","status":"completed","advance_to_tiny_adaptation":True,"screening_only":True,"training_performed":False,"model_id":p["model"]["model_id"],"revision":p["model"]["revision"]}
    runner.validate_prerequisite(good,p)
    bad=dict(good); bad["advance_to_tiny_adaptation"]=False
    try: runner.validate_prerequisite(bad,p)
    except RuntimeError: pass
    else: raise AssertionError("non-passing cheap screen must block tiny adaptation")