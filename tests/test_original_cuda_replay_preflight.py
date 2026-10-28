import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("gpu_preflight",ROOT/"scripts"/"preflight_original_cuda_replay.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def example(i,correct=False,valid=False):
    return {"sample_id":f"sample-{i}", "exact_correct":correct,"valid_json":valid}


def report(gain=True):
    a=[example(i) for i in range(24)]
    b=[example(i,correct=(gain and i==0),valid=(gain and i==0)) for i in range(24)]
    return {"schema":"cpu-chat-json-repair-result.v1",
      "claim_scope":"original_generated_development_only",
      "optimizer_steps":96,"epoch_completed":True,"weights_changed":True,
      "evaluation_holdout_used":False,"GPU_used":False,"model_promoted":False,
      "external_certification":False,"frozen_five_cuda_roles_executed":False,
      "baseline":{"n":24,"examples":a},
      "candidate":{"n":24,"examples":b},
      "development_exact_correct_change":int(gain),
      "development_valid_json_change":int(gain)}


def frozen():
    r,digest=mod.original_frozen_roles(ROOT/"docs"/"evidence"/"role-model-stack-v1"/"certification.json")
    assert len(r["roles"])==5 and digest.startswith("sha256:")
    return r,digest


def test_passing_cpu_evidence_only_proposes_gpu_without_certification():
    r,d=frozen()
    result=mod.evaluate(report(),{"archive_sha256":"sha256:"+"1"*64,
                                    "adapter_sha256":"sha256:"+"2"*64},r,d)
    assert result["cpu_exact_correct_gain"]==1
    assert result["gpu_execution_authorized"] is False
    assert result["gpu_executed"] is False
    assert result["independent_scientific_certification"] is False
    assert result["issue_102_closable"] is False
    assert len(result["frozen_roles"])==5


@pytest.mark.parametrize("change",["no_improvement","modified_eval","fake_cert","promoted","hard_regression"])
def test_missing_or_false_scientific_prerequisite_blocks_gpu(change):
    r,d=frozen()
    x=report()
    if change=="no_improvement":
        x=report(False)
    elif change=="modified_eval":
        x["candidate"]["examples"][0]["sample_id"]="replaced"
    elif change=="fake_cert":
        x["external_certification"]=True
    elif change=="promoted":
        x["model_promoted"]=True
    elif change=="hard_regression":
        x["baseline"]["examples"][1]["exact_correct"]=True
        x["baseline"]["examples"][1]["valid_json"]=True
    with pytest.raises((PermissionError,ValueError)):
        mod.evaluate(x,{"archive_sha256":"sha256:"+"1"*64,
                         "adapter_sha256":"sha256:"+"2"*64},r,d)
