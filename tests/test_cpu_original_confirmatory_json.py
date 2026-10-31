"""Fast tests for precommitted one-shot original generated evaluation."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("original_confirmatory",
    ROOT/"scripts/run_cpu_original_confirmatory_json.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_original_adapter_and_evaluation_selection_are_precommitted():
    p=mod.validated_config(ROOT/"configs/research/cpu_original_json_confirmatory_v1.json")
    assert p["train_seed"]==1117
    assert p["evaluation_sample_count"]==24
    assert p["original_five_cuda_gpu_executed"] is False
    assert p["external_certification"] is False


@pytest.mark.parametrize("key,value",[
 ("train_seed",3337),("max_new_tokens",128),
 ("evaluation_sample_count",96),("source_artifact_id",0),
 ("model_promoted",True),("original_five_cuda_gpu_executed",True)])
def test_refuse_to_rewrite_frozen_confirmatory_protocol(tmp_path,key,value):
    p=mod.validated_config(ROOT/"configs/research/cpu_original_json_confirmatory_v1.json")
    p[key]=value
    a=tmp_path/"altered.json"
    a.write_text(json.dumps(p))
    with pytest.raises(PermissionError):
        mod.validated_config(a)


def test_strict_json_answer_types_and_no_code_fence_laundering():
    assert mod.check_output('{"result":4}','{"result":4}') == (True,True)
    assert mod.check_output('{"result":"4"}','{"result":4}') == (True,False)
    assert mod.check_output('```json\n{"result":4}\n```','{"result":4}') == (False,False)
