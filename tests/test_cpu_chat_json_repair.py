import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("cpu_json_repair",ROOT/"scripts"/"run_cpu_chat_json_repair.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_original_repair_protocol_is_precommitted():
    config=mod.validate_protocol(ROOT/"configs"/"research"/"cpu_chat_json_repair_v1.json")
    assert config["optimizer_steps"]==96
    assert config["training_examples"]==96
    assert config["development_examples"]==24
    assert config["evaluation_holdout_allowed"] is False
    assert config["paid_gpu_authorized"] is False


@pytest.mark.parametrize("field,replacement",[
    ("optimizer_steps",192),("model_revision","latest"),("epoch_count",2),
    ("development_examples",48),("evaluation_holdout_allowed",True),
    ("learning_rate",0.1),("model_promotion_authorized",True),("train_seed",1118)])
def test_mutating_any_scientific_variable_denied(tmp_path,field,replacement):
    data=mod.validate_protocol(ROOT/"configs"/"research"/"cpu_chat_json_repair_v1.json")
    data[field]=replacement
    path=tmp_path/"mutated.json"
    path.write_text(json.dumps(data))
    with pytest.raises(PermissionError):
        mod.validate_protocol(path)


def test_strict_json_scoring_never_strips_fences_or_casts_types():
    assert mod.check_json('{"result":3}', '{"result":3}') == (True,True)
    assert mod.check_json('```json\n{"result":3}\n```','{"result":3}') == (False,False)
    assert mod.check_json('{"result":"3"}','{"result":3}') == (True,False)
    assert mod.check_json('{"result":4}','{"result":3}') == (True,False)


def test_pinned_chat_format_is_required():
    class Tokenizer:
        chat_template="real"
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs["add_generation_prompt"] is True
            assert messages[0]["role"]=="system"
            return messages[1]["content"]
    assert mod.chat_prompt(Tokenizer(),"Compute 4 plus 3")=="Compute 4 plus 3"
    class NoTemplate:
        chat_template=None
    with pytest.raises(ValueError):
        mod.chat_prompt(NoTemplate(),"Compute 4 plus 3")
