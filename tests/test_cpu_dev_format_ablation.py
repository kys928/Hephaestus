"""No model download required for precommitted CPU development protocol tests."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_dev_format", ROOT / "scripts" / "run_cpu_dev_format_ablation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_precommitted_dev_protocol_is_readable():
    p = module.check_protocol(ROOT / "configs/research/cpu_dev_format_ablation_v1.json")
    assert len(p["arms"]) == 4
    assert p["original_development_artifact_sha256"].startswith("sha256:")
    assert not p["GPU_execution_allowed"]


def test_unreviewed_mode_and_holdout_promotion_are_blocked(tmp_path):
    p = module.check_protocol(ROOT / "configs/research/cpu_dev_format_ablation_v1.json")
    p["arms"][0]["max_new_tokens"] = 128
    q = tmp_path / "protocol.json"
    q.write_text(json.dumps(p))
    with pytest.raises(PermissionError, match="controlled"):
        module.check_protocol(q)
    p["arms"][0]["max_new_tokens"] = 16
    p["model_promotion_allowed"] = True
    q.write_text(json.dumps(p))
    with pytest.raises(PermissionError, match="unscoped"):
        module.check_protocol(q)


def test_exact_json_answer_without_silent_format_forgiveness():
    assert module.exact_result('{"result":14}', '{"result":14}') == (True, True)
    assert module.exact_result("```json\n{\"result\":14}\n```", '{"result":14}') == (False, False)
    assert module.exact_result('{"result":"14"}', '{"result":14}') == (True, False)
    assert module.exact_result('{"wrong":14}', '{"result":14}') == (True, False)


def test_raw_prompt_unchanged_and_chat_template_explicit():
    class Tok:
        chat_template = "nonempty"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is True
            return "CHAT:" + messages[1]["content"]
    assert module.prompt_text(Tok(), "Compute 1 plus 2.", "original_raw") == "Compute 1 plus 2.\nAnswer: "
    assert module.prompt_text(Tok(), "Compute 1 plus 2.", "pinned_instruct_chat") == "CHAT:Compute 1 plus 2."
    with pytest.raises(ValueError):
        module.prompt_text(Tok(), "x", "unapproved")
