from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from hephaestus.providers.models.generation_observation import GenerationObserver, token_termination


def test_eos_observation_excludes_padding_and_retains_eos():
    assert token_termination([7, 9, 2, 2, 2], 2, 96) == {
        "eos_observed": True, "actual_generated_tokens": 3, "actual_finish_reason": "eos"}
    assert token_termination([7] * 96, 2, 96)["actual_finish_reason"] == "length"
    assert token_termination([7], 2, 96)["actual_finish_reason"] == "unknown"


def test_passive_observer_returns_identical_tensor_and_decoding_arguments():
    class Row(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            return Row(value) if isinstance(index, slice) else value
        def tolist(self):
            return list(self)
    result = [Row([100, 101, 7, 2, 2])]
    class Inputs:
        shape = (1, 2)
    inputs = Inputs()
    class Model:
        def generate(self, **kwargs):
            assert kwargs == {"input_ids": inputs, "do_sample": False, "max_new_tokens": 96,
                              "eos_token_id": 2, "pad_token_id": 2}
            return result
    observer = GenerationObserver(Model())
    assert observer.generate(input_ids=inputs, do_sample=False, max_new_tokens=96,
                             eos_token_id=2, pad_token_id=2) is result
    assert observer.observations[0]["actual_generated_tokens"] == 2


def test_v5_bootstrap_frozen_protocol_and_native_router():
    # This bootstrap intentionally crosses the optional S3 execution boundary.
    # Generic/core CI does not install that extra; the dedicated V5 preflight
    # installs .[s3,test] and therefore executes this assertion in full.
    if importlib.util.find_spec("boto3") is None:
        pytest.skip("V5 bootstrap integration requires the optional s3 extra")

    # Script wrappers mutate their historical module globals, so isolate imports.
    code = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
import launch_positive_promotion_proof_v5 as launch
import run_positive_promotion_proof_v5 as run
import runpod_positive_promotion_driver_v5 as driver
spec = run.SPEC
plan = run.proof.EvaluationGenerationService(
    artifact_root=Path("unused-test-artifacts"), backend=None, pack_name="semantic_behavior_v1",
    config_dir=Path("configs")).plan()
assert plan.content_hash == spec["protocol"]["content_hash"]
assert plan.decoding_config == spec["protocol"]["decoding_config"]
assert len(plan.tasks) == 18
assert len(run.proof.CANDIDATES) == 2
assert all(c["judge_revision"] == "582efe62d7cfafd242bffca71ecbde1bcecc1bcc" for c in run.proof.CANDIDATES)
shell = launch.pod_shell_v5()
assert '\"$PY\" scripts/run_positive_promotion_proof_v5.py' in shell
assert '\"$PY\" scripts/run_positive_promotion_proof_v4.py' not in shell
assert 'requirements_sha256' in shell
assert "-r " in shell
assert "transformers>=" not in shell
class Execution:
    def _create_pod(self, body):
        assert body["gpuCount"] == 1
        assert body["cloudType"] == "SECURE"
        assert body["gpuTypePriority"] == "availability"
        assert body["containerDiskInGb"] == 400
        assert body["networkVolumeId"] == "cviwpryzao"
        assert body["dockerStartCmd"][-1] == shell
        return {"id": "test"}
assert driver.create_pod(Execution(), proof_run_id="test", repo_sha="a"*40, attempt=1,
                         controlled_bootstrap_failure=False)[0]["id"] == "test"
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)
