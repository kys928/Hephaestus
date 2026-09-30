from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from hephaestus.providers.models.admission import validate_hub_metadata, validate_identity
from hephaestus.providers.models.huggingface import HuggingFaceModelProvider
from hephaestus.providers.models.selection import DeterministicModelSelectionService
from hephaestus.schemas.discovery_contract import ModelSearchRequest


def spec():
    return {"model_id": "owner/model", "revision": "a" * 40, "license": "apache-2.0",
            "architecture": "Qwen3ForCausalLM", "model_type": "qwen3", "source_dtype": "bfloat16"}


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39, "z" * 40, "a" * 64])
def test_admission_rejects_floating_and_malformed_revisions(revision):
    with pytest.raises(ValueError, match="immutable"):
        validate_identity({**spec(), "revision": revision})


@pytest.mark.parametrize("license_id", ["unknown", "other", "lfm1.0"])
def test_admission_rejects_licenses_outside_governed_allowlist(license_id):
    with pytest.raises(ValueError, match="license"):
        validate_identity({**spec(), "license": license_id})


@pytest.mark.parametrize("change", [
    {"architectures": ["UnknownModel"]}, {"model_type": "unknown"},
    {"dtype": "float8"}, {"quantization_config": {"quant_method": "fp8"}},
    {"text_config": {"quantization_config": {"quant_method": "gptq"}}},
])
def test_admission_rejects_architecture_dtype_and_quantization_drift(change):
    config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "dtype": "bfloat16"}
    with pytest.raises(ValueError):
        validate_hub_metadata(spec(), revision="a" * 40, license_id="apache-2.0",
                              config={**config, **change})


def test_metadata_discovery_cannot_claim_a_real_smoke_test(monkeypatch):
    payload = {"sha": "a" * 40, "cardData": {"license": "apache-2.0"},
               "config": {"architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 4096},
               "safetensors": {"total": 1000}, "pipeline_tag": "text-generation"}
    class Response:
        def read(self):
            return json.dumps(payload).encode()
    @contextmanager
    def fake_open(*args, **kwargs):
        yield Response()
    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    request = ModelSearchRequest(
        request_id="request", diagnosis_report_id="diagnosis", problem_statement="runtime proof",
        metadata={"model_id": "owner/model"}, license_allowlist=["apache-2.0"],
        runtime_constraints={"smoke_test_required": True},
    )
    candidate = HuggingFaceModelProvider(enable_network=True).search(request)[0]
    assert candidate.compatibility["smoke_test"] is not True
    assert candidate.compatibility["remote_code_required"] is None
    decision = DeterministicModelSelectionService().select(request, [candidate])
    assert decision.status == "blocked"
    assert "smoke_test_unsuitable" in decision.rejected_candidates[candidate.candidate_id]
