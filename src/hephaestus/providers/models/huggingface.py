"""Read-only Hugging Face model metadata provider for governed discovery.

Network access is opt-in and metadata only.  It never imports model repository
code and it never turns a floating revision into provenance: the Hub response
must expose an immutable commit SHA.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import quote

from hephaestus.schemas.discovery_contract import ModelCandidate, ModelSearchRequest


@dataclass(slots=True)
class HuggingFaceModelProvider:
    provider_id: str = "huggingface"
    enable_network: bool = False
    endpoint: str = "https://huggingface.co"
    timeout_seconds: float = 15.0

    def search(self, request: ModelSearchRequest) -> tuple[ModelCandidate, ...]:
        if request.provider_allowlist and self.provider_id not in request.provider_allowlist:
            return ()
        if not self.enable_network:
            raise RuntimeError("Hugging Face model discovery network access is disabled")
        raw_ids = request.metadata.get("model_ids", [])
        model_ids = [str(item).strip() for item in raw_ids] if isinstance(raw_ids, list) else []
        single = str(request.metadata.get("model_id", "")).strip()
        if single:
            model_ids.append(single)
        model_ids = sorted(set(item for item in model_ids if item))
        if not model_ids:
            raise ValueError("Hugging Face model discovery requires explicit metadata.model_ids")
        requested_revision = str(request.metadata.get("revision", "main")).strip() or "main"
        return tuple(self._candidate(model_id, requested_revision) for model_id in model_ids)

    def _candidate(self, model_id: str, requested_revision: str) -> ModelCandidate:
        path = f"/api/models/{quote(model_id, safe='/')}/revision/{quote(requested_revision, safe='')}"
        request = urllib.request.Request(
            f"{self.endpoint.rstrip('/')}{path}",
            headers={"Accept": "application/json", "User-Agent": "hephaestus-model-discovery/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Hugging Face model metadata unavailable for {model_id}: {type(exc).__name__}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Hugging Face model response is not an object")
        revision = str(payload.get("sha") or "").strip()
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        card = payload.get("cardData") if isinstance(payload.get("cardData"), dict) else {}
        safetensors = payload.get("safetensors") if isinstance(payload.get("safetensors"), dict) else {}
        parameters = safetensors.get("total")
        try:
            parameter_count = int(parameters) if parameters is not None else None
        except (TypeError, ValueError):
            parameter_count = None
        architectures = config.get("architectures") if isinstance(config, dict) else None
        architecture = (
            str(architectures[0])
            if isinstance(architectures, list) and architectures
            else str(config.get("model_type") or "") if isinstance(config, dict) else ""
        ) or None
        context = None
        for key in ("max_position_embeddings", "n_positions", "seq_length"):
            value = config.get(key) if isinstance(config, dict) else None
            try:
                if value:
                    context = int(value)
                    break
            except (TypeError, ValueError):
                pass
        license_value = card.get("license") or payload.get("license")
        pipeline = str(payload.get("pipeline_tag") or "")
        return ModelCandidate(
            candidate_id=f"huggingface:{model_id}@{revision or 'unknown'}",
            provider_id=self.provider_id,
            model_id=model_id,
            revision=revision or None,
            architecture_family=architecture,
            parameter_count=parameter_count,
            context_length=context,
            tokenizer_ref=model_id,
            license=str(license_value) if license_value else None,
            capabilities=sorted(set([item for item in (pipeline, "causal_lm") if item])),
            runtime_requirements={"supported_backends": ["transformers_causal_lm"]},
            compatibility={
                "immutable_revision": len(revision) in {40, 64},
                "remote_code_required": False,
                "smoke_test": True,
            },
            artifact_ref=f"hf://models/{model_id}@{revision}" if revision else None,
            evidence_refs=[f"{self.endpoint.rstrip('/')}/api/models/{model_id}/revision/{revision or requested_revision}"],
            missing_metadata=[
                name
                for name, value in (
                    ("revision", revision),
                    ("architecture_family", architecture),
                    ("parameter_count", parameter_count),
                    ("context_length", context),
                    ("license", license_value),
                )
                if value in (None, "")
            ],
            score_components={"provider_evidence": 1.0, "metadata_completeness": 1.0},
            metadata={
                "requested_revision": requested_revision,
                "pipeline_tag": pipeline,
                "downloads": payload.get("downloads"),
                "likes": payload.get("likes"),
                "provenance_ref": f"hf://models/{model_id}@{revision}" if revision else None,
            },
        )
