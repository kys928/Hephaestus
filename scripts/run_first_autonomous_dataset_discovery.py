#!/usr/bin/env python3
"""Discover, audit, select, and plan the first post-failure dataset intervention.

This integration driver consumes the exact frozen DatasetSearchRequest produced by
Hephaestus after the first failed checkpoint. It uses the production Hugging Face
metadata provider, resolves immutable provenance, performs a bounded public
Dataset Server preview audit, validates preview text against the exact frozen
training tokenizer, runs the existing deterministic dataset selector, and only
then allows the existing Planner to create the next one-primary-variable
ExperimentProposal.

It does NOT acquire a dataset, mutate the Network Volume, launch training, apply
an approval, or execute the proposed experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import boto3
from botocore.config import Config
from tokenizers import Tokenizer

from hephaestus.data import DatasetProviderRegistry, DeterministicDatasetSelectionService
from hephaestus.planning import ResolvedEvidenceExperimentPlanner
from hephaestus.providers.datasets import HuggingFaceDatasetProvider
from hephaestus.schemas.diagnosis_contract import DiagnosisReport
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest
from hephaestus.schemas.experiment_contract import InterventionProposal

ROOT = Path(__file__).resolve().parents[1]
BRANCH_BASE = "2d285eb8db75a6a310a9ea2a14a5ee534a8bfce0"
TRAIN_RUN = "first-bounded-scientific-training-001-33866198758"
TOKENIZER_ID = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
TOKENIZER_KEY = (
    f"hephaestus/scientific/v1/runs/{TRAIN_RUN}/checkpoint_step_100/tokenizer/tokenizer.json"
)
FROZEN_PLANNER = ROOT / "docs/evidence/first-targeted-post-failure-diagnostics-001-33874832481/planner.json"
FROZEN_DIAGNOSIS = ROOT / "docs/evidence/first-targeted-post-failure-diagnostics-001-33874832481/diagnosis.json"
FROZEN_PLANNER_SHA = "52e6db0f5c4f9297e763ed11581f98a6e7c92af45e45ca5e6922504e8f9b3325"
FROZEN_DIAGNOSIS_SHA = "11cb67960df8c9f60a861e57b4f74145ad64cd25f5b52bb3835c754cde74f086"
EXPECTED_REQUEST_ID = "dataset-search-4c7158b167aff959"
EXPECTED_INTERVENTION_ID = "intervention-0b05ae5169cd5943"
MAX_DISCOVERED = 30
MAX_ENRICHED = 14
PREVIEW_TIMEOUT_SECONDS = 15.0
PREVIEW_ROWS_LIMIT = 100
DATA_SUFFIXES = (
    ".jsonl", ".json", ".csv", ".parquet", ".arrow", ".txt", ".text", ".gz", ".zst", ".zip"
)

PROMPT_KEYS = {
    "instruction", "prompt", "question", "query", "input", "user", "request", "problem"
}
RESPONSE_KEYS = {
    "response", "output", "answer", "completion", "assistant", "target", "chosen", "label"
}
CHAT_KEYS = {"messages", "conversations", "conversation", "dialogue", "dialog", "chat"}
CONSTRAINT_PATTERNS = (
    "exactly", "at most", "no more than", "one sentence", "short sentence", "reply with",
    "respond with", "return json", "json with", "in json", "under ", "maximum ",
)
TERMINAL_CHARS = set(".!?)]}\"'")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def atomic_json(path: Path, payload: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 4, "mode": "standard"}),
    )


def get_s3_bytes(client: Any, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def request_json(url: str) -> object:
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hephaestus-first-autonomous-dataset-discovery/1",
        },
    )
    try:
        with urlopen(req, timeout=PREVIEW_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"preview request failed: {type(exc).__name__}: {exc}") from exc


def load_frozen_chain() -> tuple[DatasetSearchRequest, DiagnosisReport, InterventionProposal]:
    if sha_file(FROZEN_PLANNER) != FROZEN_PLANNER_SHA:
        raise RuntimeError("frozen targeted planner evidence drifted")
    if sha_file(FROZEN_DIAGNOSIS) != FROZEN_DIAGNOSIS_SHA:
        raise RuntimeError("frozen targeted diagnosis evidence drifted")
    planner_payload = json.loads(FROZEN_PLANNER.read_text(encoding="utf-8"))
    diagnosis_payload = json.loads(FROZEN_DIAGNOSIS.read_text(encoding="utf-8"))
    raw_request = planner_payload.get("dataset_search_request")
    raw_intervention = planner_payload.get("selected_intervention")
    if not isinstance(raw_request, dict) or not isinstance(raw_intervention, dict):
        raise RuntimeError("frozen planner does not contain the expected dataset boundary")
    request = DatasetSearchRequest(**raw_request)
    diagnosis = DiagnosisReport.from_dict(diagnosis_payload)
    intervention = InterventionProposal.from_dict(raw_intervention)
    if request.request_id != EXPECTED_REQUEST_ID:
        raise RuntimeError(f"unexpected DatasetSearchRequest: {request.request_id}")
    if intervention.intervention_id != EXPECTED_INTERVENTION_ID:
        raise RuntimeError(f"unexpected intervention: {intervention.intervention_id}")
    if intervention.primary_variable != "dataset_mixture":
        raise RuntimeError("expected dataset_mixture as the one primary variable")
    return request, diagnosis, intervention


def discovery_queries(request: DatasetSearchRequest) -> list[str]:
    joined = " ".join(
        [request.problem_statement, *request.capability_targets, *request.required_formats, *request.required_domains]
    ).casefold()
    queries = ["instruction"]
    if "instruction" in joined:
        queries.extend(["instruction tuning", "instruction response", "assistant conversation"])
    if "json" in joined or "structured" in joined:
        queries.extend(["json instruction", "structured response"])
    if "causal" in joined or "continuation" in joined:
        queries.append("text generation")
    if "non-repetition" in joined or "termination" in joined:
        queries.append("short response instruction")
    return list(dict.fromkeys(query.strip() for query in queries if query.strip()))


def derived_request(base: DatasetSearchRequest, query: str, index: int) -> DatasetSearchRequest:
    payload = base.to_dict()
    payload["request_id"] = f"{base.request_id}-discovery-{index:02d}"
    payload["problem_statement"] = query
    payload["capability_targets"] = []
    payload["provider_allowlist"] = ["huggingface"]
    payload["metadata"] = {**dict(base.metadata), "parent_request_id": base.request_id, "query": query}
    return DatasetSearchRequest(**payload)


def metadata_discovery(request: DatasetSearchRequest, provider: HuggingFaceDatasetProvider) -> tuple[list[DatasetCandidate], list[dict[str, object]]]:
    registry = DatasetProviderRegistry(provider_allowlist={"huggingface"})
    registry.register(provider)
    candidates: dict[str, DatasetCandidate] = {}
    attempts: list[dict[str, object]] = []
    for index, query in enumerate(discovery_queries(request), start=1):
        child = derived_request(request, query, index)
        result = registry.discover(child)
        attempts.append(
            {
                "query": query,
                "request_id": child.request_id,
                "provider_ids": list(result.provider_ids),
                "candidate_count": len(result.candidates),
                "issues": [issue.to_dict() for issue in result.issues],
            }
        )
        for candidate in result.candidates:
            key = f"{candidate.provider_id}:{candidate.dataset_id}"
            previous = candidates.get(key)
            if previous is None:
                candidates[key] = candidate
            else:
                previous.task_types = sorted(set(previous.task_types + candidate.task_types))
                previous.languages = sorted(set(previous.languages + candidate.languages))
                previous.evidence_refs = sorted(set(previous.evidence_refs + candidate.evidence_refs))
        if len(candidates) >= MAX_DISCOVERED:
            break
    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            0 if item.license else 1,
            0 if item.revision else 1,
            -int(item.metadata.get("downloads") or 0),
            item.dataset_id.casefold(),
        ),
    )[:MAX_DISCOVERED]
    return ordered, attempts


def hf_splits(dataset_id: str) -> list[dict[str, object]]:
    payload = request_json(
        "https://datasets-server.huggingface.co/splits?" + urlencode({"dataset": dataset_id})
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("splits"), list):
        return []
    return [dict(item) for item in payload["splits"] if isinstance(item, dict)]


def hf_first_rows(dataset_id: str, config: str, split: str) -> list[dict[str, object]]:
    payload = request_json(
        "https://datasets-server.huggingface.co/first-rows?"
        + urlencode({"dataset": dataset_id, "config": config, "split": split})
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return []
    rows: list[dict[str, object]] = []
    for item in payload["rows"][:PREVIEW_ROWS_LIMIT]:
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            rows.append({str(key): value for key, value in item["row"].items()})
    return rows


def flatten_text(value: object, *, limit: int = 32) -> list[str]:
    output: list[str] = []
    stack: list[object] = [value]
    while stack and len(output) < limit:
        current = stack.pop()
        if isinstance(current, str):
            text = current.strip()
            if text:
                output.append(text)
        elif isinstance(current, dict):
            stack.extend(reversed(list(current.values())))
        elif isinstance(current, (list, tuple)):
            stack.extend(reversed(list(current)))
        elif current is not None and isinstance(current, (int, float, bool)):
            output.append(str(current))
    return output


def row_prompt_response(row: dict[str, object]) -> tuple[list[str], list[str], bool]:
    prompts: list[str] = []
    responses: list[str] = []
    chat_pair = False
    for key, value in row.items():
        lowered = key.casefold()
        if lowered in PROMPT_KEYS:
            prompts.extend(flatten_text(value))
        if lowered in RESPONSE_KEYS:
            responses.extend(flatten_text(value))
        if lowered in CHAT_KEYS and isinstance(value, list):
            user_seen = False
            assistant_seen = False
            for message in value:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or message.get("from") or "").casefold()
                content = message.get("content") if "content" in message else message.get("value")
                texts = flatten_text(content)
                if role in {"user", "human", "instruction"}:
                    prompts.extend(texts)
                    user_seen = user_seen or bool(texts)
                elif role in {"assistant", "gpt", "bot"}:
                    responses.extend(texts)
                    assistant_seen = assistant_seen or bool(texts)
            chat_pair = chat_pair or (user_seen and assistant_seen)
    return prompts, responses, chat_pair


def repeated_bigram_fraction(text: str) -> float:
    tokens = re.findall(r"\S+", text.casefold())
    if len(tokens) < 3:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def coverage_audit(rows: list[dict[str, object]], tokenizer: Tokenizer) -> dict[str, object]:
    total = len(rows)
    instruction_pairs = 0
    structured = 0
    explicit_constraints = 0
    terminated = 0
    non_repetitive = 0
    continuation = 0
    response_count = 0
    tokenizer_text_count = 0
    token_count = 0
    unk_count = 0
    over_context = 0
    unk_id = tokenizer.token_to_id("<|unk|>")

    for row in rows:
        prompts, responses, chat_pair = row_prompt_response(row)
        if prompts and responses:
            instruction_pairs += 1
        if chat_pair:
            instruction_pairs += 0  # pair is already represented above; retained as explicit detection
        prompt_text = "\n".join(prompts)
        for response in responses:
            response_count += 1
            stripped = response.strip()
            if stripped and stripped[-1] in TERMINAL_CHARS:
                terminated += 1
            if repeated_bigram_fraction(stripped) <= 0.15:
                non_repetitive += 1
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    structured += 1
            except (json.JSONDecodeError, TypeError):
                pass
        if prompt_text and any(pattern in prompt_text.casefold() for pattern in CONSTRAINT_PATTERNS):
            explicit_constraints += 1
        if any(key.casefold() in {"text", "completion", "continuation"} for key in row):
            continuation += 1

        texts = flatten_text(row, limit=16)
        for text in texts:
            encoding = tokenizer.encode(text)
            ids = encoding.ids
            if not ids:
                continue
            tokenizer_text_count += 1
            token_count += len(ids)
            if unk_id is not None:
                unk_count += sum(1 for token_id in ids if token_id == unk_id)
            if len(ids) > 256:
                over_context += 1

    denom = max(total, 1)
    response_denom = max(response_count, 1)
    token_denom = max(token_count, 1)
    text_denom = max(tokenizer_text_count, 1)
    unk_fraction = unk_count / token_denom
    tokenizer_compatible = tokenizer_text_count > 0 and unk_fraction <= 0.01
    capability = {
        "instruction adherence": instruction_pairs / denom,
        "structured JSON response": structured / response_denom,
        "bounded response length": explicit_constraints / denom,
        "termination": terminated / response_denom,
        "non-repetition": non_repetitive / response_denom,
        "causal-LM continuation": continuation / denom,
    }
    semantic_formats: list[str] = []
    if instruction_pairs:
        semantic_formats.append("instruction_response")
    if structured:
        semantic_formats.append("structured_json")
    if continuation or not instruction_pairs:
        semantic_formats.append("causal_lm_text")
    return {
        "scope": "bounded_public_preview_heuristic_not_full_dataset_proof",
        "rows_previewed": total,
        "instruction_pair_rows": instruction_pairs,
        "response_count": response_count,
        "structured_response_count": structured,
        "explicit_constraint_rows": explicit_constraints,
        "terminated_response_count": terminated,
        "non_repetitive_response_count": non_repetitive,
        "continuation_rows": continuation,
        "capability_coverage": {key: round(value, 6) for key, value in capability.items()},
        "semantic_formats": semantic_formats,
        "tokenizer": {
            "tokenizer_ref": TOKENIZER_ID,
            "texts_checked": tokenizer_text_count,
            "tokens_checked": token_count,
            "unknown_token_count": unk_count,
            "unknown_token_fraction": round(unk_fraction, 8),
            "over_context_text_count": over_context,
            "over_context_fraction": round(over_context / text_denom, 8),
            "compatible": tokenizer_compatible,
            "criterion": "preview unknown-token fraction <= 0.01; over-context rows are preprocessable rather than incompatible",
        },
    }


def enrich_candidate(provider: HuggingFaceDatasetProvider, candidate: DatasetCandidate, tokenizer: Tokenizer) -> tuple[DatasetCandidate | None, dict[str, object]]:
    evidence: dict[str, object] = {"candidate_id": candidate.candidate_id, "dataset_id": candidate.dataset_id}
    try:
        snapshot = provider.resolve_revision(candidate.dataset_id, candidate.revision or "main")
        files = provider.enumerate_files(snapshot)
    except Exception as exc:  # provider boundary; preserve rejection evidence
        evidence.update({"status": "rejected", "reason": f"immutable_metadata_unavailable:{type(exc).__name__}:{exc}"})
        return None, evidence

    if snapshot.gated or snapshot.private:
        evidence.update({"status": "rejected", "reason": "gated_or_private_dataset"})
        return None, evidence
    if snapshot.remote_code_required:
        evidence.update({"status": "rejected", "reason": "remote_dataset_code_required"})
        return None, evidence

    material = [item for item in files if item.relative_path.casefold().endswith(DATA_SUFFIXES)]
    if not material:
        evidence.update({"status": "rejected", "reason": "no_supported_data_files"})
        return None, evidence
    known_sizes = [int(item.size_bytes) for item in material if item.size_bytes is not None]
    estimated_bytes = sum(known_sizes) if known_sizes else None
    suffixes = sorted({Path(item.relative_path).suffix.casefold().lstrip(".") for item in material})

    try:
        splits = hf_splits(candidate.dataset_id)
        preferred = next((item for item in splits if str(item.get("split", "")).casefold() == "train"), None)
        split_info = preferred or (splits[0] if splits else None)
        if split_info is None:
            raise RuntimeError("dataset viewer exposed no split")
        config_name = str(split_info.get("config") or "default")
        split_name = str(split_info.get("split") or "train")
        rows = hf_first_rows(candidate.dataset_id, config_name, split_name)
        if not rows:
            raise RuntimeError("dataset viewer exposed no preview rows")
        preview = coverage_audit(rows, tokenizer)
        preview_status = "checked"
        estimated_rows = sum(
            int(item.get("num_examples") or 0)
            for item in splits
            if str(item.get("config") or "") == config_name
        ) or None
    except Exception as exc:
        evidence.update({"status": "rejected", "reason": f"bounded_preview_unavailable:{type(exc).__name__}:{exc}"})
        return None, evidence

    enriched = DatasetCandidate.from_dict(candidate.to_dict())
    enriched.revision = snapshot.resolved_revision
    enriched.license = snapshot.license or enriched.license
    enriched.provenance = {
        **dict(enriched.provenance),
        "kind": "huggingface_immutable_snapshot",
        "resolved_revision": snapshot.resolved_revision,
        "dataset_card_ref": snapshot.dataset_card_ref,
        "dataset_card_revision": snapshot.dataset_card_revision,
        "license_source": snapshot.license_source,
        "provenance_confidence": snapshot.provenance_confidence,
    }
    enriched.estimated_rows = estimated_rows
    enriched.estimated_bytes = estimated_bytes
    enriched.compatibility.update(
        {
            "remote_code_required": False,
            "immutable_revision_available": True,
            "tokenizer_compatible": bool(preview["tokenizer"]["compatible"]),
            "model_compatible": True,
        }
    )
    capability = dict(preview["capability_coverage"])
    supported_capabilities = [key for key, value in capability.items() if float(value) > 0.0]
    enriched.task_types = sorted(set(enriched.task_types + supported_capabilities))
    if float(capability.get("instruction adherence", 0.0)) > 0.0:
        enriched.domains = sorted(set(enriched.domains + ["instruction following", "general language"]))
    elif not enriched.domains:
        enriched.domains = ["general language"]
    enriched.format_profile = {
        "record_format": suffixes[0] if len(suffixes) == 1 else "mixed",
        "source_formats": suffixes,
        "semantic_formats": list(preview["semantic_formats"]),
    }
    enriched.trust_level = "external_metadata"
    enriched.risk_signals = sorted(
        set(
            [signal for signal in enriched.risk_signals if signal != "requires_explicit_acquisition_enablement"]
            + ["external_dataset_requires_acquisition_approval", "bounded_preview_not_full_scan"]
        )
    )
    enriched.evidence_refs = sorted(
        set(enriched.evidence_refs + [snapshot.dataset_card_ref or "", f"https://datasets-server.huggingface.co/splits?dataset={quote(candidate.dataset_id)}"])
        - {""}
    )
    enriched.missing_metadata = sorted(
        set(snapshot.missing_metadata)
        | ({"estimated_bytes"} if estimated_bytes is None else set())
        | ({"estimated_rows"} if estimated_rows is None else set())
    )
    enriched.metadata.update(
        {
            "preview_status": preview_status,
            "preview_audit": preview,
            "capability_targets": supported_capabilities,
            "requires_normalization": True,
            "immutable_file_count": len(material),
            "source_formats": suffixes,
        }
    )
    evidence.update(
        {
            "status": "audited",
            "resolved_revision": snapshot.resolved_revision,
            "license": snapshot.license,
            "dataset_card_ref": snapshot.dataset_card_ref,
            "provenance_confidence": snapshot.provenance_confidence,
            "material_file_count": len(material),
            "estimated_rows": estimated_rows,
            "estimated_bytes": estimated_bytes,
            "source_formats": suffixes,
            "preview": preview,
        }
    )
    if not bool(preview["tokenizer"]["compatible"]):
        evidence["status"] = "rejected"
        evidence["reason"] = "tokenizer_preview_incompatible"
        return None, evidence
    return enriched, evidence


def rank_for_enrichment(candidates: Iterable[DatasetCandidate]) -> list[DatasetCandidate]:
    def score(item: DatasetCandidate) -> tuple[object, ...]:
        task_text = " ".join(item.task_types).casefold()
        instruction_hint = 0 if any(token in task_text or token in item.dataset_id.casefold() for token in ("instruction", "chat", "assistant")) else 1
        return (
            instruction_hint,
            0 if item.license else 1,
            0 if item.revision else 1,
            -int(item.metadata.get("downloads") or 0),
            item.dataset_id.casefold(),
        )
    return sorted(candidates, key=score)[:MAX_ENRICHED]


def main() -> int:
    request, diagnosis, intervention = load_frozen_chain()
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    s3 = s3_client()
    tokenizer_bytes = get_s3_bytes(s3, bucket, TOKENIZER_KEY)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    if TOKENIZER_ID != request.tokenizer_ref:
        raise RuntimeError("frozen DatasetSearchRequest tokenizer identity mismatch")

    provider = HuggingFaceDatasetProvider(enable_network=True, max_results=12, timeout_seconds=15.0)
    discovered, attempts = metadata_discovery(request, provider)
    if not discovered:
        raise RuntimeError("autonomous Hugging Face discovery produced no candidates")

    enriched: list[DatasetCandidate] = []
    audit_records: list[dict[str, object]] = []
    for candidate in rank_for_enrichment(discovered):
        material, record = enrich_candidate(provider, candidate, tokenizer)
        audit_records.append(record)
        if material is not None:
            enriched.append(material)

    if not enriched:
        raise RuntimeError("no candidate survived immutable provenance + preview + tokenizer audit")

    selector = DeterministicDatasetSelectionService()
    decision = selector.select(request, enriched)

    planner = ResolvedEvidenceExperimentPlanner()
    experiment = None
    planner_called = False
    if decision.status == "selected":
        selected_ids = set(decision.selected_candidate_ids)
        selected_candidates = [item for item in enriched if item.candidate_id in selected_ids]
        if not selected_candidates:
            raise RuntimeError("selection decision references no enriched candidate")
        if any(item.metadata.get("preview_status") != "checked" for item in selected_candidates):
            raise RuntimeError("refusing Planner handoff: selected candidate lacks completed preview audit")
        if any(item.compatibility.get("tokenizer_compatible") is not True for item in selected_candidates):
            raise RuntimeError("refusing Planner handoff: selected candidate lacks tokenizer compatibility")
        planner_called = True
        experiment = planner.propose_experiment(diagnosis, intervention, decision, None)
        if experiment.primary_variable != "dataset_mixture":
            raise RuntimeError("Planner violated the one-primary-variable boundary")
        if experiment.dataset_selection_id != decision.decision_id:
            raise RuntimeError("Planner did not bind the exact dataset selection decision")
    else:
        planner_called = False

    result = {
        "result_version": "first-autonomous-dataset-discovery.v1",
        "source": {
            "branch_base": BRANCH_BASE,
            "dataset_search_request": request.to_dict(),
            "diagnosis_report_id": diagnosis.report_id,
            "intervention_id": intervention.intervention_id,
            "primary_variable": intervention.primary_variable,
            "tokenizer_ref": TOKENIZER_ID,
            "tokenizer_s3_key": TOKENIZER_KEY,
            "tokenizer_file_sha256": sha_bytes(tokenizer_bytes),
        },
        "discovery": {
            "provider": "huggingface",
            "network_enabled": True,
            "queries": attempts,
            "unique_metadata_candidates": len(discovered),
            "enrichment_limit": MAX_ENRICHED,
        },
        "candidate_audits": audit_records,
        "enriched_candidates": [item.to_dict() for item in enriched],
        "selection": decision.to_dict(),
        "planner": {
            "called_after_selected_decision": planner_called,
            "experiment_proposal": None if experiment is None else experiment.to_dict(),
            "training_launched": False,
            "dataset_acquired": False,
            "approval_applied": False,
        },
    }
    atomic_json(Path("first_autonomous_dataset_discovery.json"), result)
    atomic_json(Path("first_autonomous_dataset_candidates.json"), result["enriched_candidates"])
    atomic_json(Path("first_autonomous_dataset_selection.json"), decision.to_dict())
    atomic_json(Path("first_autonomous_dataset_experiment.json"), result["planner"])

    print(
        json.dumps(
            {
                "metadata_candidates": len(discovered),
                "fully_audited_candidates": len(enriched),
                "selection_status": decision.status,
                "selected_candidate_ids": decision.selected_candidate_ids,
                "required_approvals": decision.required_approvals,
                "planner_called": planner_called,
                "experiment_id": None if experiment is None else experiment.experiment_id,
                "primary_variable": None if experiment is None else experiment.primary_variable,
                "training_launched": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
