"""Actual certified-role execution: pinned HF weights + SHA-verified S3 PEFT.

This is an opt-in inference worker, never an autonomous action executor.
Downloads only immutable adapter objects via S3 GET. HF model snapshots must
already exist in the offline cache at their exact 40-character revisions.
No automatic deterministic fallback and no writes to S3 or model registry.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, DeterministicRoleGate, EvidenceRegistry,
    guard_role_output,
)
from hephaestus.providers.models.role_stack import (
    CertifiedRoleModel, CertifiedRoleModelStack,
)
from .role_contract import RoleAnswer, RoleMasteryContract, RoleTask


MAX_ARCHIVE_MEMBERS = 80
MAX_EXPANDED_ADAPTER_BYTES = 900 * 1024 * 1024
MODEL_CONTROL_ROLES = frozenset({"controller", "judge"})


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


class S3GetObject(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class VerifiedRoleAdapters:
    """S3-read-only materialization; archive hash and extracted tree verified."""

    client: S3GetObject
    bucket: str
    cache_root: Path

    def materialize(self, spec: CertifiedRoleModel) -> Path:
        if not self.bucket.strip():
            raise ValueError("S3 volume bucket is missing")
        target = self.cache_root / spec.role / spec.adapter.sha256
        adapter = target / "adapter"
        archive = target / "selected-adapter.tar.gz"
        receipt = target / "receipt.json"
        if self._cache_valid(spec, adapter, archive, receipt):
            return adapter
        self.cache_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f".{spec.role}-", dir=self.cache_root))
        try:
            compressed = work / "selected-adapter.tar.gz"
            item = self.client.get_object(Bucket=self.bucket, Key=spec.adapter.s3_key)
            body = item["Body"]
            digest = hashlib.sha256()
            size = 0
            try:
                with compressed.open("wb") as output:
                    while True:
                        piece = body.read(1024 * 1024)
                        if not piece:
                            break
                        size += len(piece)
                        if size > spec.adapter.bytes:
                            raise ValueError("adapter S3 object exceeds certified byte count")
                        digest.update(piece)
                        output.write(piece)
            finally:
                if hasattr(body, "close"):
                    body.close()
            if size != spec.adapter.bytes or digest.hexdigest() != spec.adapter.sha256:
                raise ValueError(f"certified {spec.role} adapter bytes/SHA-256 mismatch")
            extracted = work / "adapter"
            extracted.mkdir()
            checksums: dict[str, str] = {}
            total = 0
            with tarfile.open(compressed, "r:gz") as bundle:
                members = bundle.getmembers()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise ValueError("adapter archive has excessive members")
                for member in members:
                    path = PurePosixPath(member.name)
                    if (path.is_absolute() or ".." in path.parts
                            or not path.parts or path.parts[0] != "adapter"):
                        raise ValueError("unsafe adapter archive path")
                    if member.issym() or member.islnk() or member.isdev():
                        raise ValueError("adapter archive contains unsafe link or device")
                    if member.isdir():
                        continue
                    if not member.isfile() or len(path.parts) != 2:
                        raise ValueError("adapter archive contains unexpected member")
                    total += member.size
                    if total > MAX_EXPANDED_ADAPTER_BYTES:
                        raise ValueError("adapter exceeds decompressed size limit")
                    dest = extracted / path.name
                    source = bundle.extractfile(member)
                    if source is None or dest.exists():
                        raise ValueError("adapter archive contains duplicate or missing member")
                    hashed = hashlib.sha256()
                    count = 0
                    with source, dest.open("wb") as output:
                        while True:
                            piece = source.read(1024 * 1024)
                            if not piece:
                                break
                            count += len(piece)
                            if count > member.size:
                                raise ValueError("adapter member exceeded declared size")
                            hashed.update(piece)
                            output.write(piece)
                    if count != member.size:
                        raise ValueError("adapter member is truncated")
                    checksums[path.name] = hashed.hexdigest()
            if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(checksums):
                raise ValueError("PEFT adapter configuration or safetensors weights missing")
            adapter_cfg = json.loads((extracted / "adapter_config.json").read_text())
            if not isinstance(adapter_cfg, dict) or adapter_cfg.get("peft_type") != "LORA":
                raise ValueError("unexpected PEFT adapter type")
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            (work / "receipt.json").write_text(json.dumps({
                "role": spec.role, "model_id": spec.model_id, "revision": spec.revision,
                "archive_sha256": spec.adapter.sha256,
                "archive_bytes": spec.adapter.bytes, "files": checksums,
            }, sort_keys=True), encoding="utf-8")
            os.replace(work, target)
            return target / "adapter"
        finally:
            if work.exists():
                shutil.rmtree(work)

    @staticmethod
    def _cache_valid(spec: CertifiedRoleModel, adapter: Path, archive: Path, receipt: Path) -> bool:
        try:
            if (adapter.is_symlink() or archive.is_symlink() or receipt.is_symlink()
                    or adapter.parent.is_symlink()):
                return False
            doc = json.loads(receipt.read_text(encoding="utf-8"))
            if (doc["role"] != spec.role or doc["model_id"] != spec.model_id
                or doc["revision"] != spec.revision
                or doc["archive_sha256"] != spec.adapter.sha256
                or doc["archive_bytes"] != spec.adapter.bytes
                or archive.stat().st_size != spec.adapter.bytes
                or _sha_file(archive) != spec.adapter.sha256):
                return False
            files = doc["files"]
            if not isinstance(files, dict) or not files:
                return False
            return all(
                isinstance(name, str) and PurePosixPath(name).name == name
                and not (adapter / name).is_symlink()
                and (adapter / name).is_file()
                and _sha_file(adapter / name) == sha
                for name, sha in files.items()
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False


class RoleGenerator(Protocol):
    def generate(self, *, spec: CertifiedRoleModel, profile: Mapping[str, object],
                 adapter_path: Path, prompt: str, max_new_tokens: int,
                 seed: int) -> str: ...


@dataclass(slots=True)
class PinnedTransformersRoleGenerator:
    """One GPU-resident role at a time; actual Transformers + PEFT inference."""

    hf_cache_dir: Path
    device: str = "cuda:0"
    _loaded_identity: tuple[str, str, str] | None = field(default=None, init=False)
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_identity = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load(self, spec: CertifiedRoleModel, profile: Mapping[str, object], adapter_path: Path) -> None:
        identity = (spec.role, spec.revision, spec.adapter.sha256)
        if self._loaded_identity == identity:
            return
        self.close()
        try:
            import torch
            from huggingface_hub import snapshot_download
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("role runtime requires torch, transformers, huggingface_hub and peft") from exc
        if self.device != "cuda:0" or not torch.cuda.is_available():
            raise RuntimeError("certified inference requires one configured CUDA:0 GPU; no CPU fallback")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("certified role worker must run with exactly one visible CUDA GPU")
        snapshot = snapshot_download(
            repo_id=spec.model_id, revision=spec.revision,
            cache_dir=str(self.hf_cache_dir), local_files_only=True,
        )
        try:
            if profile["load_kind"] == "mistral3":
                from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
                tokenizer = MistralCommonBackend.from_pretrained(snapshot)
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    snapshot, local_files_only=True, torch_dtype=torch.bfloat16,
                    device_map={"": 0}, trust_remote_code=False,
                )
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    snapshot, local_files_only=True, trust_remote_code=False,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    snapshot, local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.bfloat16, device_map={"": 0},
                    low_cpu_mem_usage=True,
                )
            model = PeftModel.from_pretrained(
                model, str(adapter_path), is_trainable=False, local_files_only=True,
            )
            model.eval()
            if not all(str(p.device).startswith("cuda") for p in model.parameters()):
                raise RuntimeError("role model parameters are not fully resident on CUDA")
            self._tokenizer, self._model, self._loaded_identity = tokenizer, model, identity
        except Exception:
            self.close()
            raise

    def generate(self, *, spec: CertifiedRoleModel, profile: Mapping[str, object],
                 adapter_path: Path, prompt: str, max_new_tokens: int, seed: int) -> str:
        if not (16 <= max_new_tokens <= 1024):
            raise ValueError("max_new_tokens outside bounded role contract")
        self._load(spec, profile, adapter_path)
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        if profile["load_kind"] == "mistral3":
            encoded = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt", return_dict=True,
                **dict(profile.get("chat_template_kwargs") or {}),
            )
        else:
            encoded = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
                **dict(profile.get("chat_template_kwargs") or {}),
            )
        ids = encoded["input_ids"]
        prompt_len = ids.shape[-1]
        if prompt_len > 4096:
            raise ValueError("role prompt exceeds 4096-token limit")
        encoded = {key: value.to(self.device) if hasattr(value, "to") else value
                   for key, value in encoded.items()}
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        deadline = time.monotonic() + 90.0

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        decoding = dict(profile["decoding"])
        with torch.inference_mode():
            sequence = self._model.generate(
                **encoded, **decoding, max_new_tokens=max_new_tokens,
                stopping_criteria=StoppingCriteriaList([Deadline()]),
                pad_token_id=getattr(self._tokenizer, "pad_token_id", None)
                    or getattr(self._tokenizer, "eos_token_id", None),
            )[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("role inference exhausted 90-second hard generation window")
        continuation = sequence[prompt_len:]
        return self._tokenizer.decode(continuation, skip_special_tokens=True).strip()


@dataclass(frozen=True, slots=True)
class ModelRoleObservation:
    role: str
    answer: RoleAnswer
    model_id: str
    revision: str
    adapter_sha256: str
    prompt_sha256: str
    output_sha256: str
    action_gate: DeterministicRoleGate | None
    advisory_only: bool = True


@dataclass(slots=True)
class CertifiedRoleReasoner:
    """Verified input -> actual model execution -> bounded proposal. No action IO."""

    stack: CertifiedRoleModelStack
    contract: RoleMasteryContract
    adapters: VerifiedRoleAdapters
    generator: RoleGenerator

    def propose(self, task: RoleTask, registry: EvidenceRegistry, *,
                context: DeterministicRoleContext, seed: int = 271828,
                max_new_tokens: int = 384) -> ModelRoleObservation:
        spec = self.stack.resolve(task.role)
        self.contract.validate_model(spec)
        if not self.stack.production_certified or not self.stack.runtime_registry_ready:
            raise PermissionError("role model stack is not certified and admitted")
        if (context.run_id != task.run_id or context.lineage_id != task.lineage_id
                or context.stage_name != task.stage_name):
            raise ValueError("role context does not match task identity")
        refs = [e.ref for e in task.evidence]
        validated = registry.validate(refs, run_id=task.run_id, lineage_id=task.lineage_id)
        if not validated.passed or set(refs) != set(validated.valid_refs):
            raise ValueError("role evidence is not independently verified")
        prompt = self.contract.prompt(task)
        adapter = self.adapters.materialize(spec)
        raw = self.generator.generate(
            spec=spec, profile=self.contract.profile(task.role),
            adapter_path=adapter, prompt=prompt,
            max_new_tokens=max_new_tokens, seed=seed,
        )
        answer = self.contract.parse(task.role, raw, set(refs))
        # Only Controller and Judge propose finite executable transitions.
        # Diagnosis, Planner, and Evaluator outputs remain advisory, even if
        # their training vocabulary uses names resembling governed actions.
        gate = None
        if task.role in MODEL_CONTROL_ROLES:
            gate = guard_role_output(
                role=task.role, output=answer.to_dict(),
                context=context, evidence_registry=registry,
            )
        return ModelRoleObservation(
            role=task.role, answer=answer, model_id=spec.model_id,
            revision=spec.revision, adapter_sha256=spec.adapter.sha256,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            action_gate=gate,
        )
