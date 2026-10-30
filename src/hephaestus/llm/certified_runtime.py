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
import re
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol
import platform
from importlib.metadata import version, PackageNotFoundError
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes
from hephaestus.state.single_writer import single_writer
from .inference_receipts import RoleGeneration

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


def validate_adapter_base(spec: CertifiedRoleModel, config: Mapping[str, object]) -> None:
    if config.get("peft_type") != "LORA":
        raise ValueError("unexpected PEFT adapter type")
    base = str(config.get("base_model_name_or_path") or "").replace("\\", "/").rstrip("/")
    parts = PurePosixPath(base).parts
    slug = "models--" + spec.model_id.replace("/", "--")
    snapshot = len(parts) >= 3 and parts[-3:] == (slug, "snapshots", spec.revision)
    if base != spec.model_id and not snapshot:
        raise ValueError("adapter base-model identity does not match certified registry")
    if config.get("revision") not in {None, "", spec.revision}:
        raise ValueError("adapter base revision does not match certified registry")


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
        if spec.role not in {"controller", "diagnosis", "planner", "evaluator", "judge"} or not re.fullmatch(r"[0-9a-f]{64}", spec.adapter.sha256):
            raise ValueError("invalid certified adapter cache identity")
        root = self.cache_root / spec.role / spec.adapter.sha256
        with single_writer(root, "materialize", blocking=True):
            pointer = root / "current.json"
            try:
                generation = json.loads(pointer.read_text())["generation"]
                if not re.fullmatch(r"[0-9a-f]{32}", generation):
                    raise ValueError("invalid adapter cache generation")
                target = root / "generations" / generation
                if self._cache_valid(spec, target / "adapter", target / "selected-adapter.tar.gz", target / "receipt.json"):
                    return target / "adapter"
            except (OSError, ValueError, KeyError, TypeError):
                pass  # A new immutable generation repairs the cache; old paths stay in place.
            return self._materialize_generation(spec, root, pointer)

    def _materialize_generation(self, spec: CertifiedRoleModel, root: Path, pointer: Path) -> Path:
        target = root / "generations" / uuid4().hex
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
            if not isinstance(adapter_cfg, dict):
                raise ValueError("unexpected PEFT adapter configuration")
            validate_adapter_base(spec, adapter_cfg)
            target.parent.mkdir(parents=True, exist_ok=True)
            (work / "receipt.json").write_text(json.dumps({
                "role": spec.role, "model_id": spec.model_id, "revision": spec.revision,
                "archive_sha256": spec.adapter.sha256,
                "archive_bytes": spec.adapter.bytes, "files": checksums,
            }, sort_keys=True), encoding="utf-8")
            os.replace(work, target)
            # Atomic publication changes only a small pointer. No reader's
            # immutable adapter path is deleted, renamed or overwritten.
            with tempfile.NamedTemporaryFile(dir=root, mode="wb", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(canonical_bytes({"generation":target.name}))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, pointer)
            finally:
                temporary.unlink(missing_ok=True)
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
            if not isinstance(files, dict) or not {"adapter_config.json", "adapter_model.safetensors"} <= set(files):
                return False
            # A mutable receipt cannot vouch for its own checksums. Recover the
            # expected file hashes from the independently pinned archive bytes.
            authoritative = {}
            with tarfile.open(archive, "r:gz") as bundle:
                members = bundle.getmembers()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    return False
                total = 0
                for member in members:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "adapter":
                        return False
                    if member.isdir():
                        continue
                    if not member.isfile() or len(path.parts) != 2 or path.name in authoritative:
                        return False
                    total += member.size
                    if total > MAX_EXPANDED_ADAPTER_BYTES:
                        return False
                    stream = bundle.extractfile(member)
                    if stream is None:
                        return False
                    digest = hashlib.sha256()
                    with stream:
                        for chunk in iter(lambda:stream.read(1024*1024), b""):
                            digest.update(chunk)
                    authoritative[path.name] = digest.hexdigest()
            if files != authoritative or {p.name for p in adapter.iterdir()} != set(files):
                return False
            validate_adapter_base(spec, json.loads((adapter / "adapter_config.json").read_text()))
            return all(
                isinstance(name, str) and PurePosixPath(name).name == name
                and not (adapter / name).is_symlink()
                and (adapter / name).is_file()
                and _sha_file(adapter / name) == sha
                for name, sha in files.items()
            )
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError):
            return False


class RoleGenerator(Protocol):
    def generate(self, *, spec: CertifiedRoleModel, profile: Mapping[str, object],
                 adapter_path: Path, prompt: str, max_new_tokens: int,
                 seed: int) -> RoleGeneration: ...


@dataclass(slots=True)
class PinnedTransformersRoleGenerator:
    """One GPU-resident role at a time; actual Transformers + PEFT inference."""

    hf_cache_dir: Path
    device: str = "cuda:0"
    allow_remote_pinned_download: bool = False
    _loaded_identity: tuple[str, str, str, str, str] | None = field(default=None, init=False)
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

    def _evict_retired_role_cache(self, retired_identity: tuple[str, str, str, str, str]) -> None:
        """Discard only one previously completed role's *temporary* HF cache.

        The signed original-role identities and all durable receipts remain
        untouched. No cache cleanup can target a mounted /workspace volume.
        We need this because the 5 pinned weight sets total 53.64 GiB, while
        the authorized r4 worker used a 40 GB container disk.
        """
        if not self.allow_remote_pinned_download:
            return
        model_id = retired_identity[1]
        if (not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", model_id)
                or not self.hf_cache_dir.is_absolute()
                or self.hf_cache_dir.is_symlink()):
            raise PermissionError("retired model cache identity or root unsafe")
        cache_root = self.hf_cache_dir.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if (cache_root == temp_root or not cache_root.is_relative_to(temp_root)
                or cache_root == Path("/")):
            raise PermissionError("original role cache eviction must stay in temporary storage")
        role_cache = cache_root / ("models--" + model_id.replace("/", "--"))
        if role_cache.is_symlink():
            raise PermissionError("original role cache eviction refuses symlinks")
        if role_cache.exists():
            if not role_cache.is_dir():
                raise PermissionError("original role cache must be directory")
            shutil.rmtree(role_cache)
            print("ORIGINAL_CUDA_RETIRED_ROLE_CACHE_CLEARED=" + retired_identity[0], flush=True)

    def _ensure_role_cache_capacity(self) -> None:
        """Do not attempt a weight download into an already exhausted disk."""
        cache_root = self.hf_cache_dir.resolve()
        # 16.61 GiB is the largest exact pinned role snapshot. Allow extra
        # space for tokenizer, temporary HF/Xet blobs, and worker state.
        # This guard fails before any scientific state can be fabricated.
        existing = cache_root
        while not existing.exists():
            existing = existing.parent
        if self.allow_remote_pinned_download:
            free=shutil.disk_usage(existing).free
            # Numeric local capacity only: no model text, account data, paths
            # or secret-bearing exception body is printed.
            print("ORIGINAL_CUDA_ROLE_CACHE_FREE_GIB_BEFORE_DOWNLOAD="+
                  str(free//(1024**3)),flush=True)
            if free < 22 * 1024**3:
                print("ORIGINAL_CUDA_ROLE_CACHE_PRE_DOWNLOAD_REFUSAL=insufficient_disk",flush=True)
                raise RuntimeError("original CUDA temporary role cache has insufficient free disk")

    def _load(self, spec: CertifiedRoleModel, profile: Mapping[str, object], adapter_path: Path) -> None:
        identity = (spec.role, spec.model_id, spec.revision, spec.adapter.sha256, str(adapter_path.resolve()))
        if self._loaded_identity == identity:
            return
        retired_identity = self._loaded_identity
        self.close()
        if retired_identity is not None:
            print("ORIGINAL_CUDA_ROLE_RETIREMENT_ATTEMPT="+retired_identity[0],flush=True)
            self._evict_retired_role_cache(retired_identity)
        self._ensure_role_cache_capacity()
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
            cache_dir=str(self.hf_cache_dir),
            local_files_only=not self.allow_remote_pinned_download,
            # A pinned revision preserves the exact certified base-model
            # identity; skip redundant alternate weight packaging, not weights
            # needed by Transformers/PEFT.
            ignore_patterns=["consolidated.safetensors", "*.gguf", "*.pth"],
        )
        if not (isinstance(spec.revision, str) and len(spec.revision) == 40
                and all(ch in "0123456789abcdef" for ch in spec.revision)):
            raise PermissionError("original CUDA base snapshot revision must be a full immutable commit")

        try:
            if profile["load_kind"] == "mistral3":
                # Original Ministral-3-2512 configs contain a nested
                # model_type='ministral3' unsupported by Transformers 4.57.x.
                # Import the official backend from the explicitly pinned v5
                # runtime; do not alias the old 4.x tokenizer or use a fallback.
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
                 adapter_path: Path, prompt: str, max_new_tokens: int, seed: int) -> RoleGeneration:
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
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._tokenizer, "eos_token_id", None)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            sequence = self._model.generate(
                **encoded, **decoding, max_new_tokens=max_new_tokens,
                stopping_criteria=StoppingCriteriaList([Deadline()]),
                pad_token_id=pad_id,
            )[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("role inference exhausted 90-second hard generation window")
        continuation = sequence[prompt_len:]
        packages = {}
        for package in ("torch", "transformers", "peft", "tokenizers", "huggingface-hub", "mistral-common"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        generated_ids = continuation.tolist()
        eos = getattr(self._model.generation_config, "eos_token_id", None)
        eos_ids = eos if isinstance(eos, list) else [eos]
        finish = ("eos" if generated_ids and generated_ids[-1] in eos_ids else
                  "length" if len(generated_ids) >= max_new_tokens else "completed")
        return RoleGeneration(
            self._tokenizer.decode(continuation, skip_special_tokens=True), finish,
            {"rendered_prompt": self._tokenizer.decode(ids[0].tolist(), skip_special_tokens=False),
             "input_token_ids":ids[0].tolist(), "generated_token_ids":generated_ids,
             "tokenizer_id":spec.model_id, "tokenizer_revision":spec.revision,
             "decoding":{**decoding, "max_new_tokens":max_new_tokens, "pad_token_id":pad_id, "eos_token_id":eos},
             "seed":seed, "elapsed_seconds":time.monotonic()-started,
             "python":platform.python_version(), "packages":packages,
             "cuda_version":torch.version.cuda, "cudnn_version":torch.backends.cudnn.version(),
             "gpu_name":torch.cuda.get_device_name(0), "device":self.device,
             "compute_capability":list(torch.cuda.get_device_capability(0)),
             "peak_allocated_bytes":torch.cuda.max_memory_allocated(0),
             "deterministic_algorithms":torch.are_deterministic_algorithms_enabled()})


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
    inference_receipt_ref: str
    advisory_only: bool = True


@dataclass(slots=True)
class CertifiedRoleReasoner:
    """Verified input -> actual model execution -> bounded proposal. No action IO."""

    stack: CertifiedRoleModelStack
    contract: RoleMasteryContract
    adapters: VerifiedRoleAdapters
    generator: RoleGenerator
    artifact_store: ArtifactStore
    active_evaluator: object | None = None  # Opt-in governed evaluator routing; frozen default unchanged.

    def propose(self, task: RoleTask, registry: EvidenceRegistry, *,
                context: DeterministicRoleContext, seed: int = 271828,
                max_new_tokens: int = 384) -> ModelRoleObservation:
        spec = self.stack.resolve(task.role)
        active_adapter_path = None
        if task.role == "evaluator" and self.active_evaluator is not None:
            # Resolve the selected finalized adapter and recheck every manifest
            # component on each new request. Simulated activations cannot enter
            # normal production routing.
            spec, active_adapter_path = self.active_evaluator.resolve()
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
        adapter = active_adapter_path if active_adapter_path is not None else self.adapters.materialize(spec)
        generated = self.generator.generate(
            spec=spec, profile=self.contract.profile(task.role),
            adapter_path=adapter, prompt=prompt,
            max_new_tokens=max_new_tokens, seed=seed,
        )
        if not isinstance(generated, RoleGeneration):
            raise TypeError("role generator must return text with execution provenance")
        raw = generated.text
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        output_hash = hashlib.sha256(raw.encode()).hexdigest()
        receipt = {
            "schema":"role-inference.v1", "role":task.role, "run_id":task.run_id,
            "lineage_id":task.lineage_id, "stage_name":task.stage_name,
            "prompt":prompt, "raw_output":raw, "prompt_sha256":prompt_hash, "output_sha256":output_hash,
            "model_id":spec.model_id, "base_revision":spec.revision,
            "adapter_sha256":spec.adapter.sha256, "adapter_source":spec.adapter.s3_key,
            "tokenizer_id":spec.model_id, "tokenizer_revision":spec.revision,
            "profile":self.contract.profile(task.role), "max_new_tokens":max_new_tokens,
            "seed":seed, "finish_reason":generated.finish_reason,
            "execution":dict(generated.metadata), "generator":type(self.generator).__name__,
            "python":platform.python_version(), "evidence_refs":refs,
        }
        try:
            answer = self.contract.parse(task.role, raw, set(refs))
        except ValueError as exc:
            # Keep original real generator text and execution metadata in a
            # content-addressed rejected receipt. A caller may perform one
            # independent, policy-bounded FORMAT repair, never override parse.
            receipt.update(status="rejected", error=str(exc))
            rejected = self.artifact_store.put_bytes(
                canonical_bytes(receipt), media_type="application/json")
            exc.rejected_inference_receipt_ref = rejected.artifact_ref
            raise
        receipt.update(status="accepted", answer=answer.to_dict())
        artifact = self.artifact_store.put_bytes(canonical_bytes(receipt), media_type="application/json")
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
            prompt_sha256=prompt_hash, output_sha256=output_hash,
            inference_receipt_ref=artifact.artifact_ref,
            action_gate=gate,
        )
