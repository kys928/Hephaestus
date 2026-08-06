"""Frozen-pack generation and evaluator-ready evidence materialization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.experiment_contract import TrainingRunHandle
from hephaestus.storage.base import ArtifactStore

from .backends import GenerationBackend, GenerationBackendError, load_generation_instructions
from .models import (
    GeneratedText,
    GenerationPlan,
    GenerationReport,
    GenerationResult,
    GenerationSample,
    GenerationTask,
)

_TASK_FIELDS = (
    "generation_probes",
    "continuation_prompts",
    "ranking_sets",
    "regression_prompts",
    "structure_tests",
    "repetition_checks",
    "length_termination_checks",
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _issue(
    code: str,
    category: str,
    message: str,
    *,
    retryable: bool = False,
    blocking: bool = True,
    evidence_refs: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> ContractIssue:
    return ContractIssue(
        code=code,
        category=category,
        message=message,
        retryable=retryable,
        blocking=blocking,
        evidence_refs=evidence_refs or [],
        metadata=metadata or {},
    )


@dataclass(slots=True)
class EvaluationGenerationService:
    """Generate all frozen task/seed samples and attach evaluator metadata."""

    artifact_root: Path
    backend: GenerationBackend
    pack_name: str = "semantic_behavior_v1"
    config_dir: Path = Path("configs")
    artifact_store: ArtifactStore | None = None

    def plan(self) -> GenerationPlan:
        pack = load_eval_pack(self.pack_name, config_dir=self.config_dir)
        if not bool(pack.get("frozen", False)):
            raise ValueError("generation requires a frozen evaluation pack")
        if not bool(pack.get("content_hash_verified", False)):
            raise ValueError("generation requires verified frozen-pack content hash")
        content_hash = str(pack.get("content_hash") or "")
        if not content_hash:
            raise ValueError("generation requires a frozen-pack content hash")
        normalized = dict(pack["eval_pack"])
        seeds = [int(seed) for seed in dict(normalized.get("decoding_config", {})).get("seeds", [])]
        if not seeds:
            raise ValueError("generation pack has no explicit seeds")
        tasks: list[GenerationTask] = []
        seen: set[str] = set()
        for field_name in _TASK_FIELDS:
            raw_tasks = normalized.get(field_name, [])
            if not isinstance(raw_tasks, list):
                raise ValueError(f"evaluation task field {field_name} must be a list")
            for raw in raw_tasks:
                if not isinstance(raw, dict):
                    raise ValueError(f"evaluation task in {field_name} must be an object")
                task_id = str(raw.get("task_id") or "").strip()
                prompt = str(raw.get("prompt") or "")
                if not task_id or not prompt:
                    raise ValueError(f"evaluation task in {field_name} lacks task_id or prompt")
                if task_id in seen:
                    raise ValueError(f"duplicate evaluation task_id: {task_id}")
                seen.add(task_id)
                for seed in seeds:
                    tasks.append(GenerationTask(task_id, field_name, prompt, seed))
        if not tasks:
            raise ValueError("evaluation pack has no generation tasks")
        decoding = dict(normalized.get("decoding_config", {}))
        settings_payload = {
            "protocol": "evaluation-generation.v1",
            "eval_pack_id": pack["eval_pack_id"],
            "eval_pack_version": pack["eval_pack_version"],
            "content_hash": content_hash,
            "decoding_config": decoding,
        }
        return GenerationPlan(
            eval_pack_id=str(pack["eval_pack_id"]),
            eval_pack_version=str(pack["eval_pack_version"]),
            content_hash=content_hash,
            decoding_config=decoding,
            generation_settings_id=f"generation-settings-{_stable_hash(settings_payload)[:24]}",
            seed_identity=f"seed-set-{_stable_hash(seeds)[:24]}",
            tasks=tuple(tasks),
        )

    def generate(
        self,
        run: TrainingRunHandle,
        *,
        generation_handoff_ref: str | None = None,
        checkpoint_ref: str | None = None,
    ) -> GenerationResult:
        plan = self.plan()
        checkpoint = str(checkpoint_ref or (run.checkpoint_refs[-1] if run.checkpoint_refs else "")).strip()
        issues: list[ContractIssue] = []
        if run.status != "completed":
            issues.append(
                _issue(
                    "generation_run_not_completed",
                    "policy_blocked",
                    f"Run '{run.run_id}' must be completed before semantic generation.",
                )
            )
        if not checkpoint:
            issues.append(
                _issue(
                    "generation_checkpoint_missing",
                    "missing_evidence",
                    f"Run '{run.run_id}' has no concrete checkpoint reference.",
                )
            )
        handoff_ref = str(
            generation_handoff_ref
            or run.metadata.get("generation_handoff_ref")
            or run.metadata.get("loading_instructions_ref")
            or ""
        ).strip()
        loading: dict[str, object] = {}
        if not handoff_ref:
            issues.append(
                _issue(
                    "generation_handoff_missing",
                    "missing_evidence",
                    f"Run '{run.run_id}' has no generation handoff reference.",
                )
            )
        else:
            try:
                loading = load_generation_instructions(handoff_ref)
            except GenerationBackendError as exc:
                issues.append(
                    _issue(
                        exc.code,
                        "artifact_integrity",
                        str(exc),
                        retryable=exc.retryable,
                        evidence_refs=[handoff_ref],
                    )
                )
        report_root = (
            self.artifact_root
            / run.run_id
            / "semantic_generation"
            / plan.generation_settings_id
        )
        report = GenerationReport(
            report_id=f"generation-report-{_stable_hash([run.run_id, checkpoint, plan.generation_settings_id])[:24]}",
            run_id=run.run_id,
            checkpoint_ref=checkpoint,
            backend_id=self.backend.backend_id,
            eval_pack_id=plan.eval_pack_id,
            eval_pack_version=plan.eval_pack_version,
            content_hash=plan.content_hash,
            decoding_config=dict(plan.decoding_config),
            generation_settings_id=plan.generation_settings_id,
            seed_identity=plan.seed_identity,
            completion_status="failed" if issues else "running",
            issues=issues,
            metadata={
                "generation_handoff_ref": handoff_ref or None,
                "task_count": len(plan.tasks),
                "does_not_score": True,
                "does_not_promote": True,
            },
        )
        if issues:
            return GenerationResult(report=report, run_handle=TrainingRunHandle.from_dict(run.to_dict()))

        pending: list[GenerationTask] = []
        samples_by_identity: dict[tuple[str, int], GenerationSample] = {}
        for task in plan.tasks:
            cached = self._load_cached_sample(report_root, run.run_id, checkpoint, plan, task)
            if cached is None:
                pending.append(task)
            else:
                samples_by_identity[(task.task_id, task.seed)] = cached

        if pending:
            try:
                outputs = self.backend.generate_batch(
                    pending,
                    run_id=run.run_id,
                    loading_instructions=loading,
                    decoding_config=plan.decoding_config,
                )
                if len(outputs) != len(pending):
                    raise GenerationBackendError(
                        "generation_result_count_mismatch",
                        "Generation backend returned the wrong number of samples.",
                    )
                for task, generated in zip(pending, outputs, strict=True):
                    sample = self._persist_sample(
                        report_root, run.run_id, checkpoint, plan, task, generated
                    )
                    samples_by_identity[(task.task_id, task.seed)] = sample
            except GenerationBackendError as exc:
                report.issues.append(
                    _issue(
                        exc.code,
                        "provider_unavailable",
                        str(exc),
                        retryable=exc.retryable,
                        evidence_refs=[handoff_ref, checkpoint],
                    )
                )
            except Exception as exc:
                report.issues.append(
                    _issue(
                        "generation_backend_failed",
                        "runtime_failure",
                        f"Generation backend failed: {type(exc).__name__}.",
                        retryable=True,
                        evidence_refs=[handoff_ref, checkpoint],
                    )
                )

        report.samples = [
            samples_by_identity[key]
            for key in sorted(samples_by_identity)
        ]
        expected = {(task.task_id, task.seed) for task in plan.tasks}
        missing = sorted(expected - set(samples_by_identity))
        if missing:
            report.issues.append(
                _issue(
                    "generation_samples_missing",
                    "missing_evidence",
                    f"Generation is missing {len(missing)} required task/seed samples.",
                    retryable=True,
                    metadata={"missing": [f"{task_id}:{seed}" for task_id, seed in missing]},
                )
            )
        report.evidence_refs = sorted({sample.evidence_ref for sample in report.samples})
        report.completion_status = "completed" if not any(issue.blocking for issue in report.issues) else "partial"
        report_path = report_root / "generation_report.json"
        report.report_ref = str(report_path)
        _atomic_json(report_path, report.to_dict())
        if self.artifact_store is not None:
            artifact = self.artifact_store.put_file(
                report_path,
                expected_hash=f"sha256:{_file_hash(report_path)}",
                media_type="application/json",
            )
            if not self.artifact_store.verify(artifact.artifact_ref):
                report.issues.append(
                    _issue(
                        "generation_report_store_verification_failed",
                        "artifact_integrity",
                        "Artifact store could not verify the generation report.",
                        evidence_refs=[artifact.artifact_ref],
                    )
                )
                report.completion_status = "partial"
            else:
                report.report_ref = artifact.artifact_ref
                report.evidence_refs = sorted({*report.evidence_refs, artifact.artifact_ref})
                _atomic_json(report_path, report.to_dict())

        updated = TrainingRunHandle.from_dict(run.to_dict())
        updated.metadata = dict(updated.metadata)
        updated.metadata["semantic_evaluation"] = report.semantic_evaluation_metadata()
        updated.metadata["generation_report"] = {
            "report_id": report.report_id,
            "report_ref": report.report_ref,
            "completion_status": report.completion_status,
            "generation_settings_id": report.generation_settings_id,
            "seed_identity": report.seed_identity,
        }
        return GenerationResult(report=report, run_handle=updated)

    def _load_cached_sample(
        self,
        root: Path,
        run_id: str,
        checkpoint_ref: str,
        plan: GenerationPlan,
        task: GenerationTask,
    ) -> GenerationSample | None:
        path = self._sample_path(root, task)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._quarantine(path)
            return None
        expected_identity = self._sample_identity(run_id, checkpoint_ref, plan, task)
        output = str(payload.get("output") or "")
        valid = (
            payload.get("sample_id") == expected_identity
            and payload.get("output_hash") == f"sha256:{hashlib.sha256(output.encode('utf-8')).hexdigest()}"
            and str(payload.get("checkpoint_ref") or "") == checkpoint_ref
            and str(payload.get("generation_settings_id") or "") == plan.generation_settings_id
        )
        if not valid:
            self._quarantine(path)
            return None
        try:
            return GenerationSample(
                sample_id=str(payload["sample_id"]),
                run_id=str(payload["run_id"]),
                task_id=str(payload["task_id"]),
                task_kind=str(payload["task_kind"]),
                seed=int(payload["seed"]),
                prompt_hash=str(payload["prompt_hash"]),
                output=output,
                output_hash=str(payload["output_hash"]),
                evidence_ref=str(payload.get("evidence_ref") or path),
                checkpoint_ref=str(payload["checkpoint_ref"]),
                finish_reason=str(payload.get("finish_reason") or "completed"),
                prompt_tokens=(int(payload["prompt_tokens"]) if payload.get("prompt_tokens") is not None else None),
                generated_tokens=(int(payload["generated_tokens"]) if payload.get("generated_tokens") is not None else None),
                metadata=dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError):
            self._quarantine(path)
            return None

    def _persist_sample(
        self,
        root: Path,
        run_id: str,
        checkpoint_ref: str,
        plan: GenerationPlan,
        task: GenerationTask,
        generated: GeneratedText,
    ) -> GenerationSample:
        path = self._sample_path(root, task)
        output_hash = f"sha256:{hashlib.sha256(generated.output.encode('utf-8')).hexdigest()}"
        sample = GenerationSample(
            sample_id=self._sample_identity(run_id, checkpoint_ref, plan, task),
            run_id=run_id,
            task_id=task.task_id,
            task_kind=task.task_kind,
            seed=task.seed,
            prompt_hash=f"sha256:{hashlib.sha256(task.prompt.encode('utf-8')).hexdigest()}",
            output=generated.output,
            output_hash=output_hash,
            evidence_ref=str(path),
            checkpoint_ref=checkpoint_ref,
            finish_reason=generated.finish_reason,
            prompt_tokens=generated.prompt_tokens,
            generated_tokens=generated.generated_tokens,
            metadata={
                **generated.metadata,
                "generation_settings_id": plan.generation_settings_id,
                "eval_pack_id": plan.eval_pack_id,
                "eval_pack_version": plan.eval_pack_version,
            },
        )
        payload = sample.to_dict()
        payload["generation_settings_id"] = plan.generation_settings_id
        _atomic_json(path, payload)
        if self.artifact_store is not None:
            artifact = self.artifact_store.put_file(
                path,
                expected_hash=f"sha256:{_file_hash(path)}",
                media_type="application/json",
            )
            if not self.artifact_store.verify(artifact.artifact_ref):
                raise GenerationBackendError(
                    "generation_sample_store_verification_failed",
                    f"Artifact store could not verify sample {task.task_id}/{task.seed}.",
                )
            sample = GenerationSample(
                **{**sample.to_dict(), "evidence_ref": artifact.artifact_ref}
            )
        return sample

    @staticmethod
    def _sample_identity(
        run_id: str,
        checkpoint_ref: str,
        plan: GenerationPlan,
        task: GenerationTask,
    ) -> str:
        return f"sample-{_stable_hash([run_id, checkpoint_ref, plan.generation_settings_id, task.task_id, task.seed])[:24]}"

    @staticmethod
    def _sample_path(root: Path, task: GenerationTask) -> Path:
        safe = _stable_hash([task.task_id, task.seed])[:24]
        return root / "samples" / f"{safe}.json"

    @staticmethod
    def _quarantine(path: Path) -> None:
        target = path.with_suffix(path.suffix + ".corrupt")
        try:
            os.replace(path, target)
        except OSError:
            path.unlink(missing_ok=True)


def generation_evidence_refs(reports: Iterable[GenerationReport]) -> list[str]:
    return sorted(
        {
            ref
            for report in reports
            for ref in [report.report_ref, *report.evidence_refs]
            if ref
        }
    )
