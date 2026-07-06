from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hephaestus.schemas.run_readiness_report import RunReadinessReport
from hephaestus.schemas.stage_contract import StageContract


_STRONG_EVAL_PACK_INTEGRITY = {"content_hash_verified", "reference_only"}


def _present(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    return value not in (None, "", [], {})


def _check(passed: bool, reason: str | None = None, **metadata: object) -> dict[str, object]:
    payload: dict[str, object] = {"passed": passed}
    if reason:
        payload["reason"] = reason
    payload.update(metadata)
    return payload


@dataclass(slots=True)
class RunReadinessPolicy:
    def evaluate(
        self,
        *,
        run_id: str,
        lineage_id: str,
        stage_name: str,
        stage_contract: StageContract,
        backend_name: str,
        dataset_manifest: dict[str, Any],
        data_contract: dict[str, Any],
        eval_pack: dict[str, Any],
    ) -> RunReadinessReport:
        blockers: list[str] = []
        warnings: list[str] = []
        checks: dict[str, dict[str, object]] = {}

        allowed_backends = list(stage_contract.allowed_backends)
        backend_allowed = not allowed_backends or backend_name in allowed_backends
        if not backend_allowed:
            blockers.append(f"unsupported_backend:{backend_name}")
        checks["backend"] = _check(
            backend_allowed,
            backend_name=backend_name,
            allowed_backends=allowed_backends,
        )

        missing_manifest = [
            field
            for field in stage_contract.required_manifest_fields
            if not _present(dataset_manifest, field)
        ]
        manifest_completeness = float(dataset_manifest.get("completeness_score", 0.0) or 0.0)
        manifest_ready = not missing_manifest and manifest_completeness >= stage_contract.min_manifest_completeness
        if missing_manifest:
            blockers.append(f"missing_manifest_fields:{','.join(missing_manifest)}")
        if manifest_completeness < stage_contract.min_manifest_completeness:
            blockers.append("manifest_completeness_below_stage_minimum")
        checks["dataset_manifest"] = _check(
            manifest_ready,
            missing_fields=missing_manifest,
            completeness_score=manifest_completeness,
            min_manifest_completeness=stage_contract.min_manifest_completeness,
            integrity_level=str(dataset_manifest.get("manifest_integrity_level", "unknown")),
        )

        missing_data_contract = [
            field
            for field in stage_contract.required_data_contract_fields
            if not _present(data_contract, field)
        ]
        if missing_data_contract:
            blockers.append(f"missing_data_contract_fields:{','.join(missing_data_contract)}")
        checks["trainable_data_contract"] = _check(
            not missing_data_contract,
            missing_fields=missing_data_contract,
            schema_version=str(data_contract.get("schema_version", "")),
        )

        eval_integrity = str(eval_pack.get("eval_pack_integrity_level") or eval_pack.get("integrity_level") or "insufficient")
        eval_allowed = eval_integrity in stage_contract.accepted_eval_pack_integrity_levels
        eval_strong = eval_integrity in _STRONG_EVAL_PACK_INTEGRITY
        if not eval_allowed:
            blockers.append(f"unsupported_eval_pack_integrity:{eval_integrity}")
        elif not eval_strong:
            warnings.append(f"weak_eval_pack_integrity:{eval_integrity}")
        checks["eval_pack"] = _check(
            eval_allowed,
            integrity_level=eval_integrity,
            strong_integrity=eval_strong,
            eval_pack_id=str(eval_pack.get("eval_pack_id", "")),
            eval_pack_version=str(eval_pack.get("eval_pack_version", "")),
        )

        missing_refs = self._missing_contract_refs(
            required_refs=stage_contract.required_contract_refs,
            stage_contract=stage_contract,
            dataset_manifest=dataset_manifest,
        )
        if missing_refs:
            warnings.append(f"missing_contract_refs:{','.join(missing_refs)}")
        checks["contract_refs"] = _check(
            not missing_refs,
            missing_refs=missing_refs,
        )

        status = "blocked" if blockers else "inconclusive" if warnings else "ready"
        return RunReadinessReport(
            report_id=f"readiness-{run_id}",
            run_id=run_id,
            lineage_id=lineage_id,
            stage_name=stage_name,
            stage_contract_id=stage_contract.contract_id,
            status=status,
            launch_allowed=not blockers,
            blockers=sorted(set(blockers)),
            warnings=sorted(set(warnings)),
            checks=checks,
            metadata={
                "backend_name": backend_name,
                "eval_pack_ref": stage_contract.eval_pack_ref,
            },
        )

    def _missing_contract_refs(
        self,
        *,
        required_refs: list[str],
        stage_contract: StageContract,
        dataset_manifest: dict[str, Any],
    ) -> list[str]:
        missing: list[str] = []
        for ref in required_refs:
            if ref == "stage.eval_pack_ref" and not stage_contract.eval_pack_ref:
                missing.append(ref)
            elif ref == "manifest.stage_data_policy_ref" and not dataset_manifest.get("stage_data_policy_ref"):
                missing.append(ref)
            elif ref == "manifest.tokenizer_ref" and not dataset_manifest.get("tokenizer_ref"):
                missing.append(ref)
        return missing
