from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from hephaestus.schemas.code_edit_proposal import CodeEditProposal

DEFAULT_ALLOWED_PATH_PREFIXES = [
    "src/hephaestus/",
    "tests/",
    "docs/",
    "configs/",
]

DEFAULT_FORBIDDEN_PATH_PREFIXES = [
    ".git/",
    "secrets/",
    "private/",
    "data/",
    "artifacts/",
    "state/",
    "runs/",
    "checkpoints/",
    "model_weights/",
    "eval_packs/frozen/",
    "frozen_eval_packs/",
    "external_data/",
]

DEFAULT_FORBIDDEN_FILE_NAMES = [
    ".env",
    ".env.local",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "token.json",
]

FORBIDDEN_EXTENSIONS = {".pt", ".safetensors", ".bin", ".ckpt", ".pth"}
DEFAULT_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
HIGH_RISK_PATH_MARKERS = ["/policy/", "/control/", "/schemas/", "/state/"]


def _normalize_path(path: str) -> str:
    return PurePosixPath(path.strip()).as_posix().lstrip("./")


def _is_prefixed(path: str, prefixes: list[str]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def _is_forbidden_path(path: str, *, forbidden_prefixes: list[str], forbidden_names: set[str]) -> tuple[bool, str]:
    lowered = path.lower()
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(path).suffix.lower()

    if _is_prefixed(path, forbidden_prefixes):
        return True, "forbidden_prefix"
    if name in forbidden_names:
        return True, "forbidden_file_name"
    if suffix in FORBIDDEN_EXTENSIONS:
        return True, "forbidden_model_weight_extension"
    if "eval_packs/frozen" in lowered or "frozen_eval_packs" in lowered:
        return True, "frozen_eval_pack_protected"
    return False, ""


def _classify_risk(allowed_files: list[str]) -> str:
    if not allowed_files:
        return "medium"
    if all(path.startswith("docs/") or path.startswith("tests/") for path in allowed_files):
        return "low"
    if any(any(marker in f"/{path}" for marker in HIGH_RISK_PATH_MARKERS) for path in allowed_files):
        return "high"
    if all(path.startswith("src/hephaestus/") for path in allowed_files):
        return "medium"
    return "medium"


def evaluate_code_edit_proposal(
    proposal: CodeEditProposal | dict[str, object],
    policy: dict[str, object] | None = None,
) -> CodeEditProposal:
    normalized = proposal if isinstance(proposal, CodeEditProposal) else CodeEditProposal.from_dict(dict(proposal))
    policy_config = dict(policy or {})

    allowed_prefixes = [str(v) for v in policy_config.get("allowed_path_prefixes", DEFAULT_ALLOWED_PATH_PREFIXES)]
    forbidden_prefixes = [str(v) for v in policy_config.get("forbidden_path_prefixes", DEFAULT_FORBIDDEN_PATH_PREFIXES)]
    forbidden_file_names = {str(v).lower() for v in policy_config.get("forbidden_file_names", DEFAULT_FORBIDDEN_FILE_NAMES)}
    max_file_size_bytes = int(policy_config.get("max_file_size_bytes", DEFAULT_MAX_FILE_SIZE_BYTES))
    file_sizes = {
        _normalize_path(str(k)): int(v)
        for k, v in dict(policy_config.get("file_sizes", {})).items()
        if isinstance(v, int) or (isinstance(v, str) and str(v).isdigit())
    }

    allowed_files: list[str] = []
    forbidden_files: list[str] = []
    reasons_by_path: dict[str, str] = {}

    for target in normalized.target_files:
        path = _normalize_path(target)
        if not path:
            continue

        forbidden, reason = _is_forbidden_path(
            path,
            forbidden_prefixes=forbidden_prefixes,
            forbidden_names=forbidden_file_names,
        )
        if not forbidden and path in file_sizes and file_sizes[path] > max_file_size_bytes:
            forbidden = True
            reason = "file_size_exceeds_policy_limit"

        if not forbidden and not _is_prefixed(path, allowed_prefixes):
            forbidden = True
            reason = "outside_allowed_path_prefixes"

        if forbidden:
            forbidden_files.append(path)
            reasons_by_path[path] = reason
        else:
            allowed_files.append(path)

    required_approvals = set(normalized.required_approvals)
    required_approvals.add("operator_approval")

    metadata = dict(normalized.metadata)
    classification: dict[str, object] = {
        "allowed_path_prefixes": allowed_prefixes,
        "forbidden_path_prefixes": forbidden_prefixes,
        "forbidden_file_names": sorted(forbidden_file_names),
        "path_reasons": reasons_by_path,
    }
    metadata["classification"] = classification

    if not normalized.target_files:
        status = "blocked"
        risk_level = "forbidden"
        required_approvals.add("not_approvable_missing_target_files")
        classification["path_reasons"] = {"<none>": "missing_target_files"}
    elif forbidden_files:
        required_approvals.add("not_approvable_forbidden_path")
        status = "blocked"
        risk_level = "forbidden"
    else:
        status = normalized.status if normalized.status in {"rejected", "blocked", "approved"} else "approval_required"
        risk_level = _classify_risk(allowed_files)
        if risk_level == "high":
            required_approvals.add("high_risk_approval")

    evaluated = CodeEditProposal.from_dict(
        {
            **normalized.to_dict(),
            "status": status,
            "risk_level": risk_level,
            "target_files": sorted(set([_normalize_path(p) for p in normalized.target_files if _normalize_path(p)])),
            "allowed_files_touched": sorted(set(allowed_files)),
            "forbidden_files_touched": sorted(set(forbidden_files)),
            "required_approvals": sorted(required_approvals),
            "metadata": metadata,
        }
    )
    return evaluated
