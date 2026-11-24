"""Read-only view of certified V1 output for Anvil investigation contexts."""

from dataclasses import dataclass
from hashlib import sha256

from hephaestus.schemas.anvil import require_digest

from .role_contract import RoleAnswer, RoleMasteryContract


@dataclass(frozen=True, slots=True)
class LegacyAnswerView:
    role: str
    receipt_ref: str
    excerpt_ref: str
    answer: RoleAnswer


def project_legacy_answer(
    contract: RoleMasteryContract, *, role: str, raw: str, receipt_ref: str, allowed_refs: set[str],
) -> LegacyAnswerView:
    """Preserve V1 semantics. The caller remains responsible for receipt custody."""
    require_digest(receipt_ref)
    answer = contract.parse(role, raw, allowed_refs)
    return LegacyAnswerView(role, receipt_ref, "sha256:" + sha256(raw.encode()).hexdigest(), answer)
