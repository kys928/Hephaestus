"""Anvil V2 advisory contributions and runtime-owned provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class ContributionKind(StrEnum):
    OBJECTIVE = "objective"
    OBSERVATION = "observation"
    MODEL_CLAIM = "model_claim"
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    CHALLENGE = "challenge"
    PROPOSAL = "proposal"
    PREDICTION = "prediction"
    CAPABILITY_REQUEST = "capability_request"
    EXPERIMENT_RESULT = "experiment_result"
    REVISION = "revision"
    UNCERTAINTY = "uncertainty"
    CONCLUSION = "conclusion"
    VERDICT = "verdict"


class RelationKind(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CHALLENGES = "challenges"
    TESTS = "tests"
    ANSWERS = "answers"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    REVISES = "revises"
    SUPERSEDES = "supersedes"


def require_digest(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("expected canonical SHA-256 reference")


@dataclass(frozen=True, slots=True)
class V1Position:
    decision: str
    action: str
    primary_variable: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() or len(value) > 128
               for value in (self.decision, self.action, self.primary_variable)):
            raise ValueError("invalid frozen v1 position")


@dataclass(frozen=True, slots=True)
class Contribution:
    kind: ContributionKind
    text: str
    links: tuple[tuple[RelationKind, str], ...] = ()
    evidence_refs: tuple[str, ...] = ()
    capability: str | None = None
    confidence: float | None = None
    uncertainties: tuple[str, ...] = ()
    v1_position: V1Position | None = None

    @classmethod
    def from_model(cls, payload: dict) -> Contribution:
        if not isinstance(payload, dict) or set(payload) - {
            "kind", "text", "links", "evidence_refs", "capability", "confidence", "uncertainties", "v1_position",
        }:
            raise ValueError("contribution contains unsupported or runtime-owned fields")
        kind = ContributionKind(payload.get("kind"))
        if kind in {ContributionKind.OBJECTIVE, ContributionKind.EVIDENCE,
                    ContributionKind.EXPERIMENT_RESULT}:
            raise ValueError("this contribution kind requires trusted runtime admission")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("contribution text must be nonempty and bounded")
        raw_links = payload.get("links", [])
        if not isinstance(raw_links, list) or len(raw_links) > 24:
            raise ValueError("expected a bounded link list")
        links = []
        for row in raw_links:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("expected a relation and contribution reference")
            relation, target = row
            require_digest(target)
            links.append((RelationKind(relation), target))
        evidence = payload.get("evidence_refs", [])
        if not isinstance(evidence, list) or len(evidence) > 24:
            raise ValueError("expected a bounded evidence list")
        for ref in evidence:
            require_digest(ref)
        if len(set(links)) != len(links) or len(set(evidence)) != len(evidence):
            raise ValueError("duplicate links or evidence references")
        if kind == ContributionKind.REVISION and not any(link[0] == RelationKind.REVISES for link in links):
            raise ValueError("a revision must reference the original contribution")
        if kind == ContributionKind.CHALLENGE and not any(link[0] == RelationKind.CHALLENGES for link in links):
            raise ValueError("a challenge must reference the challenged contribution")
        for relation, _ in links:
            if relation == RelationKind.SUPERSEDES:
                raise ValueError("supersession requires trusted runtime admission")
            if relation == RelationKind.REVISES and kind != ContributionKind.REVISION:
                raise ValueError("only a revision can declare a revision link")
        capability = payload.get("capability")
        if kind == ContributionKind.CAPABILITY_REQUEST:
            if not isinstance(capability, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", capability):
                raise ValueError("expected a registered capability identifier")
        elif capability is not None:
            raise ValueError("only a capability request can name a capability")
        confidence = payload.get("confidence")
        if confidence is not None and (type(confidence) not in (int, float) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be a number between zero and one")
        uncertainties = payload.get("uncertainties", [])
        if not isinstance(uncertainties, list) or len(uncertainties) > 30 or any(
            not isinstance(item, str) or len(item) > 300 for item in uncertainties
        ):
            raise ValueError("expected bounded uncertainty statements")
        position = payload.get("v1_position")
        if kind == ContributionKind.MODEL_CLAIM:
            if not isinstance(position, dict) or set(position) != {"decision", "action", "primary_variable"}:
                raise ValueError("model claim requires a complete frozen v1 position")
            position = V1Position(**position)
        elif position is not None:
            raise ValueError("only model claims can carry a frozen v1 position")
        return cls(kind, text, tuple(links), tuple(evidence), capability, confidence, tuple(uncertainties), position)

    def to_dict(self) -> dict:
        document = {"kind": self.kind.value, "text": self.text,
                    "links": [[kind.value, target] for kind, target in self.links],
                    "evidence_refs": list(self.evidence_refs), "capability": self.capability}
        if self.confidence is not None:
            document["confidence"] = self.confidence
        if self.uncertainties:
            document["uncertainties"] = list(self.uncertainties)
        if self.v1_position is not None:
            document["v1_position"] = {
                "decision": self.v1_position.decision,
                "action": self.v1_position.action,
                "primary_variable": self.v1_position.primary_variable,
            }
        return document


@dataclass(frozen=True, slots=True)
class ContributionOrigin:
    """Bound by the caller from trusted context and independent receipts."""

    investigation: str
    run_id: str
    lineage_id: str
    role: str
    inference_ref: str
    context_frontier: str
    idempotency_key: str
    model_id: str
    model_revision: str
    adapter_ref: str

    def __post_init__(self) -> None:
        for value in (self.investigation, self.run_id, self.lineage_id, self.role, self.idempotency_key):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
                raise ValueError("invalid runtime contribution identity")
        if not isinstance(self.model_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,96}/[A-Za-z0-9_.-]{1,96}", self.model_id,
        ):
            raise ValueError("invalid specialist model identity")
        if not isinstance(self.model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("specialist revision must be pinned")
        require_digest(self.adapter_ref)
        require_digest(self.inference_ref)
        require_digest(self.context_frontier)


@dataclass(frozen=True, slots=True)
class AcceptedContribution:
    contribution_id: str
    frontier: str
    previous_frontier: str
    sequence: int
    origin: ContributionOrigin
    payload: Contribution


@dataclass(frozen=True, slots=True)
class InvestigationSnapshot:
    investigation: str
    frontier: str
    contributions: tuple[AcceptedContribution, ...]

    def query(self, *, kind: ContributionKind | None = None,
              linked_to: str | None = None) -> tuple[AcceptedContribution, ...]:
        return tuple(
            row for row in self.contributions
            if (kind is None or row.payload.kind == kind)
            and (linked_to is None or any(target == linked_to for _, target in row.payload.links))
        )
