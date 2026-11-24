"""Host-local Anvil contract rehearsal over the existing repository and writer fence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Callable

from hephaestus.schemas.anvil import (
    AcceptedContribution, Contribution, ContributionKind, ContributionOrigin,
    InvestigationSnapshot, RelationKind, require_digest,
)
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository, StateCommitUncertain
from hephaestus.storage.verified import canonical_bytes, read_verified_json

_CURRENT_CONTEXT = frozenset({
    ContributionKind.CAPABILITY_REQUEST, ContributionKind.PROPOSAL, ContributionKind.VERDICT,
})


def _digest(payload: object) -> str:
    return "sha256:" + sha256(canonical_bytes(payload)).hexdigest()


def _collection(investigation: str) -> str:
    return "anvil.v2." + sha256(investigation.encode()).hexdigest()[:40]


@dataclass(slots=True)
class AnvilState:
    repository: SQLiteStateRepository
    verify_origin: Callable[[ContributionOrigin], bool]
    verify_evidence: Callable[[ContributionOrigin, str], bool]

    @staticmethod
    def _validate_references(
        parents: dict[str, tuple[ContributionOrigin, int]], origin: ContributionOrigin,
        contribution: Contribution, observed_sequence: int,
    ) -> None:
        for relation, target in contribution.links:
            parent = parents.get(target)
            if parent is None or (parent[0].run_id, parent[0].lineage_id) != (origin.run_id, origin.lineage_id) or parent[1] > observed_sequence:
                raise ValueError("contribution link has an absent or out-of-scope target")
            if relation == RelationKind.REVISES and parent[0].role != origin.role:
                raise ValueError("revision may only amend the contributor's own position")

    def append_contribution(self, origin: ContributionOrigin, payload: Contribution) -> AcceptedContribution:
        payload = Contribution.from_model(payload.to_dict())
        if self.verify_origin(origin) is not True:
            raise PermissionError("inference or context receipt failed admission")
        body = {"schema": "anvil.contribution.v2", "origin": asdict(origin), "payload": payload.to_dict()}
        identity = _digest([origin.investigation, origin.idempotency_key])
        with execution_owner(self.repository, "anvil-contribution", origin.investigation):
            snapshot = self.replay(origin.investigation)
            prior = next((row for row in snapshot.contributions if row.contribution_id == identity), None)
            if prior is not None:
                if prior.origin != origin or prior.payload != payload:
                    raise ValueError("conflicting contribution idempotency key")
                return prior
            frontiers = {_digest(["anvil.snapshot.v2", origin.investigation]): 0}
            frontiers.update((row.frontier, row.sequence) for row in snapshot.contributions)
            if origin.context_frontier not in frontiers:
                raise ValueError("unknown investigation frontier")
            if payload.kind in _CURRENT_CONTEXT and origin.context_frontier != snapshot.frontier:
                raise ValueError("proposal requires the current investigation frontier")
            if any(self.verify_evidence(origin, ref) is not True for ref in payload.evidence_refs):
                raise PermissionError("evidence reference failed scoped verification")
            self._validate_references(
                {row.contribution_id: (row.origin, row.sequence) for row in snapshot.contributions},
                origin, payload, frontiers[origin.context_frontier],
            )
            record = {**body, "contribution_id": identity, "previous_frontier": snapshot.frontier,
                      "sequence": len(snapshot.contributions) + 1}
            frontier = _digest(record)
            self.repository.append(_collection(origin.investigation), {**record, "frontier": frontier},
                                   operation_id=identity)
        return AcceptedContribution(identity, frontier, snapshot.frontier, record["sequence"], origin, payload)

    @classmethod
    def _replay_records(cls, investigation: str, records: list[dict[str, object]],
                        expected_frontier: str | None = None) -> InvestigationSnapshot:
        frontier = _digest(["anvil.snapshot.v2", investigation])
        frontiers = {frontier: 0}
        accepted = []
        scopes: dict[str, tuple[ContributionOrigin, int]] = {}
        for sequence, stored in enumerate(records, start=1):
            record = {key: value for key, value in stored.items() if key != "frontier"}
            origin = ContributionOrigin(**record["origin"])
            payload = Contribution.from_model(record["payload"])
            identity = _digest([investigation, origin.idempotency_key])
            if (record.get("schema") != "anvil.contribution.v2"
                    or origin.investigation != investigation or record.get("contribution_id") != identity
                    or identity in scopes or record.get("sequence") != sequence
                    or record.get("previous_frontier") != frontier
                    or stored.get("frontier") != _digest(record)
                    or origin.context_frontier not in frontiers
                    or (payload.kind in _CURRENT_CONTEXT and origin.context_frontier != frontier)):
                raise ValueError("invalid or altered Anvil history")
            cls._validate_references(scopes, origin, payload, frontiers[origin.context_frontier])
            accepted.append(AcceptedContribution(identity, stored["frontier"], frontier, sequence, origin, payload))
            scopes[identity] = (origin, sequence)
            frontier = stored["frontier"]
            frontiers[frontier] = sequence
        if expected_frontier is not None and frontier != expected_frontier:
            raise ValueError("Anvil history differs from the independently retained frontier")
        return InvestigationSnapshot(investigation, frontier, tuple(accepted))

    def replay(self, investigation: str, *, expected_frontier: str | None = None) -> InvestigationSnapshot:
        return self._replay_records(investigation, self.repository.all(_collection(investigation)), expected_frontier)

    def checkpoint(self, investigation: str, artifacts: ArtifactStore) -> str:
        """Seal one consistent history. Retain its frontier independently."""
        records = self.repository.all(_collection(investigation))
        snapshot = self._replay_records(investigation, records)
        raw = canonical_bytes({
            "schema": "anvil.checkpoint.v2",
            "investigation": investigation,
            "frontier": snapshot.frontier,
            "records": records,
        })
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("checkpoint exceeds the supported rehearsal size")
        return artifacts.put_bytes(raw, media_type="application/json").artifact_ref

    def restore(self, investigation: str, artifacts: ArtifactStore, checkpoint_ref: str,
                *, expected_frontier: str) -> InvestigationSnapshot:
        """Append a verified suffix on an admitted host-local repository."""
        require_digest(expected_frontier)
        document = read_verified_json(artifacts, checkpoint_ref)
        if (document.get("schema") != "anvil.checkpoint.v2"
                or document.get("investigation") != investigation
                or not isinstance(document.get("records"), list)
                or document.get("frontier") != expected_frontier):
            raise ValueError("checkpoint does not match the retained scope and frontier")
        records = document["records"]
        result = self._replay_records(investigation, records, expected_frontier)
        collection = _collection(investigation)
        with execution_owner(self.repository, "anvil-contribution", investigation):
            existing = self.repository.all(collection)
            self._replay_records(investigation, existing)
            if existing != records[:len(existing)]:
                raise ValueError("checkpoint would roll back or replace accepted contributions")
            if len(existing) < len(records):
                with self.repository.transaction() as transaction:
                    for record in records[len(existing):]:
                        transaction.append(collection, record, operation_id=record["contribution_id"])
        return result

    def read_snapshot(self, investigation: str, *, sequence: int | None = None) -> InvestigationSnapshot:
        current = self.replay(investigation)
        if sequence is None:
            return current
        if type(sequence) is not int or not 0 <= sequence <= len(current.contributions):
            raise ValueError("unknown investigation sequence")
        rows = current.contributions[:sequence]
        frontier = rows[-1].frontier if rows else _digest(["anvil.snapshot.v2", investigation])
        return InvestigationSnapshot(investigation, frontier, rows)

    def reconcile(self, investigation: str, idempotency_key: str) -> AcceptedContribution | None:
        """Return proven acceptance. None leaves the outcome unknown."""
        identity = _digest([investigation, idempotency_key])
        try:
            rows = self.replay(investigation).contributions
        except StateCommitUncertain:
            return None
        return next((row for row in rows if row.contribution_id == identity), None)
