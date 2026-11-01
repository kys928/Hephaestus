"""Custody exposure intents/results persisted only through the state layer."""
from dataclasses import dataclass
from hephaestus.schemas.prospective_custody import CustodySession
from hephaestus.storage.base import StateRepository

STUDY_SESSIONS = "prospective_custody_sessions"


@dataclass(slots=True)
class ProspectiveSessionStore:
    repository: StateRepository

    def get(self, study_id):
        return self.repository.get_latest(STUDY_SESSIONS, "study_id", study_id)

    def consume(self, session: CustodySession):
        if self.get(session.study_id) is not None:
            raise PermissionError("study has already consumed its exposure session")
        self.repository.append(STUDY_SESSIONS, session.to_dict())

    def finish(self, session: CustodySession):
        previous = self.get(session.study_id)
        if (previous is None or previous["status"] != "consumed"
                or previous["submission_commitment"] != session.submission_commitment):
            raise PermissionError("custody session cannot change after completion/failure")
        self.repository.append(STUDY_SESSIONS, session.to_dict())
