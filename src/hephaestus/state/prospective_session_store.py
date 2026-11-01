"""Custody exposure intents/results persisted only through the state layer."""
from dataclasses import dataclass
from hephaestus.schemas.prospective_custody import CustodySession
from hephaestus.storage.base import StateRepository

STUDY_SESSIONS = "prospective_custody_sessions"
RAW_SAMPLES = 'prospective_custody_raw_samples'


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

    def retain_authority_receipt(self, session: CustodySession):
        previous=self.get(session.study_id)
        if (previous is None or previous['status']!='consumed' or session.status!='consumed'
                or previous['submission_commitment']!=session.submission_commitment
                or previous.get('exposure_request_ref')!=session.exposure_request_ref
                or previous.get('exposure_receipt_ref') or not session.exposure_receipt_ref):
            raise PermissionError('exposure authority cannot be attached, replaced or reset')
        self.repository.append(STUDY_SESSIONS,session.to_dict())

    def retain_raw_sample(self, link):
        session=self.get(link.study_id)
        if session is None or session['status']!='consumed' or not session.get('exposure_receipt_ref'):
            raise PermissionError('raw sample requires locally and externally consumed authority')
        self.repository.append(RAW_SAMPLES,link.to_dict())

    def raw_samples(self, study_id):
        return [row for row in self.repository.all(RAW_SAMPLES) if row['study_id']==study_id]
