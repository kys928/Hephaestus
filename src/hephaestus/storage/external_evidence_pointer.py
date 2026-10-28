"""Exact, reviewed evidence-provider references. No network or publication authority.

The system can *record* independent immutable destinations; actual publishers
need a separately approved, least-privilege token and post-upload byte verification.
"""
from dataclasses import dataclass
import re

_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")
_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\Z")
_CATEGORIES = frozenset({"research_summary", "public_dataset", "public_model",
                          "internal_training_data", "internal_checkpoint", "original_cpu_evidence",
                          "protected_holdout", "private_reference", "private_checkpoint"})
_PUBLIC = frozenset({"research_summary", "public_dataset", "public_model"})
_TYPES = {"github_release", "huggingface_model", "huggingface_dataset", "kaggle_dataset"}


@dataclass(frozen=True, slots=True)
class ExternalEvidencePointer:
    provider: str
    repo_id: str
    artifact_kind: str
    artifact_sha256: str
    review_sha256: str
    visibility: str
    revision: str
    kaggle_version: int | None = None
    schema: str = "external-evidence-pointer.v1"

    def validate(self):
        if self.schema != "external-evidence-pointer.v1" or self.provider not in _TYPES:
            raise ValueError("unknown evidence storage provider/schema")
        if not isinstance(self.repo_id, str) or not _REPO.fullmatch(self.repo_id):
            raise ValueError("must reference an exact owner/repository identifier")
        if self.artifact_kind not in _CATEGORIES or self.visibility not in ("public", "private"):
            raise ValueError("unsupported evidence classification or visibility")
        if (not isinstance(self.artifact_sha256, str) or not _SHA.fullmatch(self.artifact_sha256) or
                not isinstance(self.review_sha256, str) or not _SHA.fullmatch(self.review_sha256)):
            raise ValueError("source and publication review SHA commitments are mandatory")
        if self.visibility == "public" and self.artifact_kind not in _PUBLIC:
            raise PermissionError("protected, internal, or raw evidence must never be public")
        if self.provider in ("huggingface_model", "huggingface_dataset"):
            if not isinstance(self.revision, str) or not _COMMIT.fullmatch(self.revision):
                raise ValueError("Hugging Face evidence must pin an exact 40-hex commit")
            if self.kaggle_version is not None:
                raise ValueError("Kaggle version not valid for HF repository")
            if self.provider == "huggingface_model" and self.artifact_kind not in ("public_model", "private_checkpoint", "internal_checkpoint"):
                raise ValueError("HF model repo cannot represent an unrelated evidence class")
            if self.provider == "huggingface_dataset" and self.artifact_kind not in ("public_dataset", "internal_training_data", "protected_holdout"):
                raise ValueError("HF dataset repo cannot represent an unrelated evidence class")
        elif self.provider == "kaggle_dataset":
            if (type(self.kaggle_version) is not int or self.kaggle_version < 1 or
                    not isinstance(self.revision, str) or self.revision != str(self.kaggle_version)):
                raise ValueError("Kaggle evidence requires exact positive dataset version")
            if self.artifact_kind not in ("public_dataset", "internal_training_data"):
                raise ValueError("Kaggle dataset provider is for dataset artifacts only")
        elif self.provider == "github_release":
            if (self.kaggle_version is not None or not isinstance(self.revision, str) or
                    not _TAG.fullmatch(self.revision) or self.revision.lower() in ("latest", "main", "master")):
                raise ValueError("GitHub evidence requires exact immutable release tag")
            if self.artifact_kind not in ("research_summary", "original_cpu_evidence"):
                raise ValueError("GitHub releases are not a generic dataset/weight store")
        return self

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("provider pointer must have exactly all schema fields")
        return cls(**payload).validate()

    def to_dict(self):
        self.validate()
        return {key: getattr(self, key) for key in self.__dataclass_fields__}
