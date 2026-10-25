"""Explicit source-field mapping before generic normalization.

No fallback between gold answers, student answers, schemas and feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
from hephaestus.schemas.dataset_mapping import DatasetRecordMapping
from hephaestus.utils.hashing import hash_json


@dataclass(slots=True)
class PinnedDatasetRecordMapper:
    mapping: DatasetRecordMapping

    @property
    def mapper_id(self):
        return "dataset-record-mapping:sha256:" + hash_json(self.mapping.to_dict())

    def map(self, record):
        fields = [*self.mapping.prompt_fields, self.mapping.target_field]
        if (not fields or len(set(fields)) != len(fields) or
                any(not isinstance(record.get(key), str) or not record[key].strip() for key in fields)):
            return None
        if self.mapping.raw_text:
            if self.mapping.prompt_fields:
                raise ValueError("raw text mapping cannot insert prompt fields")
            return {"text": record[self.mapping.target_field]}
        if not self.mapping.prompt_fields:
            raise ValueError("supervised mapping requires explicit prompt fields")
        # Named boundaries preserve data and schema as input; neither becomes
        # the gold answer or disappears when an instruction column is present.
        prompt = "\n\n".join(f"{key}:\n{record[key]}" for key in self.mapping.prompt_fields)
        return {"prompt": prompt, "target": record[self.mapping.target_field]}
