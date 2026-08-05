"""Control-spine adapter for evidence-based diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

from hephaestus.interfaces.services import DiagnosisService
from hephaestus.schemas.diagnosis_contract import DiagnosisReport, DiagnosisRequest


@dataclass(slots=True)
class DiagnosticianRole:
    service: DiagnosisService
    name: str = "diagnostician"

    def run(self, request: DiagnosisRequest) -> DiagnosisReport:
        """Recommend diagnostic findings without executing an intervention."""

        return self.service.diagnose(request)
