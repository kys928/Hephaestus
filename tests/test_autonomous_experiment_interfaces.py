from hephaestus.interfaces import DatasetDiscoveryProvider, DiagnosisService, TrainingLifecycleService
from hephaestus.schemas.diagnosis_contract import DiagnosisReport, DiagnosisRequest
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingControlRequest, TrainingRunHandle


class FakeDatasetProvider:
    provider_id = "fake"
    def search(self, request: DatasetSearchRequest):
        return [DatasetCandidate("c1", self.provider_id, "fixture")]


class FakeDiagnosisService:
    def diagnose(self, request: DiagnosisRequest) -> DiagnosisReport:
        return DiagnosisReport("d1", request.request_id, request.run_id, request.lineage_id, request.stage_name)


class FakeTrainingService:
    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle:
        return TrainingRunHandle(proposal.run_id, proposal.experiment_id, "fake", "running")
    def control(self, request: TrainingControlRequest) -> TrainingRunHandle:
        return TrainingRunHandle(request.run_id, "exp", "fake", request.action)
    def status(self, run_id: str) -> TrainingRunHandle:
        return TrainingRunHandle(run_id, "exp", "fake", "running")


def test_runtime_checkable_protocols_accept_conforming_fakes() -> None:
    assert isinstance(FakeDatasetProvider(), DatasetDiscoveryProvider)
    assert isinstance(FakeDiagnosisService(), DiagnosisService)
    assert isinstance(FakeTrainingService(), TrainingLifecycleService)
