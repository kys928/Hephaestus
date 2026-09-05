"""Production composition root for the real Hephaestus control loop.

This module is the single place where concrete infrastructure and domain
implementations are assembled. Domain services remain independently testable;
no service imports this composition root.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.control.autonomous_experiment import (
    ApprovalAwareDatasetSelectionService,
    GuardedTrainingLifecycleService,
    IntegratedDiagnosisService,
)
from hephaestus.control.production_autonomy import ProductionAutonomyCoordinator
from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
from hephaestus.data.acquisition_cache import DatasetAcquisitionCache
from hephaestus.data.preprocessing import AutonomousDataPreprocessor, DataProcessingConfig, TokenizerCompatibilityChecker
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.data.remote_acquisition import RemoteDatasetAcquisitionService
from hephaestus.diagnosis.service import EvidenceBasedDiagnosisService
from hephaestus.evaluation.experiment_service import ExperimentEvaluationService
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.providers.datasets.huggingface import HuggingFaceDatasetProvider
from hephaestus.providers.models.catalog import CatalogModelProvider
from hephaestus.providers.models.huggingface import HuggingFaceModelProvider
from hephaestus.providers.models.selection import DeterministicModelSelectionService
from hephaestus.recovery.service import BoundedRecoveryService
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService

from .actions import GovernedActionExecutor
from .recovery import InfrastructureRecoveryController
from .state import (
    DurableRecoveryAttemptStore,
    ProductionLoopStateStore,
    RepositoryIntegrationRecordSink,
)


@dataclass(slots=True)
class ProductionCompositionSettings:
    state_root: Path
    artifact_root: Path
    cache_root: Path | None = None
    database_path: Path | None = None
    model_catalog_path: Path | None = None
    enable_dataset_network: bool = False
    enable_model_network: bool = False
    dataset_provider_allowlist: tuple[str, ...] = ("huggingface",)
    model_provider_allowlist: tuple[str, ...] = ("huggingface", "local_catalog")
    maximum_training_steps: int = 100_000
    maximum_dataset_bytes: int = 512 * 1024 * 1024
    maximum_dataset_rows: int = 1_000_000
    maximum_infrastructure_attempts: int = 5

    def resolved_cache_root(self) -> Path:
        return self.cache_root or self.state_root / "cache"

    def resolved_database_path(self) -> Path:
        return self.database_path or self.state_root / "production.sqlite3"


@dataclass(slots=True)
class ProductionRuntime:
    settings: ProductionCompositionSettings
    state_repository: SQLiteStateRepository
    artifact_store: FileSystemArtifactStore
    secrets_provider: EnvironmentSecretsProvider
    diagnosis_service: IntegratedDiagnosisService
    planner: ClosedLoopExperimentPlanner
    dataset_registry: DatasetProviderRegistry
    dataset_selector: ApprovalAwareDatasetSelectionService
    dataset_acquisition: RemoteDatasetAcquisitionService
    data_preprocessor: AutonomousDataPreprocessor
    model_providers: dict[str, object]
    model_selector: DeterministicModelSelectionService
    training_lifecycle: GuardedTrainingLifecycleService
    evaluator: ExperimentEvaluationService
    judge: SemanticComparisonJudgeAdapter
    recovery: BoundedRecoveryService
    infrastructure_recovery: InfrastructureRecoveryController
    action_executor: GovernedActionExecutor
    loop_state: ProductionLoopStateStore
    coordinator: ProductionAutonomyCoordinator

    def service_inventory(self) -> dict[str, str]:
        return {
            "state_repository": type(self.state_repository).__name__,
            "artifact_store": type(self.artifact_store).__name__,
            "secrets_provider": type(self.secrets_provider).__name__,
            "diagnosis": type(self.diagnosis_service).__name__,
            "planner": type(self.planner).__name__,
            "dataset_discovery": type(self.dataset_registry).__name__,
            "dataset_selection": type(self.dataset_selector).__name__,
            "dataset_acquisition": type(self.dataset_acquisition).__name__,
            "preprocessing": type(self.data_preprocessor).__name__,
            "model_discovery": ",".join(sorted(type(item).__name__ for item in self.model_providers.values())),
            "model_selection": type(self.model_selector).__name__,
            "training_lifecycle": type(self.training_lifecycle).__name__,
            "evaluator": type(self.evaluator).__name__,
            "judge": type(self.judge).__name__,
            "scientific_recovery": type(self.recovery).__name__,
            "infrastructure_recovery": type(self.infrastructure_recovery).__name__,
            "action_executor": type(self.action_executor).__name__,
        }


@dataclass(slots=True)
class ProductionCompositionRoot:
    settings: ProductionCompositionSettings
    tokenizer_checker: TokenizerCompatibilityChecker | None = None

    def build(self) -> ProductionRuntime:
        settings = self.settings
        settings.state_root.mkdir(parents=True, exist_ok=True)
        settings.artifact_root.mkdir(parents=True, exist_ok=True)
        settings.resolved_cache_root().mkdir(parents=True, exist_ok=True)

        state = SQLiteStateRepository(settings.resolved_database_path())
        artifacts = FileSystemArtifactStore(settings.artifact_root)
        secrets = EnvironmentSecretsProvider()
        record_sink = RepositoryIntegrationRecordSink(state)

        diagnosis = IntegratedDiagnosisService(EvidenceBasedDiagnosisService())
        planner = ClosedLoopExperimentPlanner()

        hf_dataset = HuggingFaceDatasetProvider(enable_network=settings.enable_dataset_network)
        registry = DatasetProviderRegistry(
            provider_allowlist={item.casefold() for item in settings.dataset_provider_allowlist}
        )
        registry.register(hf_dataset)
        dataset_selector = ApprovalAwareDatasetSelectionService()
        acquisition = RemoteDatasetAcquisitionService(
            providers={hf_dataset.provider_id: hf_dataset},
            cache=DatasetAcquisitionCache(settings.resolved_cache_root() / "datasets"),
            secrets_provider=secrets,
            artifact_store=artifacts,
        )
        preprocessor = AutonomousDataPreprocessor(
            DataProcessingConfig(
                artifact_root=settings.artifact_root / "prepared-data",
                max_input_bytes=settings.maximum_dataset_bytes,
                max_rows=min(settings.maximum_dataset_rows, 1_000_000),
            ),
            tokenizer_checker=self.tokenizer_checker,
        )

        configured_model_providers = {item.casefold() for item in settings.model_provider_allowlist}
        model_providers: dict[str, object] = {}
        if "huggingface" in configured_model_providers:
            hf_model = HuggingFaceModelProvider(enable_network=settings.enable_model_network)
            model_providers[hf_model.provider_id] = hf_model
        if settings.model_catalog_path is not None and "local_catalog" in configured_model_providers:
            catalog = CatalogModelProvider.from_json(settings.model_catalog_path)
            model_providers[catalog.provider_id] = catalog
        model_selector = DeterministicModelSelectionService()

        raw_training = TransformersTrainingLifecycleService(
            settings.artifact_root / "training-runs",
            maximum_allowed_steps=settings.maximum_training_steps,
            maximum_dataset_bytes=settings.maximum_dataset_bytes,
            maximum_rows=settings.maximum_dataset_rows,
        )
        training = GuardedTrainingLifecycleService(raw_training)
        evaluator = ExperimentEvaluationService()
        judge = SemanticComparisonJudgeAdapter()

        recovery_store = DurableRecoveryAttemptStore(state)
        recovery = BoundedRecoveryService(recovery_store)
        infra_recovery = InfrastructureRecoveryController(
            state,
            maximum_attempts=settings.maximum_infrastructure_attempts,
        )
        action_executor = GovernedActionExecutor(settings.state_root, state)
        loop_state = ProductionLoopStateStore(state)

        coordinator = ProductionAutonomyCoordinator(
            diagnosis_service=diagnosis,
            planner=planner,
            dataset_registry=registry,
            dataset_selector=dataset_selector,
            model_providers=model_providers,  # type: ignore[arg-type]
            model_selector=model_selector,
            training_service=training,
            evaluation_service=evaluator,
            record_sink=record_sink,
            data_preprocessor=preprocessor,
        )

        return ProductionRuntime(
            settings=settings,
            state_repository=state,
            artifact_store=artifacts,
            secrets_provider=secrets,
            diagnosis_service=diagnosis,
            planner=planner,
            dataset_registry=registry,
            dataset_selector=dataset_selector,
            dataset_acquisition=acquisition,
            data_preprocessor=preprocessor,
            model_providers=model_providers,
            model_selector=model_selector,
            training_lifecycle=training,
            evaluator=evaluator,
            judge=judge,
            recovery=recovery,
            infrastructure_recovery=infra_recovery,
            action_executor=action_executor,
            loop_state=loop_state,
            coordinator=coordinator,
        )
