"""Governed dataset discovery, selection, and bounded local processing."""

from .acquisition import DatasetAcquisitionApproval
from .preprocessing import AutonomousDataPreprocessor, DataFactoryResult, DataProcessingConfig
from .registry import DatasetDiscoveryResult, DatasetProviderRegistry
from .selection import DeterministicDatasetSelectionService

__all__ = [
    "AutonomousDataPreprocessor",
    "DataFactoryResult",
    "DataProcessingConfig",
    "DatasetAcquisitionApproval",
    "DatasetDiscoveryResult",
    "DatasetProviderRegistry",
    "DeterministicDatasetSelectionService",
]
