"""Governed dataset discovery providers.

Providers only discover metadata. Selection, approval, and acquisition are
separate data-factory boundaries.
"""

from .acquisition import (
    DatasetProviderAcquisitionError,
    ProviderDatasetFile,
    ProviderDatasetSnapshot,
    RemoteDatasetAcquisitionProvider,
)
from .fake import FakeDatasetProvider
from .huggingface import HuggingFaceApiClient, HuggingFaceDatasetProvider
from .local_fixture import LocalFixtureDatasetProvider, LocalFixtureDescriptor

__all__ = [
    "DatasetProviderAcquisitionError",
    "FakeDatasetProvider",
    "HuggingFaceApiClient",
    "HuggingFaceDatasetProvider",
    "LocalFixtureDatasetProvider",
    "LocalFixtureDescriptor",
    "ProviderDatasetFile",
    "ProviderDatasetSnapshot",
    "RemoteDatasetAcquisitionProvider",
]
