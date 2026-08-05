"""Governed dataset discovery providers.

Providers only discover metadata. Selection, approval, and acquisition are
separate data-factory boundaries.
"""

from .fake import FakeDatasetProvider
from .huggingface import HuggingFaceDatasetProvider
from .local_fixture import LocalFixtureDatasetProvider, LocalFixtureDescriptor

__all__ = [
    "FakeDatasetProvider",
    "HuggingFaceDatasetProvider",
    "LocalFixtureDatasetProvider",
    "LocalFixtureDescriptor",
]
