"""Governed model discovery and selection."""

from .catalog import (
    CatalogModelProvider,
    ExternalModelRegistryProvider,
    FakeModelProvider,
)
from .selection import DeterministicModelSelectionService
from .role_stack import (
    AdapterArtifact,
    CertifiedRoleModel,
    CertifiedRoleModelStack,
    RoleCertification,
    load_certified_role_model_stack,
)

__all__ = [
    "CatalogModelProvider",
    "DeterministicModelSelectionService",
    "ExternalModelRegistryProvider",
    "FakeModelProvider",
    "AdapterArtifact",
    "CertifiedRoleModel",
    "CertifiedRoleModelStack",
    "RoleCertification",
    "load_certified_role_model_stack",
]
