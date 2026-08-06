"""Execution-infrastructure adapter boundaries."""

from .config import InfrastructureConfig, InfrastructureConfigError
from .health import DependencyHealthService, HealthReport, HealthService
from .observability import EventSink, OpenTelemetryEventSink, StructuredEvent
from .secrets import (
    AwsSecretsManagerProvider,
    EnvironmentSecretsProvider,
    FileMountedSecretsProvider,
    InjectedSecretsProvider,
    SecretReference,
    SecretsProvider,
)

__all__ = [
    "AwsSecretsManagerProvider",
    "DependencyHealthService",
    "EnvironmentSecretsProvider",
    "EventSink",
    "FileMountedSecretsProvider",
    "HealthReport",
    "HealthService",
    "InfrastructureConfig",
    "InfrastructureConfigError",
    "InjectedSecretsProvider",
    "OpenTelemetryEventSink",
    "SecretReference",
    "SecretsProvider",
    "StructuredEvent",
]
