"""Clear errors for optional infrastructure capabilities."""


class OptionalCapabilityError(RuntimeError):
    """Raised when a configured optional adapter dependency is unavailable."""
