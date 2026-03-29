class CrickAIError(Exception):
    """Base application exception."""


class PreprocessingError(CrickAIError):
    """Raised when video preprocessing fails unexpectedly."""


class CalibrationError(CrickAIError):
    """Raised when calibration data handling fails unexpectedly."""


class FeatureError(CrickAIError):
    """Raised when a feature module fails unexpectedly."""


class AuthenticationError(CrickAIError):
    """Raised when auth credentials or tokens are invalid."""


class AuthorizationError(CrickAIError):
    """Raised when the current user is not allowed to access a resource."""


class ConflictError(CrickAIError):
    """Raised when a unique-resource conflict occurs."""
