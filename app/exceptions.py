class CrickAIError(Exception):
    """Base application exception."""


class PreprocessingError(CrickAIError):
    """Raised when video preprocessing fails unexpectedly."""


class CalibrationError(CrickAIError):
    """Raised when calibration data handling fails unexpectedly."""


class FeatureError(CrickAIError):
    """Raised when a feature module fails unexpectedly."""
