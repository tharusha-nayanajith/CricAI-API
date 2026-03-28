from enum import StrEnum

from pydantic import BaseModel

YORKER_MAX = 2.0
FULL_MAX = 4.0
GOOD_LENGTH_MAX = 7.0
SHORT_OF_LENGTH_MAX = 9.0


class LengthClass(StrEnum):
    YORKER = "yorker"
    FULL = "full"
    GOOD_LENGTH = "good_length"
    SHORT_OF_LENGTH = "short_of_length"
    SHORT = "short"


class BouncePoint(BaseModel):
    x_metres: float
    z_metres: float


class BowlerPerformanceResult(BaseModel):
    speed_kmh: float
    swing_metres: float
    bounce_point: BouncePoint | None
    length_class: LengthClass | None
    confidence: float
    inlier_count: int
    raw_speed_ms: float


def classify_length(z_metres: float) -> LengthClass:
    if z_metres <= YORKER_MAX:
        return LengthClass.YORKER
    if z_metres <= FULL_MAX:
        return LengthClass.FULL
    if z_metres <= GOOD_LENGTH_MAX:
        return LengthClass.GOOD_LENGTH
    if z_metres <= SHORT_OF_LENGTH_MAX:
        return LengthClass.SHORT_OF_LENGTH
    return LengthClass.SHORT
