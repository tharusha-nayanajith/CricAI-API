import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ReleasePoint:
    frame_idx: int
    timestamp_s: float
    hand_position: tuple[float, float]
    confidence: float
    annotated_frame: np.ndarray
    raw_frame: np.ndarray | None = None


@dataclass(slots=True)
class BallDetection:
    frame_idx: int
    timestamp_s: float
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class FrameBallDetections:
    frame_idx: int
    timestamp_s: float
    detections: list[BallDetection]


class BatterMode(enum.Enum):
    NONE = "none"
    PRESENT = "present"


@dataclass(slots=True)
class BatterROI:
    x: int
    y: int
    width: int
    height: int


class ContactMethod(enum.Enum):
    BALL_VELOCITY = "ball_velocity"
    AUDIO_FALLBACK = "audio_fallback"


@dataclass(slots=True)
class BatContactResult:
    contact_frame_idx: int
    timestamp_s: float
    annotated_frame: np.ndarray
    detection_score: float | None
    method: ContactMethod


@dataclass(slots=True)
class DeliveryContext:
    standardized_video_path: Path
    batter_mode: BatterMode
    batter_roi: BatterROI | None
    fps: float = 0.0
    release_point: ReleasePoint | None = None
    batter_roi_entry_frame_idx: int | None = None
    bat_contact: BatContactResult | None = None
