from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.modules.preprocessor.models import (
    BallDetection,
    BatContactResult,
    BatterMode,
    FrameBallDetections,
    ReleasePoint,
)


@dataclass(slots=True)
class VideoArtifacts:
    release_frame: np.ndarray
    ball_path: list[BallDetection]
    bat_contact_frame: np.ndarray | None
    standardized_video_path: Path | None = None
    release_point: ReleasePoint | None = None
    batter_mode: BatterMode | None = None
    batter_roi_entry_frame_idx: int | None = None
    bat_contact: BatContactResult | None = None
    ball_candidates_by_frame: list[FrameBallDetections] = field(default_factory=list)
