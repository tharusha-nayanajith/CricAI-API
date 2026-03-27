from dataclasses import dataclass

import numpy as np

from app.modules.preprocessor.models import BallDetection


@dataclass(slots=True)
class VideoArtifacts:
    release_frame: np.ndarray
    ball_path: list[BallDetection]
    bat_contact_frame: np.ndarray | None
