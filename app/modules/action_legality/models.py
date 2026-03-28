from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ActionLegalityScaler(BaseModel):
    mean: list[float]
    scale: list[float]


class ActionLegalityMetadata(BaseModel):
    feature_dim: int
    select_landmarks: list[int]


class ActionLegalityResult(BaseModel):
    verdict: Literal["legal", "illegal"]
    illegal_probability: float
    legal_probability: float
    confidence: float
    release_frame_index: int | None = None
    release_timestamp_s: float | None = None
    release_confidence: float | None = None
    selected_landmarks: list[int]
    normalized_keypoints: list[float]
    video_url: str | None = None
    used_annotated_release_frame: bool = False
