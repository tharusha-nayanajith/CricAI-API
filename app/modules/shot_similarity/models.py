from __future__ import annotations

from pydantic import BaseModel


class PoseKeypoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: float = 1.0


class ShotReference(BaseModel):
    keypoints: list[PoseKeypoint]


class ShotSimilarityResult(BaseModel):
    similarity_percentage: float
    matched_player: str
    shot_type: str
    keypoints_detected: int
    confidence: float
    feedback: list[str]
    compared_frame: str
    video_url: str | None = None
    ai_feedback: str | None = None
    visualization_video_url: str | None = None
    normalized_user_url: str | None = None
    normalized_reference_url: str | None = None
