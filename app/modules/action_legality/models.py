from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ActionLegalityScaler(BaseModel):
    mean: list[float]
    scale: list[float]


class ActionLegalityMetadata(BaseModel):
    feature_dim: int
    select_landmarks: list[int]


class PoseLandmark2D(BaseModel):
    id: int
    name: str
    x: float
    y: float
    visibility: float


class JointAnalysisEntry(BaseModel):
    joint_id: str
    label: str
    status: Literal["ok", "warning", "critical"]
    score: float
    measured_value: float
    threshold_value: float
    explanation: str
    recommendation: str


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
    release_frame_image_url: str | None = None
    release_frame_width: int | None = None
    release_frame_height: int | None = None
    pose_landmarks_2d: list[PoseLandmark2D] = Field(default_factory=list)
    pose_connections: list[list[int]] = Field(default_factory=list)
    overlay_image_url: str | None = None
    joint_analysis: list[JointAnalysisEntry] = Field(default_factory=list)
    summary_explanation: str | None = None
    coaching_feedback: str | None = None
    used_annotated_release_frame: bool = False
