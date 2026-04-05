from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class PresentationMarker(BaseModel):
    id: str
    label: str
    frame_idx: float | None = None
    timestamp_s: float | None = None


class PresentationView(BaseModel):
    id: Literal[
        "three_d_view",
        "bowler_performance",
        "action_legality",
        "shot_classifier",
        "shot_similarity",
    ]
    label: str
    status: Literal["pending", "ready", "failed", "unavailable"]
    button_enabled: bool
    button_order: int
    payload: dict[str, Any] | None = None
    error: str | None = None


class PresentationBundle(BaseModel):
    job_id: str
    overall_status: Literal["pending", "processing", "done", "partial"]
    original_video_url: str | None = None
    playback_video_url: str | None = None
    available_views: list[str]
    markers: list[PresentationMarker]
    views: list[PresentationView]
