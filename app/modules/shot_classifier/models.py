from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ShotClassifierResult(BaseModel):
    predicted_shot: str
    confidence: float
    probabilities: dict[str, float]
    frames_used: int
    frame_start_index: int
    frame_end_index: int
    roi_entry_frame_index: int | None = None
    trigger_source: Literal["batter_roi_entry", "bat_contact_fallback"]
    video_url: str | None = Field(default=None)
