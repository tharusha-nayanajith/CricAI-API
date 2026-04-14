from __future__ import annotations

from typing import Any, Literal

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
    
    # Optional analysis fields - populated if mistake analysis requested
    visual_feedback: dict[str, Any] | None = Field(default=None)
    mistake_analysis: list[dict[str, Any]] | None = Field(default=None)
    coaching_feedback: str | None = Field(default=None)
    correction_summary: str | None = Field(default=None)
