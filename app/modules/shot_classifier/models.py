from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SHOT_CLASS_LABELS = (
    "cut",
    "drive",
    "flick",
    "pull",
    "slog",
    "sweep",
    "misc",
)

SHOT_CLASS_ALIASES = {
    "cover_drive": "drive",
    "cover drive": "drive",
    "straight_drive": "drive",
    "straight drive": "drive",
    "drive_shot": "drive",
    "drive shot": "drive",
    "pull_shot": "pull",
    "pull shot": "pull",
    "cut_shot": "cut",
    "cut shot": "cut",
    "flick_shot": "flick",
    "flick shot": "flick",
    "slog_shot": "slog",
    "slog shot": "slog",
    "sweep_shot": "sweep",
    "sweep shot": "sweep",
}


def normalize_shot_label(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        return None

    return SHOT_CLASS_ALIASES.get(normalized, normalized)


class ShotClassifierResult(BaseModel):
    predicted_shot: str
    confidence: float
    probabilities: dict[str, float]
    frames_used: int
    frame_start_index: int
    frame_end_index: int
    roi_entry_frame_index: int | None = None
    trigger_source: Literal["batter_roi_entry", "bat_contact_fallback", "ball_path_end_fallback"]
    video_url: str | None = Field(default=None)
    intended_shot: str | None = Field(default=None)
    intent_match: bool | None = Field(default=None)
    intended_shot_score: float | None = Field(default=None)
    mistake_analysis_basis: Literal["predicted_shot", "intended_shot"] | None = Field(default=None)
    mistake_analysis_reference_shot: str | None = Field(default=None)
    technique_map: dict[str, float] | None = Field(default=None)
    technique_map_basis: Literal["pose_landmark_heuristic"] | None = Field(default=None)
    technique_details: list[dict[str, Any]] | None = Field(default=None)

    # Optional analysis fields - populated if mistake analysis requested
    visual_feedback: dict[str, Any] | None = Field(default=None)
    mistake_analysis: list[dict[str, Any]] | None = Field(default=None)
    coaching_feedback: str | None = Field(default=None)
    correction_summary: str | None = Field(default=None)
