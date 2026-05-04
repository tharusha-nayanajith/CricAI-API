from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.job import FeatureName


class FeatureHistoryPreview(BaseModel):
    status: Literal["pending", "processing", "done", "failed"]
    error: str | None = None
    preview: dict[str, object] | None = None


class JobHistoryItem(BaseModel):
    job_id: str
    session_id: str | None = None
    filename: str
    overall_status: Literal["pending", "processing", "done", "partial", "failed"]
    requested_features: list[FeatureName]
    created_at: datetime
    updated_at: datetime
    thumbnail_image_url: str | None = None
    feature_results: dict[FeatureName, FeatureHistoryPreview] = Field(default_factory=dict)


class JobHistoryResponse(BaseModel):
    items: list[JobHistoryItem]
    total: int
    limit: int
    offset: int


class SessionHistoryItem(BaseModel):
    session_id: str
    overall_status: Literal["pending", "processing", "done", "partial", "failed"]
    delivery_count: int
    created_at: datetime
    updated_at: datetime
    avg_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    thumbnail_image_url: str | None = None


class SessionHistoryResponse(BaseModel):
    items: list[SessionHistoryItem]
    total: int
    limit: int
    offset: int
