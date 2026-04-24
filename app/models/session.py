from typing import Literal

from pydantic import BaseModel

from app.models.job import FeatureResult


class SessionDeliveryRef(BaseModel):
    delivery_id: str
    filename: str


class SessionCreateResponse(BaseModel):
    session_id: str
    delivery_ids: list[str]


class SessionProgress(BaseModel):
    total: int
    pending: int
    processing: int
    completed: int
    failed: int
    partial: int


class BowlerSessionDelivery(BaseModel):
    delivery_id: str
    filename: str
    overall_status: Literal["pending", "processing", "done", "partial", "failed"]
    bowler_performance: FeatureResult


class SessionSummary(BaseModel):
    avg_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    avg_wicket_risk_percentage: float | None = None
    length_breakdown: dict[str, int]


class SessionResult(BaseModel):
    session_id: str
    overall_status: Literal["pending", "processing", "done", "partial", "failed"]
    progress: SessionProgress
    deliveries: list[BowlerSessionDelivery]
    summary: SessionSummary
