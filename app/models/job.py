from typing import Literal

from pydantic import BaseModel


class FeatureResult(BaseModel):
    status: Literal["pending", "processing", "done", "failed"]
    result: dict | None = None
    error: str | None = None


class JobStatus(BaseModel):
    job_id: str
    overall_status: Literal["pending", "processing", "done", "partial", "failed"]
    bowler_performance: FeatureResult
    action_legality: FeatureResult
    shot_classifier: FeatureResult
    shot_similarity: FeatureResult
