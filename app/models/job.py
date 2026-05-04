from typing import Literal

from pydantic import BaseModel, Field

FEATURE_NAMES = (
    "bowler_performance",
    "action_legality",
    "shot_classifier",
    "shot_similarity",
)
FeatureName = Literal[
    "bowler_performance",
    "action_legality",
    "shot_classifier",
    "shot_similarity",
]


class FeatureResult(BaseModel):
    status: Literal["pending", "processing", "done", "failed"]
    result: dict | None = None
    error: str | None = None


class JobStatus(BaseModel):
    job_id: str
    overall_status: Literal["pending", "processing", "done", "partial", "failed"]
    requested_features: list[FeatureName] = Field(default_factory=lambda: list(FEATURE_NAMES))
    bowler_performance: FeatureResult
    action_legality: FeatureResult
    shot_classifier: FeatureResult
    shot_similarity: FeatureResult
