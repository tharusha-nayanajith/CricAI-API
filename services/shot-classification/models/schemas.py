from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class BowlingAnalyzeRequest(BaseModel):
    measurements: List[float] = Field(default_factory=list)
    hand: Optional[Literal["left", "right"]] = None
    notes: Optional[str] = None


class BowlingAnalyzeResponse(BaseModel):
    score: float
    insights: List[str]


class BattingClassifyRequest(BaseModel):
    features: List[float] = Field(default_factory=list)


class BattingClassifyResponse(BaseModel):
    label: str
    confidence: float


class ActionValidateRequest(BaseModel):
    action_name: str
    features: List[float] = Field(default_factory=list)


class ActionValidateResponse(BaseModel):
    valid: bool
    errors: List[str]


class SimilarityCompareRequest(BaseModel):
    a: List[float] = Field(default_factory=list)
    b: List[float] = Field(default_factory=list)
    metric: Literal["cosine", "euclidean"] = "cosine"


class SimilarityCompareResponse(BaseModel):
    similarity: float
    metric: str


class HealthResponse(BaseModel):
    status: str
    timestamp: float
