from fastapi import APIRouter

from models.schemas import BowlingAnalyzeRequest, BowlingAnalyzeResponse
from services.bowling_service import analyze_bowling


router = APIRouter(prefix="/api/bowling", tags=["bowling"])


@router.post("/analyze", response_model=BowlingAnalyzeResponse)
def analyze(req: BowlingAnalyzeRequest) -> BowlingAnalyzeResponse:
    return analyze_bowling(req.measurements)
