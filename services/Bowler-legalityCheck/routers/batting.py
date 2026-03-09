from fastapi import APIRouter

from models.schemas import BattingClassifyRequest, BattingClassifyResponse
from services.batting_service import classify_batting


router = APIRouter(prefix="/api/batting", tags=["batting"])


@router.post("/classify", response_model=BattingClassifyResponse)
def classify(req: BattingClassifyRequest) -> BattingClassifyResponse:
    return classify_batting(req.features)
