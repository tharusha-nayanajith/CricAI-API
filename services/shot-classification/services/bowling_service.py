from typing import List
from statistics import mean

from models.schemas import BowlingAnalyzeResponse
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def analyze_bowling(measurements: List[float]) -> BowlingAnalyzeResponse:
    if not measurements:
        logger.info("Bowling analysis: empty measurements")
        return BowlingAnalyzeResponse(score=0.0, insights=["No data provided"]) 

    avg = mean(measurements)
    score = max(0.0, min(1.0, avg / 100.0))

    insights = []
    if score > 0.8:
        insights.append("Excellent consistency and speed")
    elif score > 0.5:
        insights.append("Good control; room for improvement")
    else:
        insights.append("Work on release and follow-through")

    logger.debug(f"Bowling analysis score={score:.3f}")
    return BowlingAnalyzeResponse(score=score, insights=insights)
