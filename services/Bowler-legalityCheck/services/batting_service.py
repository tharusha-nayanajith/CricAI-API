from statistics import mean

from models.schemas import BattingClassifyResponse
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def classify_batting(features: list[float]) -> BattingClassifyResponse:
    if not features:
        return BattingClassifyResponse(label="unknown", confidence=0.0)

    m = mean(features)
    if m >= 0.5:
        label = "aggressive"
        conf = min(1.0, m)
    else:
        label = "defensive"
        conf = min(1.0, 1.0 - m)

    logger.debug(f"Batting classified as {label} (confidence={conf:.2f})")
    return BattingClassifyResponse(label=label, confidence=conf)
