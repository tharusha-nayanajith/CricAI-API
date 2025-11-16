from models.schemas import SimilarityCompareResponse
from utils.math_utils import cosine_similarity, euclidean_distance
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def compare_vectors(a: list[float], b: list[float], metric: str) -> SimilarityCompareResponse:
    metric_lower = metric.lower()
    if metric_lower == "cosine":
        sim = cosine_similarity(a, b)
        logger.debug(f"Cosine similarity={sim:.4f}")
        return SimilarityCompareResponse(similarity=sim, metric="cosine")
    elif metric_lower == "euclidean":
        dist = euclidean_distance(a, b)
        if dist == float("inf"):
            sim = 0.0
        else:
            sim = 1.0 / (1.0 + dist)
        logger.debug(f"Euclidean distance={dist:.4f} → similarity={sim:.4f}")
        return SimilarityCompareResponse(similarity=sim, metric="euclidean")
    else:
        logger.info(f"Unknown metric '{metric}', defaulting to cosine")
        sim = cosine_similarity(a, b)
        return SimilarityCompareResponse(similarity=sim, metric="cosine")
