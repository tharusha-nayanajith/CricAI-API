from fastapi import APIRouter

from models.schemas import SimilarityCompareRequest, SimilarityCompareResponse
from services.similarity_service import compare_vectors


router = APIRouter(prefix="/api/similarity", tags=["similarity"])


@router.post("/compare", response_model=SimilarityCompareResponse)
def compare(req: SimilarityCompareRequest) -> SimilarityCompareResponse:
    return compare_vectors(req.a, req.b, req.metric)
