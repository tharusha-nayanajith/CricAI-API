"""
API Router Aggregator
Combines all endpoint routers
"""

from fastapi import APIRouter

from app.api.v1.endpoints import analysis, calibration, health, session


api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(session.router, prefix="/session", tags=["Session"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(calibration.router, prefix="/calibration", tags=["Calibration"])
