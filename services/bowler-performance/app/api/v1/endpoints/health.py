"""
Health Check Endpoint
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Simple health check endpoint to verify service is running
    
    Returns:
        dict: Service status and timestamp
    """
    return {
        "status": "healthy",
        "service": "Cricket Bowling Analysis API",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0"
    }
