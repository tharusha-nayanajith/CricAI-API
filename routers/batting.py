"""
Batting API Router
Handles HTTP endpoints for shot analysis
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import Optional

from services.batting_service import get_batting_service

router = APIRouter(prefix="/batting", tags=["Batting Analysis"])


@router.get("/shot-types")
async def get_shot_types():
    """
    Get available shot types for analysis
    
    Returns:
        List of shot types
    """
    try:
        service = get_batting_service()
        shot_types = service.get_shot_types()
        
        return {
            "success": True,
            "shot_types": shot_types,
            "message": "Available shot types retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-shot")
async def analyze_shot(
    video: UploadFile = File(..., description="Cricket shot video"),
    intended_shot: str = Form(..., description="User's intended shot type")
):
    """
    Analyze cricket shot with intent-based scoring
    
    Args:
        video: Video file of the cricket shot
        intended_shot: User's intended shot (e.g., 'cut', 'drive', 'pull')
        
    Returns:
        Analysis results with intent score
    """
    temp_video_path = None
    
    try:
        # Validate file type
        if not video.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload a video file."
            )
        
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_video_path = temp_file.name
        
        # Get service and analyze
        service = get_batting_service()
        result = service.analyze_shot(temp_video_path, intended_shot)
        
        return {
            "success": True,
            "data": result,
            "message": "Shot analysis completed successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@router.post("/batch-analyze")
async def batch_analyze_shots(
    videos: list[UploadFile] = File(..., description="Multiple cricket shot videos"),
    intended_shots: str = Form(..., description="Comma-separated intended shots")
):
    """
    Analyze multiple shots in batch
    
    Args:
        videos: List of video files
        intended_shots: Comma-separated list of intended shots
        
    Returns:
        Batch analysis results
    """
    temp_paths = []
    
    try:
        intended_shot_list = [s.strip() for s in intended_shots.split(',')]
        
        if len(videos) != len(intended_shot_list):
            raise HTTPException(
                status_code=400,
                detail="Number of videos must match number of intended shots"
            )
        
        results = []
        service = get_batting_service()
        
        # Process each video
        for video, intended_shot in zip(videos, intended_shot_list):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                content = await video.read()
                temp_file.write(content)
                temp_path = temp_file.name
                temp_paths.append(temp_path)
            
            # Analyze
            result = service.analyze_shot(temp_path, intended_shot)
            results.append({
                "filename": video.filename,
                "analysis": result
            })
        
        # Calculate aggregate statistics
        avg_intent_score = sum(r['analysis']['intent_score'] for r in results) / len(results)
        correct_count = sum(1 for r in results if r['analysis']['is_correct'])
        
        return {
            "success": True,
            "results": results,
            "summary": {
                "total_shots": len(results),
                "average_intent_score": round(avg_intent_score, 2),
                "correct_predictions": correct_count,
                "accuracy": round((correct_count / len(results)) * 100, 2)
            },
            "message": "Batch analysis completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")
    finally:
        # Clean up all temporary files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)


@router.get("/health")
async def health_check():
    """Check if batting service is running"""
    try:
        service = get_batting_service()
        return {
            "success": True,
            "status": "healthy",
            "message": "Batting service is running"
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "message": str(e)
        }