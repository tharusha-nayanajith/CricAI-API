"""
Stance Consistency API Router
Handles HTTP endpoints for stance consistency analysis
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import List

from services.stance_consistency_service import get_stance_service

router = APIRouter(prefix="/stance-consistency", tags=["Stance Consistency"])


@router.post("/analyze")
async def analyze_stance_consistency(
    videos: List[UploadFile] = File(..., description="6 cricket stance videos")
):
    """
    Analyze batting stance consistency across multiple videos
    
    Args:
        videos: List of 6 video files showing batting stances
        
    Returns:
        Comprehensive consistency analysis with AI-generated feedback
    """
    temp_paths = []
    
    try:
        # Validate number of videos
        if len(videos) < 2:
            raise HTTPException(
                status_code=400,
                detail="Minimum 2 videos required for consistency analysis"
            )
        
        if len(videos) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 videos allowed for analysis"
            )
        
        # Validate file types
        for video in videos:
            if not video.content_type.startswith('video/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for {video.filename}. Only video files are accepted."
                )
        
        # Save all videos temporarily
        print(f"Receiving {len(videos)} videos for analysis...")
        for i, video in enumerate(videos, 1):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                content = await video.read()
                temp_file.write(content)
                temp_paths.append(temp_file.name)
                print(f"Saved video {i}/{len(videos)}: {video.filename}")
        
        # Get service and analyze
        service = get_stance_service()
        results = service.analyze_consistency(temp_paths)
        
        return {
            "success": True,
            "data": results,
            "message": "Stance consistency analysis completed successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    finally:
        # Clean up temporary files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_path}: {e}")


@router.post("/quick-compare")
async def quick_compare_two_stances(
    video1: UploadFile = File(..., description="First stance video"),
    video2: UploadFile = File(..., description="Second stance video")
):
    """
    Quick comparison between two stances
    
    Args:
        video1: First video file
        video2: Second video file
        
    Returns:
        Similarity score and basic comparison
    """
    temp_paths = []
    
    try:
        # Validate file types
        for video in [video1, video2]:
            if not video.content_type.startswith('video/'):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Only video files are accepted."
                )
        
        # Save videos temporarily
        for video in [video1, video2]:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                content = await video.read()
                temp_file.write(content)
                temp_paths.append(temp_file.name)
        
        # Analyze
        service = get_stance_service()
        results = service.analyze_consistency(temp_paths)
        
        # Simplify response for quick compare
        similarity_score = results['consistency_analysis']['pairwise_similarities'][0]['similarity']
        
        quick_results = {
            'similarity_score': similarity_score,
            'rating': 'High Similarity' if similarity_score >= 80 else 
                     'Moderate Similarity' if similarity_score >= 60 else 
                     'Low Similarity',
            'video1_consistency': results['individual_video_scores'][0]['consistency_score'],
            'video2_consistency': results['individual_video_scores'][1]['consistency_score'],
            'feedback': results['feedback']['overall_assessment']
        }
        
        return {
            "success": True,
            "data": quick_results,
            "message": "Quick comparison completed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )
    finally:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_path}: {e}")


@router.get("/health")
async def health_check():
    """Check if stance consistency service is running"""
    try:
        service = get_stance_service()
        return {
            "success": True,
            "status": "healthy",
            "message": "Stance consistency service is running"
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "message": str(e)
        }


@router.get("/info")
async def get_service_info():
    """Get information about stance consistency analysis"""
    return {
        "success": True,
        "service": "Stance Consistency Analysis",
        "description": "Analyzes batting stance consistency using pose estimation and cosine similarity",
        "features": [
            "Automatic stance moment detection",
            "RTMPose-based keypoint extraction",
            "Cosine similarity comparison",
            "AI-generated coaching feedback",
            "Pairwise similarity analysis",
            "Reference stance calculation"
        ],
        "requirements": {
            "min_videos": 2,
            "max_videos": 10,
            "recommended_videos": 6,
            "video_format": "MP4 preferred",
            "video_duration": "3-10 seconds per video"
        }
    }