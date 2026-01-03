"""
Session Management Endpoints
Handles bowling session creation and calibration
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
import cv2
import numpy as np
import json
import uuid
from typing import Dict, Any

from app.db.database import get_db
from app.db.models import BowlingSession
from app.models.schemas import SessionStartResponse, BoundingBox
from app.services.vision_engine import vision_engine
from app.services.calibration_service import (
    StumpDetection,
    separate_near_far_stumps,
    calculate_homography,
    serialize_matrix
)

router = APIRouter()


@router.post("/start", response_model=SessionStartResponse)
async def start_session(
    file: UploadFile = File(..., description="Image of pitch with stumps"),
    pitch_length: float = Form(22.0, description="Pitch length in yards"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Start a new bowling session with camera calibration
    
    This endpoint:
    1. Detects stumps in the uploaded image using YOLO
    2. Validates that exactly 6 stumps are detected (2 sets of 3)
    3. Calculates homography matrix for perspective transformation
    4. Saves session data to database
    5. Returns session ID and calibration data
    
    Args:
        file: Image file showing cricket pitch with visible stumps
        pitch_length: Length of cricket pitch in yards (default: 22.0)
        db: Database session (injected)
        
    Returns:
        Session data with calibration matrix and visualization
        
    Raises:
        400: Invalid image file
        422: Insufficient stumps detected or calibration failed
        500: Server error
    """
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        height, width = img.shape[:2]
        
        # Ensure YOLO model is loaded
        if not vision_engine.is_loaded():
            try:
                vision_engine.load_model()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load YOLO model: {str(e)}"
                )
        
        # Detect stumps using YOLO
        stump_detections = vision_engine.detect_stumps(img)
        
        # Handle both detection modes:
        # Mode 1: 6 individual stumps detected
        # Mode 2: 2 stump sets detected (each set contains 3 stumps)
        
        if len(stump_detections) < 2:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient stumps detected. Found {len(stump_detections)}, need at least 2 stump sets"
            )
        
        # If we have 2-3 detections, assume they are stump sets (not individual stumps)
        # Convert each stump set into 3 virtual individual stumps
        if len(stump_detections) <= 3:
            # Take the 2 most confident detections as stump sets
            stump_sets = sorted(
                stump_detections,
                key=lambda x: x["confidence"],
                reverse=True
            )[:2]
            
            # Convert each stump set bounding box into 3 virtual stumps
            stumps = []
            for stump_set in stump_sets:
                bbox = stump_set["bbox"]
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                width = x2 - x1
                
                # Create 3 virtual stumps: left, center, right
                # Each stump is approximately 1/3 of the total width
                stump_width = width / 3
                
                for i in range(3):
                    stump_x1 = int(x1 + i * stump_width)
                    stump_x2 = int(x1 + (i + 1) * stump_width)
                    stump_center_x = (stump_x1 + stump_x2) // 2
                    
                    stumps.append(StumpDetection(
                        x1=stump_x1,
                        y1=y1,
                        x2=stump_x2,
                        y2=y2,
                        confidence=stump_set["confidence"],
                        center_x=stump_center_x,
                        center_y=bbox["center_y"]
                    ))
        
        # If we have 6+ detections, assume they are individual stumps
        elif len(stump_detections) >= 6:
            # Take the 6 most confident detections
            stump_detections = sorted(
                stump_detections,
                key=lambda x: x["confidence"],
                reverse=True
            )[:6]
            
            # Convert detections to StumpDetection objects
            stumps = [
                StumpDetection(
                    x1=d["bbox"]["x1"],
                    y1=d["bbox"]["y1"],
                    x2=d["bbox"]["x2"],
                    y2=d["bbox"]["y2"],
                    confidence=d["confidence"],
                    center_x=d["bbox"]["center_x"],
                    center_y=d["bbox"]["center_y"]
                )
                for d in stump_detections
            ]
        
        else:
            # 4-5 detections is ambiguous
            raise HTTPException(
                status_code=422,
                detail=f"Ambiguous stump detection. Found {len(stump_detections)} stumps. Expected either 2 stump sets or 6 individual stumps."
            )
        
        # Separate into near and far stump sets
        try:
            near_stumps, far_stumps = separate_near_far_stumps(stumps)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        # Calculate homography matrix
        try:
            homography_matrix = calculate_homography(
                near_stumps,
                far_stumps,
                pitch_length
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to calculate calibration: {str(e)}"
            )
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Serialize matrix for database storage
        matrix_json = serialize_matrix(homography_matrix)
        
        # Serialize detection data
        detections_json = json.dumps([
            {
                "x1": s.x1,
                "y1": s.y1,
                "x2": s.x2,
                "y2": s.y2,
                "confidence": s.confidence,
                "center_x": s.center_x,
                "center_y": s.center_y
            }
            for s in stumps
        ])
        
        # Save to database
        db_session = BowlingSession(
            id=session_id,
            pitch_length_yards=pitch_length,
            calibration_matrix=matrix_json,
            image_width=width,
            image_height=height,
            stump_detections=detections_json
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        # Prepare visualization data
        visualization = {
            "image_width": width,
            "image_height": height,
            "stumps_detected": len(stumps),
            "detections": [
                {
                    "bounding_box": {
                        "x1": s.x1,
                        "y1": s.y1,
                        "x2": s.x2,
                        "y2": s.y2,
                        "width": s.x2 - s.x1,
                        "height": s.y2 - s.y1,
                        "center_x": s.center_x,
                        "center_y": s.center_y
                    },
                    "confidence": s.confidence
                }
                for s in stumps
            ]
        }
        
        return {
            "session_id": session_id,
            "status": "calibrated",
            "pitch_length_yards": pitch_length,
            "calibration_matrix": homography_matrix.tolist(),
            "visualization": visualization
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Retrieve session data by ID
    
    Args:
        session_id: Unique session identifier
        db: Database session
        
    Returns:
        Session data including calibration matrix
        
    Raises:
        404: Session not found
    """
    session = db.query(BowlingSession).filter(BowlingSession.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Deserialize calibration matrix
    matrix = json.loads(session.calibration_matrix)
    
    return {
        "session_id": session.id,
        "pitch_length_yards": session.pitch_length_yards,
        "calibration_matrix": matrix,
        "image_width": session.image_width,
        "image_height": session.image_height,
        "created_at": session.created_at.isoformat() if session.created_at else None
    }
