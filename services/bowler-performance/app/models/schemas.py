"""
Pydantic Schemas for Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============= Request Models =============

class AnalysisRequest(BaseModel):
    """
    Request model for video analysis
    (File upload handled separately via UploadFile)
    """
    frame_skip: Optional[int] = Field(default=1, description="Process every nth frame")
    detect_stumps: Optional[bool] = Field(default=True, description="Auto-detect stumps for calibration")


class CalibrationRequest(BaseModel):
    """
    Request model for manual calibration
    """
    stump_positions: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Manual stump positions if auto-detection fails"
    )


# ============= Response Models =============

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    center_x: int
    center_y: int


class SessionStartResponse(BaseModel):
    """
    Response model for session start endpoint
    """
    session_id: str = Field(description="Unique session identifier")
    status: str = Field(description="Calibration status")
    pitch_length_yards: float = Field(description="Pitch length in yards")
    calibration_matrix: List[List[float]] = Field(description="3x3 homography matrix")
    visualization: Dict[str, Any] = Field(description="Detection visualization data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "status": "calibrated",
                "pitch_length_yards": 22.0,
                "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "visualization": {
                    "image_width": 1920,
                    "image_height": 1080,
                    "stumps_detected": 6,
                    "detections": []
                }
            }
        }


class AnalysisResponse(BaseModel):
    """
    Response model for video analysis results
    """
    status: str = Field(description="Processing status")
    message: Optional[str] = Field(default=None, description="Status message")
    filename: Optional[str] = Field(default=None, description="Processed filename")
    content_type: Optional[str] = Field(default=None, description="File content type")
    
    # Bowling metrics (to be populated later)
    speed_kmh: Optional[float] = Field(default=None, description="Ball speed in km/h")
    line: Optional[str] = Field(default=None, description="Bowling line")
    length: Optional[str] = Field(default=None, description="Bowling length")
    swing_type: Optional[str] = Field(default=None, description="Type of swing")
    bounce_point: Optional[Dict[str, float]] = Field(default=None, description="Ball bounce coordinates")
    trajectory_points: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Ball trajectory coordinates"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "filename": "bowling_video.mp4",
                "speed_kmh": 142.5,
                "line": "off",
                "length": "good",
                "swing_type": "out-swing"
            }
        }


class CalibrationResponse(BaseModel):
    """
    Response model for calibration results
    """
    status: str = Field(description="Calibration status")
    message: Optional[str] = Field(default=None, description="Status message")
    filename: Optional[str] = Field(default=None, description="Calibration image filename")
    
    # Calibration data (to be populated later)
    stump_positions: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Detected stump positions"
    )
    calibration_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="Perspective transformation matrix"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Stumps detected successfully",
                "stump_positions": [
                    {"x": 320, "y": 480},
                    {"x": 340, "y": 480},
                    {"x": 360, "y": 480}
                ]
            }
        }


class DetectionResult(BaseModel):
    """
    Model for object detection results
    """
    class_name: str = Field(description="Detected object class")
    confidence: float = Field(description="Detection confidence score")
    bbox: Dict[str, float] = Field(description="Bounding box coordinates")
    
    class Config:
        json_schema_extra = {
            "example": {
                "class_name": "cricket_ball",
                "confidence": 0.95,
                "bbox": {"x1": 100, "y1": 200, "x2": 120, "y2": 220}
            }
        }


class FrameAnalysisResponse(BaseModel):
    """
    Response model for single frame analysis
    """
    status: str
    detections: List[DetectionResult] = Field(default_factory=list)
    frame_number: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============= Internal Models =============

class BowlingMetrics(BaseModel):
    """
    Complete bowling performance metrics
    """
    delivery_id: str
    speed_kmh: float
    line: str
    length: str
    swing_type: Optional[str] = None
    spin_rpm: Optional[float] = None
    bounce_height_cm: Optional[float] = None
    release_point: Dict[str, float]
    bounce_point: Dict[str, float]
    trajectory: List[Dict[str, float]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
