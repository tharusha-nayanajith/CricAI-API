"""
SQLAlchemy database models
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Text
from sqlalchemy.sql import func
import uuid

from app.db.database import Base


class BowlingSession(Base):
    """
    Bowling session model for storing calibration data
    
    Attributes:
        id: Unique session identifier (UUID)
        pitch_length_yards: Length of the cricket pitch in yards
        calibration_matrix: JSON-serialized 3x3 homography matrix
        image_width: Width of the calibration image in pixels
        image_height: Height of the calibration image in pixels
        stump_detections: JSON-serialized stump detection data
        created_at: Timestamp when session was created
        updated_at: Timestamp when session was last updated
    """
    
    __tablename__ = "bowling_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pitch_length_yards = Column(Float, nullable=False)
    calibration_matrix = Column(Text, nullable=False)  # JSON serialized 3x3 matrix
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    stump_detections = Column(Text, nullable=True)  # JSON serialized detection data
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<BowlingSession(id={self.id}, pitch_length={self.pitch_length_yards} yards)>"
