"""
Application Configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    
    # Project Info
    PROJECT_NAME: str = "Cricket Bowling Analysis API"
    API_V1_STR: str = "/api/v1"
    
    # Server Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./cricket_sessions.db"
    
    # ML Model Configuration
    MODEL_PATH: str = "app/models/ml/best.pt"
    YOLO_MODEL_PATH: str = "app/models/ml/best.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    STUMP_CLASS_ID: int = 1  # Class ID for "Stumps" in YOLO model (0 is pitch line)
    MIN_STUMP_CONFIDENCE: float = 0.5
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "uploads"
    ALLOWED_VIDEO_EXTENSIONS: list = [".mp4", ".avi", ".mov", ".mkv"]
    
    # Processing Configuration
    FRAME_SKIP: int = 1  # Process every nth frame
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()
