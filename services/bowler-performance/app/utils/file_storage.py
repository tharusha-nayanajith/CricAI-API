"""
File Storage Utilities
Handle uploaded file storage and management
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
import uuid

from app.core.config import settings


def ensure_upload_directory() -> Path:
    """
    Ensure upload directory exists
    
    Returns:
        Path to upload directory
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename to avoid collisions
    
    Args:
        original_filename: Original uploaded filename
        
    Returns:
        Unique filename with UUID prefix
    """
    file_extension = Path(original_filename).suffix
    unique_id = uuid.uuid4().hex[:8]
    base_name = Path(original_filename).stem
    return f"{unique_id}_{base_name}{file_extension}"


async def save_upload_file(upload_file: UploadFile, destination: Optional[Path] = None) -> Path:
    """
    Save uploaded file to disk
    
    Args:
        upload_file: FastAPI UploadFile object
        destination: Optional custom destination path
        
    Returns:
        Path to saved file
        
    TODO: Implement actual file saving
    - Validate file size
    - Validate file extension
    - Save file chunks
    - Return file path
    """
    if destination is None:
        upload_dir = ensure_upload_directory()
        filename = generate_unique_filename(upload_file.filename or "upload")
        destination = upload_dir / filename
    
    # TODO: Implement chunked file writing for large files
    # with destination.open("wb") as buffer:
    #     shutil.copyfileobj(upload_file.file, buffer)
    
    pass


def validate_video_file(filename: str) -> bool:
    """
    Validate if file is an allowed video format
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if valid video file, False otherwise
    """
    file_extension = Path(filename).suffix.lower()
    return file_extension in settings.ALLOWED_VIDEO_EXTENSIONS


def cleanup_file(file_path: Path) -> bool:
    """
    Delete a file from disk
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    if file_path.exists():
        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 * 1024)
    return 0.0
