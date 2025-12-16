"""
Frame Extraction Module
Extracts frames from cricket shot videos at specified FPS
"""

import cv2
import numpy as np
from typing import List, Tuple
import os


class FrameExtractor:
    """Extract frames from video files"""
    
    def __init__(self, fps: int = 10):
        """
        Initialize frame extractor
        
        Args:
            fps: Frames per second to extract
        """
        self.fps = fps
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """
        Extract frames from video at specified FPS
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (list of frames, original video FPS)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        frame_interval = max(1, int(original_fps / self.fps))
        
        frames = []
        frame_count = 0
        
        print(f"Extracting frames from {video_path}")
        print(f"Original FPS: {original_fps}, Extracting every {frame_interval} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames from {total_frames} total frames")
        
        return frames, original_fps
    
    def save_frames(self, frames: List[np.ndarray], output_dir: str, prefix: str = "frame"):
        """
        Save extracted frames to directory
        
        Args:
            frames: List of frame arrays
            output_dir: Directory to save frames
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, frame in enumerate(frames):
            filename = f"{prefix}_{idx:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
        
        print(f"Saved {len(frames)} frames to {output_dir}")


def batch_extract_frames(video_dir: str, output_dir: str, fps: int = 10):
    """
    Extract frames from all videos in a directory
    
    Args:
        video_dir: Directory containing videos
        output_dir: Base directory for output frames
        fps: Frames per second to extract
    """
    extractor = FrameExtractor(fps=fps)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(video_dir) 
                   if os.path.splitext(f)[1].lower() in video_extensions]
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        try:
            frames, _ = extractor.extract_frames(video_path)
            extractor.save_frames(frames, video_output_dir, prefix=video_name)
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")