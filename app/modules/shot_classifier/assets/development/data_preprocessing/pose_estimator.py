"""
RTMPose Pose Estimation Module
Estimates body keypoints from cricket shot frames
"""

import numpy as np
from typing import List, Dict, Tuple
import cv2
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    RTMPOSE_CONFIG,
    RTMPOSE_CHECKPOINT
)


class PoseEstimator:
    """Estimate pose keypoints using RTMPose"""
    
    def __init__(self, config_file: str = None, checkpoint_file: str = None):
        """
        Initialize RTMPose model
        
        Args:
            config_file: Path to RTMPose config
            checkpoint_file: Path to pretrained weights
        """
        # Default RTMPose-m model (good balance of speed and accuracy)
        if config_file is None:
            config_file = RTMPOSE_CONFIG
        if checkpoint_file is None:
            checkpoint_file = RTMPOSE_CHECKPOINT
        
        print("Initializing RTMPose model...")
        self.model = init_model(config_file, checkpoint_file, device='cpu')
        print("RTMPose model loaded successfully")
    
    def estimate_pose(self, frame: np.ndarray) -> Dict:
        """
        Estimate pose keypoints from a single frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary containing keypoints and confidence scores
        """
        # Run inference
        results = inference_topdown(self.model, frame)
        
        # Extract keypoints from results
        if len(results) > 0:
            # Get the first person detected
            result = results[0]
            keypoints = result.pred_instances.keypoints[0]  # Shape: (17, 2)
            scores = result.pred_instances.keypoint_scores[0]  # Shape: (17,)
            
            return {
                'keypoints': keypoints,  # (x, y) coordinates
                'scores': scores,  # Confidence scores
                'bbox': result.pred_instances.bboxes[0] if hasattr(result.pred_instances, 'bboxes') else None
            }
        else:
            # No person detected
            return {
                'keypoints': np.zeros((17, 2)),
                'scores': np.zeros(17),
                'bbox': None
            }
    
    def estimate_pose_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Estimate pose for multiple frames
        
        Args:
            frames: List of image frames
            
        Returns:
            List of pose dictionaries
        """
        results = []
        for idx, frame in enumerate(frames):
            print(f"Processing frame {idx + 1}/{len(frames)}")
            pose_data = self.estimate_pose(frame)
            results.append(pose_data)
        
        return results
    
    def visualize_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose keypoints on frame for visualization
        
        Args:
            frame: Input image frame
            pose_data: Pose estimation results
            
        Returns:
            Frame with pose visualization
        """
        vis_frame = frame.copy()
        keypoints = pose_data['keypoints']
        scores = pose_data['scores']
        
        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw keypoints
        for idx, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:  # Only draw confident keypoints
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
                start_pt = tuple(keypoints[start_idx].astype(int))
                end_pt = tuple(keypoints[end_idx].astype(int))
                cv2.line(vis_frame, start_pt, end_pt, (0, 255, 255), 2)
        
        return vis_frame


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle between three points
    
    Args:
        p1, p2, p3: Points as (x, y) coordinates
        
    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)