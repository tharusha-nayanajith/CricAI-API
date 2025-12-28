"""
Stance Moment Detection Module
Detects the exact frame where batsman is in stance position (before shot is played)
"""

import numpy as np
from typing import List, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator


class StanceDetector:
    """Detect stance moment in cricket videos"""
    
    def __init__(self):
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_velocity(self, prev_point: np.ndarray, curr_point: np.ndarray) -> float:
        """Calculate velocity between two consecutive frames"""
        return np.linalg.norm(curr_point - prev_point)
    
    def calculate_stability_score(self, pose_data: Dict) -> float:
        """
        Calculate stability score for a frame
        Higher score = more stable (likely stance position)
        
        Criteria:
        - Feet relatively stationary
        - Body upright
        - Arms in waiting position
        - Minimal movement
        """
        keypoints = pose_data['keypoints']
        scores = pose_data['scores']
        
        # Check if all key points are detected
        critical_points = [
            'left_ankle', 'right_ankle', 'left_knee', 'right_knee',
            'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'
        ]
        
        avg_confidence = np.mean([scores[self.keypoint_indices[pt]] for pt in critical_points])
        if avg_confidence < 0.5:
            return 0.0  # Low confidence detection
        
        stability_factors = []
        
        # Factor 1: Body uprightness (check spine angle)
        left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
        left_hip = keypoints[self.keypoint_indices['left_hip']]
        left_knee = keypoints[self.keypoint_indices['left_knee']]
        
        spine_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        # Ideal stance has relatively straight spine (160-180 degrees)
        uprightness = 1.0 - abs(170 - spine_angle) / 170
        stability_factors.append(max(0, uprightness))
        
        # Factor 2: Feet positioning (should be shoulder-width apart)
        left_ankle = keypoints[self.keypoint_indices['left_ankle']]
        right_ankle = keypoints[self.keypoint_indices['right_ankle']]
        left_shoulder_pos = keypoints[self.keypoint_indices['left_shoulder']]
        right_shoulder_pos = keypoints[self.keypoint_indices['right_shoulder']]
        
        feet_distance = np.linalg.norm(left_ankle - right_ankle)
        shoulder_distance = np.linalg.norm(left_shoulder_pos - right_shoulder_pos)
        
        if shoulder_distance > 0:
            feet_ratio = feet_distance / shoulder_distance
            # Ideal ratio is around 1.0-1.5 (feet shoulder-width or slightly wider)
            feet_score = 1.0 - abs(1.2 - feet_ratio) if feet_ratio > 0 else 0
            stability_factors.append(max(0, min(1, feet_score)))
        
        # Factor 3: Arm position (hands should be near body, not extended)
        left_wrist = keypoints[self.keypoint_indices['left_wrist']]
        right_wrist = keypoints[self.keypoint_indices['right_wrist']]
        
        left_hand_to_hip = np.linalg.norm(left_wrist - left_hip)
        right_hand_to_hip = np.linalg.norm(keypoints[self.keypoint_indices['right_wrist']] - 
                                          keypoints[self.keypoint_indices['right_hip']])
        
        # Normalize by shoulder width
        if shoulder_distance > 0:
            left_hand_score = 1.0 - min(1.0, left_hand_to_hip / (shoulder_distance * 2))
            right_hand_score = 1.0 - min(1.0, right_hand_to_hip / (shoulder_distance * 2))
            stability_factors.append((left_hand_score + right_hand_score) / 2)
        
        # Calculate overall stability score
        stability_score = np.mean(stability_factors) * avg_confidence
        
        return stability_score
    
    def calculate_movement_score(self, prev_pose: Dict, curr_pose: Dict, next_pose: Dict) -> float:
        """
        Calculate movement score between frames
        Lower score = less movement (more likely to be stance)
        """
        if prev_pose is None or next_pose is None:
            return float('inf')
        
        # Key points to track for movement
        tracking_points = [
            'left_wrist', 'right_wrist', 'left_ankle', 'right_ankle',
            'left_shoulder', 'right_shoulder'
        ]
        
        movements = []
        
        for point in tracking_points:
            idx = self.keypoint_indices[point]
            
            # Calculate velocity change
            prev_point = prev_pose['keypoints'][idx]
            curr_point = curr_pose['keypoints'][idx]
            next_point = next_pose['keypoints'][idx]
            
            velocity_in = self.calculate_velocity(prev_point, curr_point)
            velocity_out = self.calculate_velocity(curr_point, next_point)
            
            # Total movement
            total_movement = velocity_in + velocity_out
            movements.append(total_movement)
        
        # Average movement across all tracked points
        avg_movement = np.mean(movements)
        
        return avg_movement
    
    def detect_stance_frame(self, pose_sequence: List[Dict]) -> int:
        """
        Detect the stance frame using combined stability and minimal movement
        
        Args:
            pose_sequence: List of pose data from all frames
            
        Returns:
            Index of the stance frame
        """
        if len(pose_sequence) < 5:
            # If too few frames, return early frame
            return min(2, len(pose_sequence) - 1)
        
        scores = []
        
        for i in range(1, len(pose_sequence) - 1):
            prev_pose = pose_sequence[i - 1]
            curr_pose = pose_sequence[i]
            next_pose = pose_sequence[i + 1]
            
            # Calculate stability (higher = more stable)
            stability = self.calculate_stability_score(curr_pose)
            
            # Calculate movement (lower = less movement)
            movement = self.calculate_movement_score(prev_pose, curr_pose, next_pose)
            
            # Combined score: high stability + low movement = likely stance
            # Normalize movement (inverse - lower movement = higher score)
            movement_score = 1.0 / (1.0 + movement)
            
            # Combined score (70% stability, 30% stillness)
            combined_score = (0.7 * stability) + (0.3 * movement_score)
            
            scores.append(combined_score)
        
        # Find frame with highest combined score
        # Bias towards earlier frames (stance happens before shot)
        # Apply exponential decay to prefer earlier frames
        adjusted_scores = []
        for i, score in enumerate(scores):
            # Decay factor: prefer earlier frames (first 40% of video)
            decay = np.exp(-i / (len(scores) * 0.4))
            adjusted_score = score * (0.7 + 0.3 * decay)
            adjusted_scores.append(adjusted_score)
        
        stance_frame_idx = np.argmax(adjusted_scores) + 1  # +1 because we started from index 1
        
        return stance_frame_idx
    
    def extract_stance_keypoints(self, video_frames: List[np.ndarray], 
                                 pose_estimator: PoseEstimator) -> Tuple[np.ndarray, int]:
        """
        Extract keypoints from stance frame
        
        Args:
            video_frames: List of video frames
            pose_estimator: Initialized pose estimator
            
        Returns:
            Tuple of (stance keypoints, frame index)
        """
        # Get pose for all frames
        print(f"Extracting poses from {len(video_frames)} frames...")
        pose_sequence = pose_estimator.estimate_pose_batch(video_frames)
        
        # Detect stance frame
        print("Detecting stance frame...")
        stance_frame_idx = self.detect_stance_frame(pose_sequence)
        
        print(f"Stance detected at frame {stance_frame_idx}/{len(video_frames)}")
        
        # Get keypoints from stance frame
        stance_keypoints = pose_sequence[stance_frame_idx]['keypoints']
        
        return stance_keypoints, stance_frame_idx