"""
Feature Engineering Module
Extracts features from pose keypoints and detects shot moment
"""

import numpy as np
from typing import List, Dict, Tuple
from ..utils.config import KEYPOINT_INDICES, ALPHA, BETA


class FeatureEngineer:
    """Extract features from pose keypoints"""
    
    def __init__(self):
        self.keypoint_indices = KEYPOINT_INDICES
        self.alpha = ALPHA
        self.beta = BETA
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)
    
    def extract_body_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Extract key body angles from keypoints
        
        Args:
            keypoints: Array of shape (17, 2) containing (x, y) coordinates
            
        Returns:
            Dictionary of angle names and values
        """
        angles = {}
        
        # Right elbow angle
        angles['right_elbow'] = self.calculate_angle(
            keypoints[self.keypoint_indices['right_shoulder']],
            keypoints[self.keypoint_indices['right_elbow']],
            keypoints[self.keypoint_indices['right_wrist']]
        )
        
        # Left elbow angle
        angles['left_elbow'] = self.calculate_angle(
            keypoints[self.keypoint_indices['left_shoulder']],
            keypoints[self.keypoint_indices['left_elbow']],
            keypoints[self.keypoint_indices['left_wrist']]
        )
        
        # Right shoulder angle
        angles['right_shoulder'] = self.calculate_angle(
            keypoints[self.keypoint_indices['right_elbow']],
            keypoints[self.keypoint_indices['right_shoulder']],
            keypoints[self.keypoint_indices['right_hip']]
        )
        
        # Left shoulder angle
        angles['left_shoulder'] = self.calculate_angle(
            keypoints[self.keypoint_indices['left_elbow']],
            keypoints[self.keypoint_indices['left_shoulder']],
            keypoints[self.keypoint_indices['left_hip']]
        )
        
        # Right knee angle
        angles['right_knee'] = self.calculate_angle(
            keypoints[self.keypoint_indices['right_hip']],
            keypoints[self.keypoint_indices['right_knee']],
            keypoints[self.keypoint_indices['right_ankle']]
        )
        
        # Left knee angle
        angles['left_knee'] = self.calculate_angle(
            keypoints[self.keypoint_indices['left_hip']],
            keypoints[self.keypoint_indices['left_knee']],
            keypoints[self.keypoint_indices['left_ankle']]
        )
        
        # Hip angle (torso bend)
        angles['hip'] = self.calculate_angle(
            keypoints[self.keypoint_indices['left_shoulder']],
            keypoints[self.keypoint_indices['left_hip']],
            keypoints[self.keypoint_indices['left_knee']]
        )
        
        return angles
    
    def extract_distances(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Extract key distances between body parts
        
        Args:
            keypoints: Array of shape (17, 2)
            
        Returns:
            Dictionary of distance names and values
        """
        distances = {}
        
        # Hand to hip distances
        distances['right_hand_to_hip'] = self.calculate_distance(
            keypoints[self.keypoint_indices['right_wrist']],
            keypoints[self.keypoint_indices['right_hip']]
        )
        
        distances['left_hand_to_hip'] = self.calculate_distance(
            keypoints[self.keypoint_indices['left_wrist']],
            keypoints[self.keypoint_indices['left_hip']]
        )
        
        # Shoulder width
        distances['shoulder_width'] = self.calculate_distance(
            keypoints[self.keypoint_indices['left_shoulder']],
            keypoints[self.keypoint_indices['right_shoulder']]
        )
        
        # Hip width
        distances['hip_width'] = self.calculate_distance(
            keypoints[self.keypoint_indices['left_hip']],
            keypoints[self.keypoint_indices['right_hip']]
        )
        
        # Feet distance (stance width)
        distances['stance_width'] = self.calculate_distance(
            keypoints[self.keypoint_indices['left_ankle']],
            keypoints[self.keypoint_indices['right_ankle']]
        )
        
        return distances
    
    def extract_features_from_frame(self, pose_data: Dict) -> np.ndarray:
        """
        Extract all features from a single frame's pose data
        
        Args:
            pose_data: Dictionary with 'keypoints' and 'scores'
            
        Returns:
            Feature vector as numpy array
        """
        keypoints = pose_data['keypoints']
        scores = pose_data['scores']
        
        # Extract angles
        angles = self.extract_body_angles(keypoints)
        
        # Extract distances
        distances = self.extract_distances(keypoints)
        
        # Normalize distances by shoulder width for scale invariance
        shoulder_width = distances['shoulder_width']
        if shoulder_width > 0:
            for key in distances:
                if key != 'shoulder_width':
                    distances[key] = distances[key] / shoulder_width
        
        # Combine all features into a vector
        feature_vector = []
        
        # Add angles
        for angle_name in sorted(angles.keys()):
            feature_vector.append(angles[angle_name])
        
        # Add distances
        for dist_name in sorted(distances.keys()):
            feature_vector.append(distances[dist_name])
        
        # Add average confidence score
        feature_vector.append(np.mean(scores))
        
        return np.array(feature_vector)
    
    def detect_shot_moment(self, pose_sequence: List[Dict]) -> int:
        """
        Detect the exact frame where shot is played using angle change + hand velocity
        
        Args:
            pose_sequence: List of pose data dictionaries from all frames
            
        Returns:
            Index of the frame with the shot moment
        """
        if len(pose_sequence) < 3:
            return len(pose_sequence) // 2  # Return middle frame if too few frames
        
        scores = []
        
        for i in range(1, len(pose_sequence) - 1):
            prev_keypoints = pose_sequence[i - 1]['keypoints']
            curr_keypoints = pose_sequence[i]['keypoints']
            next_keypoints = pose_sequence[i + 1]['keypoints']
            
            # Calculate angle changes (focusing on right arm for batting)
            prev_angles = self.extract_body_angles(prev_keypoints)
            curr_angles = self.extract_body_angles(curr_keypoints)
            next_angles = self.extract_body_angles(next_keypoints)
            
            # Total angle change
            angle_change = 0
            for key in ['right_elbow', 'right_shoulder', 'left_elbow', 'left_shoulder']:
                angle_change += abs(curr_angles[key] - prev_angles[key])
                angle_change += abs(next_angles[key] - curr_angles[key])
            
            # Calculate hand velocity (right hand movement)
            right_wrist_idx = self.keypoint_indices['right_wrist']
            hand_velocity = np.linalg.norm(
                curr_keypoints[right_wrist_idx] - prev_keypoints[right_wrist_idx]
            )
            
            # Combined score
            score = self.alpha * angle_change + self.beta * hand_velocity
            scores.append(score)
        
        # Find frame with maximum score
        shot_frame_idx = np.argmax(scores) + 1  # +1 because we started from index 1
        
        return shot_frame_idx
    
    def extract_features_from_video(self, pose_sequence: List[Dict]) -> np.ndarray:
        """
        Extract features from entire video by detecting shot moment
        
        Args:
            pose_sequence: List of pose data from all frames
            
        Returns:
            Feature vector from the shot moment frame
        """
        # Detect shot moment
        shot_frame_idx = self.detect_shot_moment(pose_sequence)
        
        print(f"Shot moment detected at frame {shot_frame_idx}/{len(pose_sequence)}")
        
        # Extract features from that frame
        features = self.extract_features_from_frame(pose_sequence[shot_frame_idx])
        
        return features