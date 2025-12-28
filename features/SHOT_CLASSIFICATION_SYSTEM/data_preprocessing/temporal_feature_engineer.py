"""
Temporal Feature Engineer - Research Ready with YOLO Integration
Extracts temporal, motion-aware, and explainable pose features
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.ball_detector import BallBatDetector

class TemporalFeatureEngineer:
    """
    Extract temporal windowed features from pose sequences
     Uses YOLO for accurate ball-bat contact detection
    """
    
    def __init__(self):
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        

        # Initialize YOLO ball-bat detector
        self.ball_detector = BallBatDetector()
        print("✓ YOLO ball-bat detector initialized")

        # Critical angles for cricket shots (with expected ranges)
        self.angle_definitions = {
            'front_elbow': {
                'points': ['right_shoulder', 'right_elbow', 'right_wrist'],
                'expected_range': {'drive': (160, 175), 'cut': (90, 120), 'pull': (80, 110)}
            },
            'back_elbow': {
                'points': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'expected_range': {'drive': (140, 165), 'cut': (100, 130), 'pull': (70, 100)}
            },
            'front_knee': {
                'points': ['right_hip', 'right_knee', 'right_ankle'],
                'expected_range': {'drive': (160, 180), 'cut': (140, 170), 'pull': (90, 130)}
            },
            'back_knee': {
                'points': ['left_hip', 'left_knee', 'left_ankle'],
                'expected_range': {'drive': (155, 175), 'cut': (130, 160), 'pull': (100, 140)}
            },
            'torso_bend': {
                'points': ['left_shoulder', 'left_hip', 'left_knee'],
                'expected_range': {'drive': (160, 180), 'cut': (140, 170), 'pull': (120, 150)}
            },
            'shoulder_rotation': {
                'points': ['left_shoulder', 'right_shoulder', 'right_hip'],
                'expected_range': {'drive': (70, 90), 'cut': (45, 75), 'pull': (30, 60)}
            }
        }
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def detect_shot_window_with_yolo(self, pose_sequence: List[Dict], 
                                    frames: List[np.ndarray]) -> Tuple[int, int, int, Dict]:
        """
        Detect shot window using YOLO ball-bat detection
        
        Returns:
            (pre_idx, contact_idx, follow_idx, contact_metadata)
        """
        print("🎯 Using YOLO for accurate ball-bat contact detection...")
        
        contact_idx, contact_metadata = self.ball_detector.detect_contact_frame(frames, pose_sequence)
        
        pre_idx = max(0, contact_idx - 4)
        follow_idx = min(len(pose_sequence) - 1, contact_idx + 3)
        
        detection_method = contact_metadata.get('detection_method', 'unknown')
        print(f"✓ Contact detected at frame {contact_idx} (method: {detection_method})")
        
        return pre_idx, contact_idx, follow_idx, contact_metadata

    def detect_shot_window_fallback(self, pose_sequence: List[Dict]) -> Tuple[int, int, int]:
        """
        Fallback method using hand velocity (when YOLO fails or frames unavailable)
        Detect shot execution window with 3 phases:
        - Pre-shot (stance)
        - Contact (shot moment)
        - Follow-through
        
        Returns: (pre_frame, contact_frame, follow_frame)
        """
        if len(pose_sequence) < 5:
            mid = len(pose_sequence) // 2
            return max(0, mid-2), mid, min(len(pose_sequence)-1, mid+2)
        
        movement_scores = []
        
        for i in range(1, len(pose_sequence) - 1):
            prev_kp = pose_sequence[i-1]['keypoints']
            curr_kp = pose_sequence[i]['keypoints']
            next_kp = pose_sequence[i+1]['keypoints']
            
            # Track hand and shoulder movement
            hand_indices = [self.keypoint_indices['right_wrist'], 
                          self.keypoint_indices['left_wrist']]
            
            movement = 0
            for idx in hand_indices:
                movement += np.linalg.norm(curr_kp[idx] - prev_kp[idx])
                movement += np.linalg.norm(next_kp[idx] - curr_kp[idx])
            
            movement_scores.append(movement)
        
        # Contact point = maximum movement
        contact_idx = np.argmax(movement_scores) + 1
        
        # Pre-shot = 3-5 frames before contact
        pre_idx = max(0, contact_idx - 4)
        
        # Follow-through = 2-4 frames after contact
        follow_idx = min(len(pose_sequence) - 1, contact_idx + 3)
        
        return pre_idx, contact_idx, follow_idx
    
    def extract_angle_features(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Extract all critical angles with their names for explainability"""
        angles = {}
        
        for angle_name, definition in self.angle_definitions.items():
            point_names = definition['points']
            points = [keypoints[self.keypoint_indices[name]] for name in point_names]
            angle_value = self.calculate_angle(points[0], points[1], points[2])
            angles[angle_name] = angle_value
        
        return angles
    
    def extract_velocity_features(self, prev_kp: np.ndarray, curr_kp: np.ndarray) -> Dict[str, float]:
        """Extract velocity of key body parts"""
        velocities = {}
        
        critical_points = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow',
                          'right_shoulder', 'left_shoulder']
        
        for point_name in critical_points:
            idx = self.keypoint_indices[point_name]
            velocity = np.linalg.norm(curr_kp[idx] - prev_kp[idx])
            velocities[f'{point_name}_velocity'] = velocity
        
        return velocities
    
    def extract_positional_features(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Extract relative positions (normalized)"""
        # Center point (mid-hip)
        left_hip = keypoints[self.keypoint_indices['left_hip']]
        right_hip = keypoints[self.keypoint_indices['right_hip']]
        center = (left_hip + right_hip) / 2
        
        # Normalize by shoulder width
        left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
        right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_width < 1e-6:
            shoulder_width = 1.0
        
        positions = {}
        
        # Key positions relative to center
        for point_name in ['right_wrist', 'left_wrist', 'right_foot', 'left_foot']:
            if point_name == 'right_foot':
                idx = self.keypoint_indices['right_ankle']
            elif point_name == 'left_foot':
                idx = self.keypoint_indices['left_ankle']
            else:
                idx = self.keypoint_indices[point_name]
            
            relative_pos = keypoints[idx] - center
            normalized_pos = relative_pos / shoulder_width
            
            positions[f'{point_name}_x'] = normalized_pos[0]
            positions[f'{point_name}_y'] = normalized_pos[1]
        
        return positions
    
    def extract_temporal_features(self, pose_sequence: List[Dict], 
                                 frames: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Extract complete temporal feature set with metadata
        
        Args:
            pose_sequence: List of pose data dictionaries
            frames: Optional list of video frames (for YOLO detection)
        Returns:
            (feature_vector, metadata_dict)
        """
        # Detect 3-phase window
        if frames is not None and len(frames) == len(pose_sequence):
            # Use YOLO for accurate ball-bat contact detection
            pre_idx, contact_idx, follow_idx, contact_metadata = self.detect_shot_window_with_yolo(
                pose_sequence, frames
            )
        else:
            # Fallback to hand velocity method
            print("⚠️  Frames not available, using fallback hand velocity detection")
            pre_idx, contact_idx, follow_idx = self.detect_shot_window_fallback(pose_sequence)
            contact_metadata = {
                'detection_method': 'fallback_velocity',
                'ball_detected': False,
                'bat_detected': False
            }
        
        pre_pose = pose_sequence[pre_idx]
        contact_pose = pose_sequence[contact_idx]
        follow_pose = pose_sequence[follow_idx]
        
        features = []
        metadata = {
            'pre_frame': pre_idx,
            'contact_frame': contact_idx,
            'follow_frame': follow_idx,
            'contact_detection': contact_metadata,  # ADDED: YOLO detection info
            'angles': {},
            'velocities': {},
            'positions': {}
        }
        
        # === CONTACT FRAME FEATURES (Main) ===
        contact_angles = self.extract_angle_features(contact_pose['keypoints'])
        for name, value in contact_angles.items():
            features.append(value)
            metadata['angles'][f'contact_{name}'] = value
        
        contact_positions = self.extract_positional_features(contact_pose['keypoints'])
        for name, value in contact_positions.items():
            features.append(value)
            metadata['positions'][f'contact_{name}'] = value
        
        # === PRE-SHOT TO CONTACT DYNAMICS ===
        pre_to_contact_vel = self.extract_velocity_features(
            pre_pose['keypoints'], 
            contact_pose['keypoints']
        )
        for name, value in pre_to_contact_vel.items():
            features.append(value)
            metadata['velocities'][f'pre_to_contact_{name}'] = value
        
        # === CONTACT TO FOLLOW-THROUGH DYNAMICS ===
        contact_to_follow_vel = self.extract_velocity_features(
            contact_pose['keypoints'],
            follow_pose['keypoints']
        )
        for name, value in contact_to_follow_vel.items():
            features.append(value)
            metadata['velocities'][f'contact_to_follow_{name}'] = value
        
        # === ANGULAR VELOCITY (Change in angles) ===
        pre_angles = self.extract_angle_features(pre_pose['keypoints'])
        follow_angles = self.extract_angle_features(follow_pose['keypoints'])
        
        for angle_name in contact_angles.keys():
            angular_velocity = follow_angles[angle_name] - pre_angles[angle_name]
            features.append(angular_velocity)
            metadata['velocities'][f'{angle_name}_angular_change'] = angular_velocity
        
        # === CONFIDENCE SCORES ===
        avg_confidence = np.mean(contact_pose['scores'])
        features.append(avg_confidence)
        metadata['avg_confidence'] = avg_confidence
        
        return np.array(features), metadata
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability"""
        names = []
        
        # Angles at contact
        for angle_name in self.angle_definitions.keys():
            names.append(f'contact_{angle_name}')
        
        # Positions at contact
        for point in ['right_wrist', 'left_wrist', 'right_foot', 'left_foot']:
            names.extend([f'contact_{point}_x', f'contact_{point}_y'])
        
        # Velocities
        critical_points = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow',
                          'right_shoulder', 'left_shoulder']
        for point in critical_points:
            names.append(f'pre_to_contact_{point}_velocity')
        
        for point in critical_points:
            names.append(f'contact_to_follow_{point}_velocity')
        
        # Angular changes
        for angle_name in self.angle_definitions.keys():
            names.append(f'{angle_name}_angular_change')
        
        # Confidence
        names.append('avg_confidence')
        
        return names