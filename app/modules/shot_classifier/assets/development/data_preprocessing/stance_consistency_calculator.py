"""
Stance Consistency Calculator
Uses cosine similarity to measure batting stance consistency
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import normalize


class StanceConsistencyCalculator:
    """Calculate stance consistency using cosine similarity"""
    
    def __init__(self):
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Critical body parts for stance consistency
        self.critical_keypoints = [
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints to be scale and translation invariant
        
        Args:
            keypoints: Array of shape (17, 2) containing (x, y) coordinates
            
        Returns:
            Normalized keypoints
        """
        # Calculate center (mid-point between hips)
        left_hip = keypoints[self.keypoint_indices['left_hip']]
        right_hip = keypoints[self.keypoint_indices['right_hip']]
        center = (left_hip + right_hip) / 2
        
        # Translate to center
        centered = keypoints - center
        
        # Calculate scale (shoulder width)
        left_shoulder = centered[self.keypoint_indices['left_shoulder']]
        right_shoulder = centered[self.keypoint_indices['right_shoulder']]
        scale = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Normalize by scale
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered
        
        return normalized
    
    def extract_feature_vector(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from keypoints for consistency analysis
        
        Args:
            keypoints: Normalized keypoints array
            
        Returns:
            Feature vector containing angles and relative positions
        """
        features = []
        
        # Extract critical keypoint coordinates
        for keypoint_name in self.critical_keypoints:
            idx = self.keypoint_indices[keypoint_name]
            features.extend(keypoints[idx])  # Add x, y coordinates
        
        # Calculate key angles
        angles = self._calculate_stance_angles(keypoints)
        features.extend(angles)
        
        # Calculate relative distances
        distances = self._calculate_relative_distances(keypoints)
        features.extend(distances)
        
        return np.array(features)
    
    def _calculate_stance_angles(self, keypoints: np.ndarray) -> List[float]:
        """Calculate key angles in batting stance"""
        angles = []
        
        # Right elbow angle
        right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
        right_elbow = keypoints[self.keypoint_indices['right_elbow']]
        right_wrist = keypoints[self.keypoint_indices['right_wrist']]
        angles.append(self._angle_between_points(right_shoulder, right_elbow, right_wrist))
        
        # Left elbow angle
        left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
        left_elbow = keypoints[self.keypoint_indices['left_elbow']]
        left_wrist = keypoints[self.keypoint_indices['left_wrist']]
        angles.append(self._angle_between_points(left_shoulder, left_elbow, left_wrist))
        
        # Right knee angle
        right_hip = keypoints[self.keypoint_indices['right_hip']]
        right_knee = keypoints[self.keypoint_indices['right_knee']]
        right_ankle = keypoints[self.keypoint_indices['right_ankle']]
        angles.append(self._angle_between_points(right_hip, right_knee, right_ankle))
        
        # Left knee angle
        left_hip = keypoints[self.keypoint_indices['left_hip']]
        left_knee = keypoints[self.keypoint_indices['left_knee']]
        left_ankle = keypoints[self.keypoint_indices['left_ankle']]
        angles.append(self._angle_between_points(left_hip, left_knee, left_ankle))
        
        # Torso angle (left side)
        angles.append(self._angle_between_points(left_shoulder, left_hip, left_knee))
        
        # Torso angle (right side)
        angles.append(self._angle_between_points(right_shoulder, right_hip, right_knee))
        
        return angles
    
    def _calculate_relative_distances(self, keypoints: np.ndarray) -> List[float]:
        """Calculate relative distances between key points"""
        distances = []
        
        # Shoulder width (already normalized, but include for completeness)
        left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
        right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
        distances.append(np.linalg.norm(left_shoulder - right_shoulder))
        
        # Hip width
        left_hip = keypoints[self.keypoint_indices['left_hip']]
        right_hip = keypoints[self.keypoint_indices['right_hip']]
        distances.append(np.linalg.norm(left_hip - right_hip))
        
        # Stance width (feet distance)
        left_ankle = keypoints[self.keypoint_indices['left_ankle']]
        right_ankle = keypoints[self.keypoint_indices['right_ankle']]
        distances.append(np.linalg.norm(left_ankle - right_ankle))
        
        # Hand positions relative to body
        left_wrist = keypoints[self.keypoint_indices['left_wrist']]
        right_wrist = keypoints[self.keypoint_indices['right_wrist']]
        distances.append(np.linalg.norm(left_wrist - left_hip))
        distances.append(np.linalg.norm(right_wrist - right_hip))
        
        return distances
    
    def _angle_between_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_reference_stance(self, stance_keypoints_list: List[np.ndarray]) -> np.ndarray:
        """
        Calculate reference stance as average of all stances
        
        Args:
            stance_keypoints_list: List of keypoint arrays from multiple videos
            
        Returns:
            Reference stance feature vector
        """
        print(f"Calculating reference stance from {len(stance_keypoints_list)} stances...")
        
        # Normalize all keypoints
        normalized_keypoints = [self.normalize_keypoints(kp) for kp in stance_keypoints_list]
        
        # Extract feature vectors
        feature_vectors = [self.extract_feature_vector(kp) for kp in normalized_keypoints]
        
        # Calculate average (reference stance)
        reference_vector = np.mean(feature_vectors, axis=0)
        
        # Normalize reference vector for cosine similarity
        reference_vector = normalize(reference_vector.reshape(1, -1))[0]
        
        return reference_vector
    
    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1, vector2: Feature vectors
            
        Returns:
            Cosine similarity (0 to 1, where 1 is identical)
        """
        # Normalize vectors
        v1_normalized = normalize(vector1.reshape(1, -1))[0]
        v2_normalized = normalize(vector2.reshape(1, -1))[0]
        
        # Calculate cosine similarity
        similarity = np.dot(v1_normalized, v2_normalized)
        
        # Ensure result is in [0, 1] range
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Convert to percentage (0-1 scale)
        return (similarity + 1) / 2  # Map from [-1, 1] to [0, 1]
    
    def analyze_consistency(self, stance_keypoints_list: List[np.ndarray]) -> Dict:
        """
        Analyze stance consistency across multiple videos
        
        Args:
            stance_keypoints_list: List of keypoint arrays (one per video)
            
        Returns:
            Dictionary containing consistency analysis
        """
        if len(stance_keypoints_list) < 2:
            raise ValueError("Need at least 2 stances for consistency analysis")
        
        # Normalize all keypoints
        normalized_keypoints = [self.normalize_keypoints(kp) for kp in stance_keypoints_list]
        
        # Extract feature vectors
        feature_vectors = [self.extract_feature_vector(kp) for kp in normalized_keypoints]
        
        # Calculate reference stance
        reference_vector = np.mean(feature_vectors, axis=0)
        reference_vector = normalize(reference_vector.reshape(1, -1))[0]
        
        # Calculate similarity of each stance to reference
        individual_scores = []
        for i, feature_vector in enumerate(feature_vectors):
            similarity = self.calculate_cosine_similarity(feature_vector, reference_vector)
            individual_scores.append({
                'video_index': i + 1,
                'consistency_score': float(similarity * 100)  # Convert to percentage
            })
        
        # Calculate overall consistency
        overall_consistency = np.mean([score['consistency_score'] for score in individual_scores])
        
        # Calculate standard deviation (lower = more consistent)
        consistency_std = np.std([score['consistency_score'] for score in individual_scores])
        
        # Calculate pairwise similarities for detailed analysis
        pairwise_similarities = []
        for i in range(len(feature_vectors)):
            for j in range(i + 1, len(feature_vectors)):
                sim = self.calculate_cosine_similarity(feature_vectors[i], feature_vectors[j])
                pairwise_similarities.append({
                    'video_1': i + 1,
                    'video_2': j + 1,
                    'similarity': float(sim * 100)
                })
        
        # Identify most consistent and least consistent stances
        sorted_scores = sorted(individual_scores, key=lambda x: x['consistency_score'], reverse=True)
        
        result = {
            'overall_consistency': float(overall_consistency),
            'consistency_std': float(consistency_std),
            'individual_scores': individual_scores,
            'most_consistent': sorted_scores[0],
            'least_consistent': sorted_scores[-1],
            'pairwise_similarities': pairwise_similarities,
            'total_videos': len(stance_keypoints_list)
        }
        
        return result