"""
Visual Feedback Generator
Creates annotated images showing pose and errors
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import base64


class VisualFeedbackGenerator:
    """Generate visual feedback with pose overlay and error highlighting"""
    
    def __init__(self):
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Skeleton connections
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Joint ID to keypoint mapping
        self.joint_to_keypoints = {
            'front_elbow': ['right_shoulder', 'right_elbow', 'right_wrist'],
            'back_elbow': ['left_shoulder', 'left_elbow', 'left_wrist'],
            'front_knee': ['right_hip', 'right_knee', 'right_ankle'],
            'back_knee': ['left_hip', 'left_knee', 'left_ankle'],
            'torso_bend': ['left_shoulder', 'left_hip', 'left_knee'],
            'shoulder_rotation': ['left_shoulder', 'right_shoulder', 'right_hip']
        }
    
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                     scores: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """Draw pose skeleton on image"""
        img = image.copy()
        
        # Draw keypoints
        for idx, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(img, (x, y), 4, color, -1)
        
        # Draw skeleton connections
        for start_idx, end_idx in self.skeleton:
            if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
                start_pt = tuple(keypoints[start_idx].astype(int))
                end_pt = tuple(keypoints[end_idx].astype(int))
                cv2.line(img, start_pt, end_pt, color, thickness)
        
        return img
    
    def highlight_error_joints(self, image: np.ndarray, keypoints: np.ndarray,
                              scores: np.ndarray, mistakes: List[Dict]) -> np.ndarray:
        """Highlight joints with errors"""
        img = image.copy()
        
        for mistake in mistakes:
            joint_id = mistake['joint_id']
            severity = mistake['severity']
            
            # Get color based on severity
            if severity == 'critical':
                color = (0, 0, 255)  # Red
                radius = 12
            elif severity == 'major':
                color = (0, 165, 255)  # Orange
                radius = 10
            else:
                color = (0, 255, 255)  # Yellow
                radius = 8
            
            # Get keypoints involved in this joint
            if joint_id in self.joint_to_keypoints:
                keypoint_names = self.joint_to_keypoints[joint_id]
                
                for kpt_name in keypoint_names:
                    idx = self.keypoint_indices[kpt_name]
                    if scores[idx] > 0.3:
                        x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                        
                        # Draw pulsing circle
                        cv2.circle(img, (x, y), radius, color, 3)
                        cv2.circle(img, (x, y), radius + 5, color, 1)
        
        return img
    
    def add_text_annotations(self, image: np.ndarray, mistakes: List[Dict],
                           keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Add text annotations for mistakes"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Add semi-transparent overlay for text background
        overlay = img.copy()
        
        y_offset = 30
        for i, mistake in enumerate(mistakes[:3]):  # Top 3 mistakes
            joint_id = mistake['joint_id']
            
            # Get position near the joint
            if joint_id in self.joint_to_keypoints:
                keypoint_names = self.joint_to_keypoints[joint_id]
                center_kpt_name = keypoint_names[1]  # Middle keypoint
                idx = self.keypoint_indices[center_kpt_name]
                
                if scores[idx] > 0.3:
                    x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                    
                    # Text to display
                    text = f"{mistake['joint']}: {mistake['actual_angle']}"
                    
                    # Background rectangle
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(overlay, 
                                (x - 5, y - text_size[1] - 10),
                                (x + text_size[0] + 5, y - 5),
                                (0, 0, 0), -1)
                    
                    # Text
                    cv2.putText(overlay, text, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend overlay
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        return img
    
    def add_mistake_legend(self, image: np.ndarray, mistakes: List[Dict]) -> np.ndarray:
        """Add legend showing all mistakes"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Create legend area
        legend_height = min(200, 30 + len(mistakes) * 25)
        legend_width = 400
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (legend_width, legend_height), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # Title
        cv2.putText(img, "Detected Issues:", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # List mistakes
        y_pos = 60
        for mistake in mistakes[:5]:  # Top 5
            severity = mistake['severity']
            
            # Severity indicator
            if severity == 'critical':
                color = (0, 0, 255)
                icon = "●"
            elif severity == 'major':
                color = (0, 165, 255)
                icon = "●"
            else:
                color = (0, 255, 255)
                icon = "○"
            
            # Draw icon
            cv2.putText(img, icon, (25, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw text
            text = f"{mistake['joint']}: {mistake['deviation']}"
            cv2.putText(img, text, (45, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_pos += 25
        
        return img
    
    def generate_feedback_images(self, frame: np.ndarray, keypoints: np.ndarray,
                                scores: np.ndarray, mistakes: List[Dict]) -> Dict[str, str]:
        """
        Generate all feedback images
        
        Returns:
            Dictionary with base64-encoded images
        """
        # 1. Original frame with pose overlay
        pose_overlay = self.draw_skeleton(frame, keypoints, scores, 
                                         color=(0, 255, 0), thickness=2)
        
        # 2. Error overlay (highlighting problem joints)
        error_overlay = pose_overlay.copy()
        error_overlay = self.highlight_error_joints(error_overlay, keypoints, 
                                                    scores, mistakes)
        
        # 3. Full annotated image (with text and legend)
        annotated = error_overlay.copy()
        annotated = self.add_text_annotations(annotated, mistakes, keypoints, scores)
        annotated = self.add_mistake_legend(annotated, mistakes)
        
        # Convert to base64
        def encode_image(img):
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return base64.b64encode(buffer).decode('utf-8')
        
        return {
            'selected_frame': encode_image(frame),
            'pose_overlay': encode_image(pose_overlay),
            'error_overlay': encode_image(error_overlay),
            'annotated_frame': encode_image(annotated)
        }