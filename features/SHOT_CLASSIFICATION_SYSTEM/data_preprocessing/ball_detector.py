"""
Ball Detector with Pose-Based Virtual Bat Detection
Works even when YOLO can't detect cricket bat
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import os


class BallBatDetector:
    """
    Detect cricket ball using YOLO
    Use pose estimation for "virtual bat" location (hands)
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to custom YOLO model (if trained on cricket)
                       If None, uses YOLOv8 with sports ball detection
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print("✓ Loaded custom cricket detection model")
        else:
            # Use pretrained YOLOv8 (detects sports balls)
            self.model = YOLO('yolov8n.pt')
            print("✓ Loaded YOLOv8 pretrained model")
    
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect ball and bat in single frame
        
        Returns:
            Dictionary with ball_bbox, bat_bbox, and confidence scores
        """
        results = self.model(frame, verbose=False)[0]
        
        ball_bbox = None
        ball_conf = 0
       
        
        # Parse detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  
            
            # Ball detection (class 32 = sports ball, 37 = sports equipment)
            if cls in [32, 37]:
                if conf > ball_conf:
                    ball_bbox = bbox
                    ball_conf = conf
            
            
        
        return {
            'ball_bbox': ball_bbox,
            'ball_confidence': ball_conf
        }
    
    def get_virtual_bat_bbox(self, pose_data: Dict) -> Optional[np.ndarray]:
        """
        Create virtual bat bounding box from hand positions
        Since YOLO can't detect cricket bat, we estimate it from pose
        """
        keypoints = pose_data['keypoints']
        scores = pose_data['scores']
        
        # Right hand (front) - index 10
        # Left hand (back) - index 9
        right_wrist = keypoints[10]
        left_wrist = keypoints[9]
        right_score = scores[10]
        left_score = scores[9]
        
        if right_score < 0.3 or left_score < 0.3:
            return None
        
        # Bat extends from hands with some width
        # Estimate bat dimensions: ~90cm long, ~10cm wide
        hand_center = (right_wrist + left_wrist) / 2
        hand_distance = np.linalg.norm(right_wrist - left_wrist)
        
        # Bat length proportional to hand separation
        bat_length = hand_distance * 1.5
        bat_width = hand_distance * 0.3
        
        # Create bbox around hand region
        x_min = hand_center[0] - bat_width
        y_min = hand_center[1] - bat_length / 2
        x_max = hand_center[0] + bat_width
        y_max = hand_center[1] + bat_length / 2
        
        return np.array([x_min, y_min, x_max, y_max])
    

    def detect_contact_frame(self, frames: List[np.ndarray], 
                           pose_sequence: List[Dict]) -> Tuple[int, Dict]:
        """
        Find exact frame where ball contacts bat (virtual or real)
        
        Args:
            frames: List of video frames
            pose_sequence: Corresponding pose data
            
        Returns:
            (contact_frame_idx, contact_metadata)
        """
        print("🎯 Detecting ball-bat contact (using pose-based virtual bat)...")
        
        contact_scores = []
        detections_log = []
        ball_detected_count = 0

        for i, frame in enumerate(frames):
            detection = self.detect_objects(frame)

            # Add virtual bat from pose
            if i < len(pose_sequence):
                virtual_bat = self.get_virtual_bat_bbox(pose_sequence[i])
                detection['virtual_bat_bbox'] = virtual_bat
            else:
                detection['virtual_bat_bbox'] = None

            detections_log.append(detection)
            
            if detection['ball_bbox'] is not None:
                ball_detected_count += 1
            
            # Calculate contact score
            contact_score = self._calculate_contact_score(
                detection, 
                pose_sequence[i] if i < len(pose_sequence) else None
            )
            contact_scores.append(contact_score)
        
        # Find frame with highest contact score
        if max(contact_scores) > 0.3:  # Lowered threshold for ball-only detection
            contact_idx = np.argmax(contact_scores)
            detection_method = 'yolo_ball_pose_bat'
        else:
            # Fallback: Use hand velocity (old method)
            contact_idx = self._fallback_detection(pose_sequence)
            detection_method = 'fallback_velocity'
        
        ball_detection_rate = (ball_detected_count / len(frames)) * 100
        
        contact_metadata = {
            'contact_frame': contact_idx,
            'contact_score': float(contact_scores[contact_idx]) if contact_scores else 0,
            'ball_detected': detections_log[contact_idx]['ball_bbox'] is not None,
            'bat_detected': detections_log[contact_idx]['virtual_bat_bbox'] is not None,
            'detection_method': detection_method,
            'ball_detection_rate': ball_detection_rate
        }
        
        print(f"✓ Contact at frame {contact_idx}/{len(frames)} "
              f"(method: {detection_method}, ball detected in {ball_detection_rate:.1f}% of frames)")
        
        return contact_idx, contact_metadata
    
    def _calculate_contact_score(self, detection: Dict, pose_data: Optional[Dict]) -> float:
        """
        Calculate likelihood of ball-bat contact
        Works with ball-only detection + pose-based virtual bat
        """
        ball_bbox = detection.get("ball_bbox")
        virtual_bat_bbox = detection.get("virtual_bat_bbox")
        
        if ball_bbox is None:
            return 0.0

        score = 0.0

        # 1. Ball detection confidence (base score)
        ball_conf = detection.get("ball_confidence", 0.0)
        score += ball_conf * 0.4
        
        # 2. Ball near hands (from pose)
        if pose_data is not None:
            hand_proximity = self._ball_near_hands(ball_bbox, pose_data)
            score += hand_proximity * 0.35  # High weight
        
        # 3. Ball-virtual bat interaction
        if virtual_bat_bbox is not None:
            # Distance between ball and virtual bat
            distance = self._bbox_distance(ball_bbox, virtual_bat_bbox)
            proximity_score = np.exp(-distance / 80.0)  # Exponential decay
            score += proximity_score * 0.35
            
            # IoU (if they overlap)
            iou = self._calculate_iou(ball_bbox, virtual_bat_bbox)
            score += iou * 0.5  # Bonus for overlap
        
        return float(np.clip(score, 0.0, 1.0))

    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _bbox_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate center-to-center distance"""
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        return np.linalg.norm(center1 - center2)
    
    def _ball_near_hands(self, ball_bbox: np.ndarray, pose_data: Dict) -> float:
        """Check if ball is near hand positions"""
        keypoints = pose_data['keypoints']
        scores = pose_data['scores']
        
        # Right hand (index 10), Left hand (index 9)
        right_wrist = keypoints[10]
        left_wrist = keypoints[9]
        
        if scores[10] < 0.3 and scores[9] < 0.3:
            return 0.0
        
        ball_center = np.array([
            (ball_bbox[0] + ball_bbox[2]) / 2,
            (ball_bbox[1] + ball_bbox[3]) / 2
        ])
        
        # Distance to nearest hand
        distances = []
        if scores[10] > 0.3:
            distances.append(np.linalg.norm(ball_center - right_wrist))
        if scores[9] > 0.3:
            distances.append(np.linalg.norm(ball_center - left_wrist))
        
        if not distances:
            return 0.0
        
        min_distance = min(distances)
        
        # Closer = higher score (threshold = 150 pixels)
        return max(0, 1 - (min_distance / 150))
    
    def _fallback_detection(self, pose_sequence: List[Dict]) -> int:
        """Fallback to hand velocity"""
        if len(pose_sequence) < 3:
            return len(pose_sequence) // 2
        
        velocities = []
        for i in range(1, len(pose_sequence) - 1):
            prev_kp = pose_sequence[i-1]['keypoints']
            curr_kp = pose_sequence[i]['keypoints']
            
            
            velocity = np.linalg.norm(curr_kp[10] - prev_kp[10])
            velocities.append(velocity)
        
        return np.argmax(velocities) + 1