"""
Ball & Bat Detector using YOLOv8
Detects exact moment of ball-bat contact
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch


class BallBatDetector:
    """
    Detect cricket ball and bat using YOLO
    Find exact contact moment
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
        bat_bbox = None
        ball_conf = 0
        bat_conf = 0
        
        # Parse detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            # Ball detection (class 32 in COCO = sports ball)
            if cls == 32:
                if conf > ball_conf:
                    ball_bbox = bbox
                    ball_conf = conf
            
            # Bat detection (class 35 in COCO = baseball bat, similar to cricket bat)
            elif cls == 35 or cls == 37:  # 37 = tennis racket (similar shape)
                if conf > bat_conf:
                    bat_bbox = bbox
                    bat_conf = conf
        
        return {
            'ball_bbox': ball_bbox,
            'ball_confidence': ball_conf,
            'bat_bbox': bat_bbox,
            'bat_confidence': bat_conf
        }
    
    def detect_contact_frame(self, frames: List[np.ndarray], 
                           pose_sequence: List[Dict]) -> Tuple[int, Dict]:
        """
        Find exact frame where ball contacts bat
        
        Args:
            frames: List of video frames
            pose_sequence: Corresponding pose data
            
        Returns:
            (contact_frame_idx, contact_metadata)
        """
        print("Detecting ball-bat contact moment...")
        
        contact_scores = []
        detections_log = []
        
        for i, frame in enumerate(frames):
            detection = self.detect_objects(frame)
            detections_log.append(detection)
            
            # Calculate contact likelihood
            contact_score = self._calculate_contact_score(
                detection, 
                pose_sequence[i] if i < len(pose_sequence) else None
            )
            contact_scores.append(contact_score)
        
        # Find frame with highest contact score
        if max(contact_scores) > 0:
            contact_idx = np.argmax(contact_scores)
        else:
            # Fallback: Use hand velocity (old method)
            contact_idx = self._fallback_detection(pose_sequence)
        
        contact_metadata = {
            'contact_frame': contact_idx,
            'contact_score': float(contact_scores[contact_idx]) if contact_scores else 0,
            'ball_detected': detections_log[contact_idx]['ball_bbox'] is not None,
            'bat_detected': detections_log[contact_idx]['bat_bbox'] is not None,
            'detection_method': 'yolo' if max(contact_scores) > 0 else 'fallback'
        }
        
        print(f"✓ Contact detected at frame {contact_idx}/{len(frames)} "
              f"(method: {contact_metadata['detection_method']})")
        
        return contact_idx, contact_metadata
    
    def _calculate_contact_score(self, detection: Dict, pose_data: Optional[Dict]) -> float:
        """
        Calculate likelihood of ball-bat contact
        """
        ball_bbox = detection.get("ball_bbox")
        bat_bbox = detection.get("bat_bbox")

        score = 0.0

        # 1️⃣ Detection confidence (soft contribution)
        if ball_bbox is not None:
            score += detection.get("ball_confidence", 0.0) * 0.4

        if bat_bbox is not None:
            score += detection.get("bat_confidence", 0.0) * 0.3

        # 2️⃣ Spatial relationship (ONLY if both exist)
        if ball_bbox is not None and bat_bbox is not None:
            # Distance-based score
            distance = self._bbox_distance(ball_bbox, bat_bbox)
            score += np.exp(-distance / 80) * 0.3

            # IoU overlap
            iou = self._calculate_iou(ball_bbox, bat_bbox)
            score += iou * 0.5   # keep this SMALL, not 50

            # Proximity bonus
            proximity_score = max(0.0, 30.0 - distance) / 30.0
            score += proximity_score * 0.3

            # 3️⃣ Pose-based refinement
            if pose_data is not None:
                hand_score = self._ball_near_hands(ball_bbox, pose_data)
                score += hand_score * 0.3

        return float(score)

    
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
        """Calculate center-to-center distance between bboxes"""
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
        
        # Closer = higher score (max distance threshold = 200 pixels)
        return max(0, 1 - (min_distance / 200))
    
    def _fallback_detection(self, pose_sequence: List[Dict]) -> int:
        """Fallback to hand velocity method if YOLO fails"""
        if len(pose_sequence) < 3:
            return len(pose_sequence) // 2
        
        velocities = []
        for i in range(1, len(pose_sequence) - 1):
            prev_kp = pose_sequence[i-1]['keypoints']
            curr_kp = pose_sequence[i]['keypoints']
            
            # Right wrist velocity
            velocity = np.linalg.norm(curr_kp[10] - prev_kp[10])
            velocities.append(velocity)
        
        return np.argmax(velocities) + 1