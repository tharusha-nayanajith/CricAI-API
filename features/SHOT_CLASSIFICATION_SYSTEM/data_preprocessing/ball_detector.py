"""
Ball-Bat Detector - 3-Tier Detection System
Tier 1: Real bat + ball (YOLO)
Tier 2: Virtual bat + ball (YOLO ball + pose bat)
Tier 3: Fallback (hand velocity)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import os
from roboflow import Roboflow
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.kalman_ball_tracker import KalmanBallTracker

class BallBatDetector:
    """
    3-Tier cricket contact detection:
    1. Real bat-ball (YOLO detects both)
    2. Virtual bat-ball (YOLO ball + pose bat)
    3. Fallback (velocity + direction change)
    """
    
    def __init__(self, model_path: str = None, use_roboflow_ball: bool = True):
        """Initialize YOLO detector"""
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print("✓ Loaded custom cricket detection model")
        else:
            
            self.model = YOLO('yolov8n.pt')
            print("✓ Loaded YOLOv8 pretrained model")

        self.ball_tracker = KalmanBallTracker()
        # Roboflow ball detector
        self.rf_ball_model = None
        if use_roboflow_ball:
            print("🔑 Loading Roboflow API key...")
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if not api_key:
                raise RuntimeError("❌ ROBOFLOW_API_KEY not found in environment")

            print("📦 Initializing Roboflow client...")
            rf = Roboflow(api_key=api_key)
            try:
                print("📁 Loading project: cricket-ball-detection-cu0z5")
                project = rf.workspace().project("cricket-ball-detection-cu0z5")

                print("🔢 Loading model version: 1")
                self.rf_ball_model = project.version(1).model

            except Exception as e:
                raise RuntimeError(f"❌ Failed to load Roboflow model: {e}")

            print("✅ Roboflow ball detector loaded successfully")
            print(f"   • Project: {project.name}")
            print(f"   • Version: {project.version(1).version}")
            print(f"   • Type: {project.type}")

    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect ball and bat using YOLO + Roboflow
        Improved: Multiple class IDs for ball detection
        """
        
        
        ball_bbox = None
        ball_conf = 0.0

        # 1️⃣ Roboflow ball (PRIMARY)
        if self.rf_ball_model is not None:
            rf_result = self.detect_ball_roboflow(frame)
            if rf_result is not None:
                ball_bbox, ball_conf = rf_result

        # 2️⃣ YOLO bat detection
        results = self.model(frame, verbose=False)[0]
        bat_bbox = None
        bat_conf = 0.0

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  
            
            if cls in [35, 37, 39]:  # bat-like
                if conf > bat_conf:
                    bat_bbox = bbox
                    bat_conf = conf
            
            
        
        return {
            "ball_bbox": ball_bbox,
            "ball_confidence": ball_conf,
            "bat_bbox": bat_bbox,
            "bat_confidence": bat_conf
        }
    
    def get_virtual_bat_bbox(self, pose_data: Dict) -> Optional[np.ndarray]:
        """
        Create virtual bat from hand positions
        """
        keypoints = pose_data['keypoints']
        scores = pose_data['scores']
        
        
        right_wrist = keypoints[10]
        left_wrist = keypoints[9]
        right_elbow = keypoints[8]
        left_elbow = keypoints[7]
        
        # Need at least one hand detected
        if scores[10] < 0.3 and scores[9] < 0.3:
            return None
        
        # Use elbows to estimate bat direction if available
        if scores[10] > 0.3 and scores[8] > 0.3:
            # Right side (front)
            hand_center = right_wrist
            bat_direction = right_wrist - right_elbow
        elif scores[9] > 0.3 and scores[7] > 0.3:
            # Left side (back)
            hand_center = left_wrist
            bat_direction = left_wrist - left_elbow
        else:
            # Fallback: use both hands if available
            if scores[10] > 0.3 and scores[9] > 0.3:
                hand_center = (right_wrist + left_wrist) / 2
                bat_direction = right_wrist - left_wrist
            else:
                hand_center = right_wrist if scores[10] > 0.3 else left_wrist
                bat_direction = np.array([0, 50])  # Default: vertical
        
        # Normalize direction
        bat_length_vector = bat_direction / (np.linalg.norm(bat_direction) + 1e-6) * 100
        
        # Bat perpendicular width
        bat_width = 30
        
        # Create bbox 
        x_min = hand_center[0] - bat_width
        y_min = hand_center[1] - 50
        x_max = hand_center[0] + bat_width
        y_max = hand_center[1] + 50
        
        return np.array([x_min, y_min, x_max, y_max])
    

    def detect_contact_frame(self, frames: List[np.ndarray], 
                           pose_sequence: List[Dict]) -> Tuple[int, Dict]:
        """
        3-Tier detection system
        """
        print("🎯 3-Tier Ball-Bat Contact Detection")
        print("-" * 50)
        
        contact_scores_tier1 = []  # Real bat + ball
        contact_scores_tier2 = []  # Virtual bat + ball
        detections_log = []
        ball_detected_count = 0
        real_bat_detected_count = 0

        for i, frame in enumerate(frames):
            detection = self.detect_objects(frame)
            if detection['ball_bbox'] is not None:
                center = np.array([
                    (detection['ball_bbox'][0] + detection['ball_bbox'][2]) / 2,
                    (detection['ball_bbox'][1] + detection['ball_bbox'][3]) / 2
                ])
                self.ball_tracker.update(center)
            else:
                predicted = self.ball_tracker.predict()
                if predicted is not None:
                    detection['ball_bbox'] = np.array([
                        predicted[0] - 8,
                        predicted[1] - 8,
                        predicted[0] + 8,
                        predicted[1] + 8
                    ])
                    detection['ball_confidence'] = 0.15

            # Track detection rates
            if detection['ball_bbox'] is not None:
                ball_detected_count += 1
            if detection['bat_bbox'] is not None:
                real_bat_detected_count += 

            # Add virtual bat 
            if i < len(pose_sequence):
                detection['virtual_bat_bbox'] = self.get_virtual_bat_bbox(pose_sequence[i])
            else:
                detection['virtual_bat_bbox'] = None

            detections_log.append(detection)
            
            # Calculate scores for both tiers
            score_tier1 = self._calculate_tier1_score(detection, pose_sequence[i] if i < len(pose_sequence) else None)
            score_tier2 = self._calculate_tier2_score(detection, pose_sequence[i] if i < len(pose_sequence) else None)
            
            contact_scores_tier1.append(score_tier1)
            contact_scores_tier2.append(score_tier2)
        
        # Detection rates
        ball_rate = (ball_detected_count / len(frames)) * 100
        bat_rate = (real_bat_detected_count / len(frames)) * 100
        
        print(f"📊 Detection Rates:")
        print(f"   Ball: {ball_rate:.1f}% ({ball_detected_count}/{len(frames)} frames)")
        print(f"   Real Bat: {bat_rate:.1f}% ({real_bat_detected_count}/{len(frames)} frames)")
        
        # TIER 1: Real bat + ball (both detected by YOLO + Roboflow)
        max_tier1 = max(contact_scores_tier1)
        if max_tier1 > 0.5 and ball_rate > 40:  # Require decent ball detection
            contact_idx = np.argmax(contact_scores_tier1)
            method = 'tier1_real_bat_ball'
            print(f"✅ Tier 1: Real bat + ball detection (score: {max_tier1:.2f})")
        
        # TIER 2: Virtual bat + ball (ball from Roboflow, bat from pose)
        elif max(contact_scores_tier2) > 0.3 and ball_rate > 20:
            contact_idx = np.argmax(contact_scores_tier2)
            method = 'tier2_virtual_bat_ball'
            print(f"✅ Tier 2: Virtual bat + ball detection (score: {max(contact_scores_tier2):.2f})")
        
        # TIER 3: Fallback (acceleration + direction change)
        else:
            
            contact_idx = self._fallback_detection(pose_sequence)
            method = 'tier3_acceleration_direction_change'
            print(f"⚠️  Tier 3: Fallback to velocity (ball detection too low: {ball_rate:.1f}%)")
        
        contact_metadata = {
            'contact_frame': contact_idx,
            'detection_method': method,
            'ball_detected': detections_log[contact_idx]['ball_bbox'] is not None,
            'bat_detected': detections_log[contact_idx]['bat_bbox'] is not None,
            'virtual_bat_used': 'tier2' in method or 'tier3' in method,
            'ball_detection_rate': ball_rate,
            'bat_detection_rate': bat_rate,
            'tier1_score': float(contact_scores_tier1[contact_idx]),
            'tier2_score': float(contact_scores_tier2[contact_idx])
        }
        
        print(f"✓ Contact at frame {contact_idx}/{len(frames)}")
        print("-" * 50)
        
        return contact_idx, contact_metadata
    
    def _calculate_tier1_score(self, detection: Dict, pose_data: Optional[Dict]) -> float:
        """
        Tier 1: Real bat + ball (both from YOLO)
        """
        ball_bbox = detection.get("ball_bbox")
        bat_bbox = detection.get("bat_bbox")
        
        # Need both detected
        if ball_bbox is None or bat_bbox is None:
            return 0.0
        
        score = 0.0
        
        # 1. Detection confidences
        ball_conf = detection.get("ball_confidence", 0.0)
        bat_conf = detection.get("bat_confidence", 0.0)
        score += ball_conf * 0.3
        score += bat_conf * 0.3
        
        # 2. Spatial relationship
        distance = self._bbox_distance(ball_bbox, bat_bbox)
        proximity = np.exp(-distance / 80.0)
        score += proximity * 0.3
        
        # 3. IoU overlap
        iou = self._calculate_iou(ball_bbox, bat_bbox)
        score += iou * 0.2
        
        # 4. Pose consistency (ball near hands)
        if pose_data is not None:
            hand_score = self._ball_near_hands(ball_bbox, pose_data)
            score += hand_score * 0.2
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_tier2_score(self, detection: Dict, pose_data: Optional[Dict]) -> float:
        """
        Tier 2: Virtual bat + ball (ball from YOLO, bat from pose)
        """
        ball_bbox = detection.get("ball_bbox")
        virtual_bat_bbox = detection.get("virtual_bat_bbox")
        
        # Need ball detected
        if ball_bbox is None:
            return 0.0

        score = 0.0

        # 1. Ball confidence
        ball_conf = detection.get("ball_confidence", 0.0)
        score += ball_conf * 0.4
        
        # 2. Ball near hands (critical for virtual bat)
        if pose_data is not None:
            hand_proximity = self._ball_near_hands(ball_bbox, pose_data)
            score += hand_proximity * 0.4  # High weight
        
        # 3. Ball-virtual bat interaction
        if virtual_bat_bbox is not None:
            
            distance = self._bbox_distance(ball_bbox, virtual_bat_bbox)
            proximity = np.exp(-distance / 80.0)
            score += proximity * 0.3
            
            # IoU (if they overlap)
            iou = self._calculate_iou(ball_bbox, virtual_bat_bbox)
            score += iou * 0.3
        
        return float(np.clip(score, 0.0, 1.0))

    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU"""
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
        """Calculate distance"""
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        return np.linalg.norm(center1 - center2)
    
    def _ball_near_hands(self, ball_bbox: np.ndarray, pose_data: Dict) -> float:
        """Check ball proximity to hands"""
        keypoints = pose_data['keypoints']
        scores = pose_data['scores']
        
        
        right_wrist = keypoints[10]
        left_wrist = keypoints[9]
        
        if scores[10] < 0.3 and scores[9] < 0.3:
            return 0.0
        
        ball_center = np.array([
            (ball_bbox[0] + ball_bbox[2]) / 2,
            (ball_bbox[1] + ball_bbox[3]) / 2
        ])
        
        
        distances = []
        if scores[10] > 0.3:
            distances.append(np.linalg.norm(ball_center - right_wrist))
        if scores[9] > 0.3:
            distances.append(np.linalg.norm(ball_center - left_wrist))
        
        if not distances:
            return 0.0
        
        min_distance = min(distances)
        
        
        return max(0, 1 - (min_distance / 150))
    
    def _fallback_detection(self, pose_sequence: List[Dict]) -> int:
        """Tier 3: Detects impact via acceleration + direction change"""
        if len(pose_sequence) < 4:
            return len(pose_sequence) // 2
        
        scores = []

        for i in range(2, len(pose_sequence) - 1):
            kp_prev = pose_sequence[i-1]['keypoints']
            kp_curr = pose_sequence[i]['keypoints']
            kp_next = pose_sequence[i+1]['keypoints']

            # Right hand
            v1 = kp_curr[10] - kp_prev[10]
            v2 = kp_next[10] - kp_curr[10]

            speed_change = np.linalg.norm(v2) - np.linalg.norm(v1)
            direction_change = 1 - np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            )

            impulse_score = max(0, speed_change) * direction_change
            scores.append(impulse_score)

        return np.argmax(scores) + 2

    def detect_ball_roboflow(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.rf_ball_model.predict(
            rgb,
            confidence=20,
            overlap=30
        ).json()

        preds = result.get("predictions", [])
        if not preds:
            return None

        best = max(preds, key=lambda x: x["confidence"])

        x, y, w, h = best["x"], best["y"], best["width"], best["height"]

        bbox = np.array([
            int(x - w / 2),
            int(y - h / 2),
            int(x + w / 2),
            int(y + h / 2)
        ])

        return bbox, float(best["confidence"])