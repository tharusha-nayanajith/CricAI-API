"""
Ball-Bat Detector - 4-Tier Detection System
============================================

Tier 1: Custom ball model  +  Custom cricket bat model  (BEST)
Tier 2: Custom ball model  +  Generic YOLO bat (class 35/37)
Tier 3: Custom ball model  +  Virtual bat from pose keypoints
Tier 4: Fallback — hand acceleration + direction change (no ball needed)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import os

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.kalman_ball_tracker import KalmanBallTracker
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import MODEL_FOLDER_PATH


class BallBatDetector:
    """
    4-Tier cricket contact detection.

    Tier 1 — Custom ball model + Custom cricket bat model
        Highest accuracy. Both objects detected by purpose-trained models.

    Tier 2 — Custom ball model + Generic YOLO bat (class 35/37/39)
        Ball precisely detected; bat detected by COCO pretrained model.
        Decent accuracy, bat bbox less precise than Tier 1.

    Tier 3 — Custom ball model + Virtual bat from pose
        Ball detected; bat inferred from wrist/elbow keypoints.
        Works when bat is occluded or YOLO misses it.

    Tier 4 — Acceleration + direction change (fallback)
        No reliable ball detection. Uses hand velocity physics only.
    """

    # ── Confidence thresholds ──────────────────────────────────────────────────
    BALL_CONF_THRESH   = 0.45   # custom ball model
    BAT_CONF_THRESH    = 0.2   # custom bat model  (higher = less false positives)
    GENERIC_BAT_THRESH = 0.30   # COCO bat classes

    # ── Tier trigger thresholds ────────────────────────────────────────────────
    TIER1_SCORE_THRESH = 0.25   # require both custom models confident
    TIER2_SCORE_THRESH = 0.45   # generic bat + ball
    TIER3_SCORE_THRESH = 0.15   # virtual bat + ball
    BALL_RATE_TIER1    = 30     # % of frames ball must be detected to use Tier 1/2
    BALL_RATE_TIER3    = 15     # % of frames ball must be detected to use Tier 3

    def __init__(
        self,
        model_path: str = None,
        use_custom_ball_detector: bool = True,
        use_custom_bat_detector: bool = True,
    ):
        """
        Args:
            model_path:               Path to general YOLO model (fallback bat/ball)
            use_custom_ball_detector: Use trained YOLOv8 ball model
            use_custom_bat_detector:  Use trained YOLOv8 cricket bat model
        """
        # ── General YOLO model (COCO pretrained — fallback) ──────────────────
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print("✓ Loaded custom general detection model")
        else:
            self.model = YOLO("yolov8n.pt")
            print("✓ Loaded YOLOv8n pretrained model (COCO)")

        # ── Kalman tracker for ball ───────────────────────────────────────────
        self.ball_tracker = KalmanBallTracker()

        # ── Custom BALL detector ──────────────────────────────────────────────
        self.yolo_ball_model = None
        if use_custom_ball_detector:
            self.yolo_ball_model = self._load_model(
                primary=Path(MODEL_FOLDER_PATH) / "yolov8_ball_detector" / "best_model.pt",
                fallback=Path(MODEL_FOLDER_PATH) / "yolov8_ball_detector" / "train" / "weights" / "best.pt",
                label="ball detector",
                required=True,
            )

        # ── Custom BAT detector ───────────────────────────────────────────────
        self.yolo_bat_model = None
        if use_custom_bat_detector:
            self.yolo_bat_model = self._load_model(
                primary=Path(MODEL_FOLDER_PATH) / "yolov8_bat_detector" / "best_model.pt",
                fallback=Path(MODEL_FOLDER_PATH) / "yolov8_bat_detector" / "train" / "weights" / "best.pt",
                label="bat detector",
                required=False,   # graceful fallback to COCO bat classes
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(
        self,
        primary: Path,
        fallback: Path,
        label: str,
        required: bool = False,
    ) -> Optional[YOLO]:
        """Load a YOLO model from primary or fallback path."""
        for path in (primary, fallback):
            if path.exists():
                print(f"🏏 Loading {label}: {path.name}")
                model = YOLO(str(path))
                print(f"✅ {label} loaded — {path}")
                return model

        msg = (
            f"❌ Trained {label} not found!\n"
            f"   Checked: {primary}\n"
            f"   And:     {fallback}\n"
            f"   Train using the appropriate training script first."
        )
        if required:
            raise FileNotFoundError(msg)
        else:
            print(f"⚠️  {msg}\n   Falling back to generic YOLO detection.")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Detection methods
    # ──────────────────────────────────────────────────────────────────────────

    def detect_ball_yolov8(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Detect ball using custom trained YOLOv8 model."""
        if self.yolo_ball_model is None:
            return None

        results = self.yolo_ball_model.predict(
            frame,
            conf=self.BALL_CONF_THRESH,
            iou=0.45,
            verbose=False,
            classes=[0],
        )

        if not results or not results[0].boxes:
            return None

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best  = confs.argmax()
        xyxy  = boxes.xyxy[best].cpu().numpy()
        conf  = float(confs[best])

        bbox = np.array([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
        return bbox, conf

    def detect_bat_yolov8(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detect cricket bat using custom trained YOLOv8 bat model.
        Returns (bbox_xyxy, confidence) or None.
        """
        if self.yolo_bat_model is None:
            return None

        results = self.yolo_bat_model.predict(
            frame,
            conf=self.BAT_CONF_THRESH,
            iou=0.45,
            verbose=False,
            classes=[1],   
        )

        if not results or not results[0].boxes:
            return None

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best  = confs.argmax()
        xyxy  = boxes.xyxy[best].cpu().numpy()
        conf  = float(confs[best])

        bbox = np.array([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
        return bbox, conf

    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect ball and bat using all available models.

        Priority order:
          Ball: custom ball model → COCO sports ball (class 32) fallback
          Bat:  custom bat model  → COCO baseball bat / racket (35/37/39) fallback
        """
        # ── Ball detection ────────────────────────────────────────────────────
        ball_bbox, ball_conf = None, 0.0

        if self.yolo_ball_model is not None:
            result = self.detect_ball_yolov8(frame)
            if result is not None:
                ball_bbox, ball_conf = result

        # ── Bat detection ─────────────────────────────────────────────────────
        bat_bbox,        bat_conf        = None,  0.0
        custom_bat_bbox, custom_bat_conf = None,  0.0

        if self.yolo_bat_model is not None:
            result = self.detect_bat_yolov8(frame)
            if result is not None:
                custom_bat_bbox, custom_bat_conf = result

        # ── Generic YOLO for fallback bat + fallback ball ─────────────────────
        generic_bat_bbox, generic_bat_conf = None, 0.0
        coco_results = self.model(frame, verbose=False)[0]

        for box in coco_results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()

            # COCO bat-like classes: 35=baseball bat, 37=tennis racket, 39=sports ball
            if cls in [35, 37, 39] and conf > self.GENERIC_BAT_THRESH:
                if conf > generic_bat_conf:
                    generic_bat_bbox = bbox
                    generic_bat_conf = conf

            # COCO ball fallback when custom ball model is absent
            elif cls == 32 and self.yolo_ball_model is None:
                if conf > ball_conf:
                    ball_bbox = bbox.astype(int)
                    ball_conf = conf

        # ── Resolve bat: prefer custom model ─────────────────────────────────
        if custom_bat_bbox is not None:
            bat_bbox = custom_bat_bbox
            bat_conf = custom_bat_conf
        else:
            bat_bbox = generic_bat_bbox
            bat_conf = generic_bat_conf

        return {
            "ball_bbox":         ball_bbox,
            "ball_confidence":   ball_conf,
            "bat_bbox":          bat_bbox,
            "bat_confidence":    bat_conf,
            # Expose both bat sources for tier logic
            "custom_bat_bbox":   custom_bat_bbox,
            "custom_bat_conf":   custom_bat_conf,
            "generic_bat_bbox":  generic_bat_bbox,
            "generic_bat_conf":  generic_bat_conf,
        }

    def get_virtual_bat_bbox(self, pose_data: Dict) -> Optional[np.ndarray]:
        """Create a virtual bat bounding box from wrist/elbow keypoints."""
        keypoints = pose_data["keypoints"]
        scores    = pose_data["scores"]

        right_wrist = keypoints[10]
        left_wrist  = keypoints[9]
        right_elbow = keypoints[8]
        left_elbow  = keypoints[7]

        if scores[10] < 0.3 and scores[9] < 0.3:
            return None

        if scores[10] > 0.3 and scores[8] > 0.3:
            hand_center   = right_wrist
            bat_direction = right_wrist - right_elbow
        elif scores[9] > 0.3 and scores[7] > 0.3:
            hand_center   = left_wrist
            bat_direction = left_wrist - left_elbow
        elif scores[10] > 0.3 and scores[9] > 0.3:
            hand_center   = (right_wrist + left_wrist) / 2
            bat_direction = right_wrist - left_wrist
        else:
            hand_center   = right_wrist if scores[10] > 0.3 else left_wrist
            bat_direction = np.array([0, 50])

        # Normalize and extend
        bat_length_vector = bat_direction / (np.linalg.norm(bat_direction) + 1e-6) * 100
        bat_width = 30

        return np.array([
            hand_center[0] - bat_width,
            hand_center[1] - 50,
            hand_center[0] + bat_width,
            hand_center[1] + 50,
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # Main contact detection
    # ──────────────────────────────────────────────────────────────────────────

    def detect_contact_frame(
        self,
        frames: List[np.ndarray],
        pose_sequence: List[Dict],
    ) -> Tuple[int, Dict]:
        """
        4-Tier contact detection across all frames.

        Returns:
            (contact_frame_index, metadata_dict)
        """
        print("🎯 4-Tier Ball-Bat Contact Detection")
        print("-" * 55)

        scores_t1, scores_t2, scores_t3 = [], [], []
        detections_log = []

        ball_detected_count       = 0
        custom_bat_detected_count = 0
        generic_bat_detected_count= 0

        for i, frame in enumerate(frames):
            detection = self.detect_objects(frame)

            # ── Kalman tracking for missing ball frames ───────────────────────
            if detection["ball_bbox"] is not None:
                center = np.array([
                    (detection["ball_bbox"][0] + detection["ball_bbox"][2]) / 2,
                    (detection["ball_bbox"][1] + detection["ball_bbox"][3]) / 2,
                ])
                self.ball_tracker.update(center)
            else:
                predicted = self.ball_tracker.predict()
                if predicted is not None:
                    detection["ball_bbox"] = np.array([
                        predicted[0] - 8, predicted[1] - 8,
                        predicted[0] + 8, predicted[1] + 8,
                    ])
                    detection["ball_confidence"] = 0.15   # low confidence for predicted

            # ── Count detections ──────────────────────────────────────────────
            if detection["ball_bbox"]       is not None: ball_detected_count       += 1
            if detection["custom_bat_bbox"] is not None: custom_bat_detected_count += 1
            if detection["generic_bat_bbox"]is not None: generic_bat_detected_count+= 1

            # ── Virtual bat ───────────────────────────────────────────────────
            pose = pose_sequence[i] if i < len(pose_sequence) else None
            detection["virtual_bat_bbox"] = self.get_virtual_bat_bbox(pose) if pose else None

            detections_log.append(detection)

            # ── Per-frame tier scores ─────────────────────────────────────────
            scores_t1.append(self._score_tier1(detection, pose))
            scores_t2.append(self._score_tier2(detection, pose))
            scores_t3.append(self._score_tier3(detection, pose))

        n = len(frames)
        ball_rate        = (ball_detected_count        / n) * 100
        custom_bat_rate  = (custom_bat_detected_count  / n) * 100
        generic_bat_rate = (generic_bat_detected_count / n) * 100

        print(f"📊 Detection Rates:")
        print(f"   Ball (custom):       {ball_rate:.1f}% ({ball_detected_count}/{n})")
        print(f"   Bat  (custom):       {custom_bat_rate:.1f}% ({custom_bat_detected_count}/{n})")
        print(f"   Bat  (generic COCO): {generic_bat_rate:.1f}% ({generic_bat_detected_count}/{n})")

        # ── Tier selection ────────────────────────────────────────────────────
        max_t1 = max(scores_t1)
        max_t2 = max(scores_t2)
        max_t3 = max(scores_t3)

        if max_t1 > self.TIER1_SCORE_THRESH and ball_rate > self.BALL_RATE_TIER1:
            peak = int(np.argmax(scores_t1))

            # search ±2 frames for closest ball-bat distance
            best_idx = peak
            best_dist = float("inf")

            for j in range(max(0, peak-2), min(len(frames), peak+3)):
                d = detections_log[j]
                if d["ball_bbox"] is None or d["bat_bbox"] is None:
                    continue

                dist = self._bbox_distance(d["ball_bbox"], d["bat_bbox"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j

            contact_idx = best_idx
            method      = "tier1_custom_ball_custom_bat"
            print(f"✅ Tier 1: Custom ball + Custom bat  (score: {max_t1:.3f})")

        elif max_t2 > self.TIER2_SCORE_THRESH and ball_rate > self.BALL_RATE_TIER1:
            contact_idx = int(np.argmax(scores_t2))
            method      = "tier2_custom_ball_generic_bat"
            print(f"✅ Tier 2: Custom ball + Generic bat  (score: {max_t2:.3f})")

        elif max_t3 > self.TIER3_SCORE_THRESH and ball_rate > self.BALL_RATE_TIER3:
            contact_idx = int(np.argmax(scores_t3))
            method      = "tier3_custom_ball_virtual_bat"
            print(f"⚠️  Tier 3: Custom ball + Virtual bat  (score: {max_t3:.3f})")

        else:
            contact_idx = self._fallback_detection(pose_sequence)
            method      = "tier4_acceleration_direction_change"
            print(f"⚠️  Tier 4: Fallback — acceleration + direction change"
                  f"  (ball rate: {ball_rate:.1f}%)")

        d = detections_log[contact_idx]
        contact_metadata = {
            "contact_frame":      contact_idx,
            "detection_method":   method,
            "ball_detected":      d["ball_bbox"]        is not None,
            "bat_detected":       d["bat_bbox"]         is not None,
            "custom_bat_used":    d["custom_bat_bbox"]  is not None,
            "virtual_bat_used":   method in ("tier3_custom_ball_virtual_bat",
                                             "tier4_acceleration_direction_change"),
            "ball_detection_rate":     round(ball_rate,        1),
            "custom_bat_detection_rate":round(custom_bat_rate,  1),
            "generic_bat_detection_rate":round(generic_bat_rate, 1),
            "tier1_score": round(float(scores_t1[contact_idx]), 4),
            "tier2_score": round(float(scores_t2[contact_idx]), 4),
            "tier3_score": round(float(scores_t3[contact_idx]), 4),
        }

        # ── Save contact frame for debugging ─────────────────────────────
        # debug_folder = "debug_contact_frames"
        # os.makedirs(debug_folder, exist_ok=True)
        # save_path = os.path.join(debug_folder, f"contact_frame_{contact_idx}.jpg")
        # frame_debug = frames[contact_idx].copy()

        # if d["ball_bbox"] is not None:
        #     x1,y1,x2,y2 = map(int, d["ball_bbox"])
        #     cv2.rectangle(frame_debug,(x1,y1),(x2,y2),(0,255,0),2)

        # if d["bat_bbox"] is not None:
        #     x1,y1,x2,y2 = map(int, d["bat_bbox"])
        #     cv2.rectangle(frame_debug,(x1,y1),(x2,y2),(255,0,0),2)

        # cv2.putText(frame_debug,"CONTACT FRAME",(40,40),
        #             cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        # cv2.imwrite(save_path, frame_debug)

        # print(f"📸 Contact frame saved → {save_path}")

        print(f"✓ Contact frame: {contact_idx}/{n}  method: {method}")
        print("-" * 55)

        return contact_idx, contact_metadata

    # ──────────────────────────────────────────────────────────────────────────
    # Tier scoring functions
    # ──────────────────────────────────────────────────────────────────────────

    def _score_tier1(self, detection: Dict, pose_data: Optional[Dict]) -> float:
        """
        Tier 1: Custom ball model + Custom bat model.
        Both must be present and confident.
        """
        ball_bbox = detection.get("ball_bbox")
        bat_bbox  = detection.get("custom_bat_bbox")   # custom bat ONLY

        if ball_bbox is None or bat_bbox is None:
            return 0.0

        score = 0.0

        # High confidence from both custom models is the key signal
        ball_conf = detection.get("ball_confidence", 0.0)
        bat_conf  = detection.get("custom_bat_conf",  0.0)
        score += ball_conf * 0.25
        score += bat_conf  * 0.30   # slightly higher weight — bat model is new

        # Spatial proximity
        distance = self._bbox_distance(ball_bbox, bat_bbox)

        # Hard contact condition
        # if distance > 60:
        #     return 0.0

        proximity = np.exp(-distance / 30.0)
        score += proximity * 0.40

        # IoU overlap
        iou = self._calculate_iou(ball_bbox, bat_bbox)
        score += iou * 0.15

        # Ball near hands (pose consistency check)
        if pose_data is not None:
            score += self._ball_near_hands(ball_bbox, pose_data) * 0.15

        return float(np.clip(score, 0.0, 1.0))

    def _score_tier2(self, detection: Dict, pose_data: Optional[Dict]) -> float:
        """
        Tier 2: Custom ball model + Generic YOLO bat.
        Custom bat must be absent (otherwise Tier 1 applies).
        """
        ball_bbox         = detection.get("ball_bbox")
        custom_bat_bbox   = detection.get("custom_bat_bbox")
        generic_bat_bbox  = detection.get("generic_bat_bbox")

        # Only use generic bat when custom bat did not fire
        if ball_bbox is None or generic_bat_bbox is None:
            return 0.0
        if custom_bat_bbox is not None:
            # Custom bat present → Tier 1 handles this, don't double-count
            return 0.0

        score = 0.0

        ball_conf     = detection.get("ball_confidence",  0.0)
        generic_conf  = detection.get("generic_bat_conf", 0.0)
        score += ball_conf    * 0.30
        score += generic_conf * 0.20   # lower weight — COCO bat is less precise

        distance  = self._bbox_distance(ball_bbox, generic_bat_bbox)
        proximity = np.exp(-distance / 80.0)
        score += proximity * 0.25

        iou = self._calculate_iou(ball_bbox, generic_bat_bbox)
        score += iou * 0.15

        if pose_data is not None:
            score += self._ball_near_hands(ball_bbox, pose_data) * 0.15

        return float(np.clip(score, 0.0, 1.0))

    def _score_tier3(self, detection: Dict, pose_data: Optional[Dict]) -> float:
        """
        Tier 3: Custom ball + Virtual bat (pose-derived).
        Only when no real bat was detected.
        """
        ball_bbox         = detection.get("ball_bbox")
        virtual_bat_bbox  = detection.get("virtual_bat_bbox")

        if ball_bbox is None:
            return 0.0

        # Prefer real bat tiers — don't compete with Tier 1/2
        if (detection.get("custom_bat_bbox")  is not None or
                detection.get("generic_bat_bbox") is not None):
            return 0.0

        score = 0.0

        ball_conf = detection.get("ball_confidence", 0.0)
        score += ball_conf * 0.40

        if pose_data is not None:
            score += self._ball_near_hands(ball_bbox, pose_data) * 0.40

        if virtual_bat_bbox is not None:
            distance  = self._bbox_distance(ball_bbox, virtual_bat_bbox)
            proximity = np.exp(-distance / 80.0)
            score += proximity * 0.25
            score += self._calculate_iou(ball_bbox, virtual_bat_bbox) * 0.25

        return float(np.clip(score, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    # Tier 4 fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _fallback_detection(self, pose_sequence: List[Dict]) -> int:
        """Tier 4: Detect impact via hand acceleration + direction change."""
        if len(pose_sequence) < 4:
            return len(pose_sequence) // 2

        scores = []
        for i in range(2, len(pose_sequence) - 1):
            kp_prev = pose_sequence[i - 1]["keypoints"]
            kp_curr = pose_sequence[i    ]["keypoints"]
            kp_next = pose_sequence[i + 1]["keypoints"]

            v1 = kp_curr[10] - kp_prev[10]
            v2 = kp_next[10] - kp_curr[10]

            speed_change     = np.linalg.norm(v2) - np.linalg.norm(v1)
            direction_change = 1 - np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            )
            scores.append(max(0, speed_change) * direction_change)

        return int(np.argmax(scores)) + 2

    # ──────────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        x1 = max(bbox1[0], bbox2[0]);  y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2]);  y2 = min(bbox1[3], bbox2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1    = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        a2    = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def _bbox_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        c1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        c2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        return float(np.linalg.norm(c1 - c2))

    def _ball_near_hands(self, ball_bbox: np.ndarray, pose_data: Dict) -> float:
        """Score [0,1] based on ball proximity to nearest detected wrist."""
        keypoints = pose_data["keypoints"]
        scores    = pose_data["scores"]

        right_wrist = keypoints[10]
        left_wrist  = keypoints[9]

        if scores[10] < 0.3 and scores[9] < 0.3:
            return 0.0

        ball_center = np.array([
            (ball_bbox[0] + ball_bbox[2]) / 2,
            (ball_bbox[1] + ball_bbox[3]) / 2,
        ])

        distances = []
        if scores[10] > 0.3:
            distances.append(np.linalg.norm(ball_center - right_wrist))
        if scores[9] > 0.3:
            distances.append(np.linalg.norm(ball_center - left_wrist))

        return max(0.0, 1.0 - (min(distances) / 150.0)) if distances else 0.0