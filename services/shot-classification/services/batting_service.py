"""
Batting Service
"""

import numpy as np
import joblib
import os
import json
import cv2
import tempfile
import shutil
from typing import Dict, List, Optional
from google import genai
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    MODEL_FOLDER_PATH,
    SHOT_TYPES,
    ANALYZE_SHOT_MODE_NEW,
    ANALYZE_SHOT_MODE_LEGACY,
    ANALYZE_SHOT_MODE_DEFAULT,
)
from features.SHOT_CLASSIFICATION_SYSTEM.utils.json_utils import to_json_safe


class EfficientNetVideoClassifier:
    """Inference wrapper for EfficientNetB4 + GRU video classifier."""

    FRAME_COUNT = 30
    MIN_FRAME_COUNT = 20
    FRAME_SIZE = (224, 224)

    def _log(self, message: str):
        print(f"[EfficientNetMode] {message}", flush=True)

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.video_model_dir = os.path.join(model_dir, "video_classifier")

        self._log("Initializing cached EfficientNetB4 + GRU model")

        metadata_path = os.path.join(self.video_model_dir, "metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.shot_types = metadata.get("shot_types", SHOT_TYPES)
            self.FRAME_COUNT = int(metadata.get("frame_count", self.FRAME_COUNT))
            self.FRAME_SIZE = tuple(metadata.get("frame_size", self.FRAME_SIZE))
        else:
            self.shot_types = SHOT_TYPES

        self._log(f"Looking for cached model at {os.path.join(self.video_model_dir, 'model_complete.keras')}")
        self.model = self._load_cached_model()
        if self.model is None:
            self._log("Cache unavailable, rebuilding architecture from weights")
            self.model = self._build_model(num_classes=len(self.shot_types))
            self._load_weights()
        else:
            self._log("Cached model loaded successfully")

        self._ensure_model_built()
        self._log("Preparing feature extractor")

        feature_input = keras.Input(
            shape=(self.FRAME_COUNT, self.FRAME_SIZE[0], self.FRAME_SIZE[1], 3)
        )
        feature_output = feature_input
        for layer in self.model.layers[:5]:
            feature_output = layer(feature_output)

        self.feature_extractor = models.Model(feature_input, feature_output)
        self._log("Feature extractor ready")

        self.prototypes = self._load_prototypes()
        self.feature_importance = self._load_feature_importance()

    def _build_model(self, num_classes: int) -> keras.Model:
        base_model = EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=(self.FRAME_SIZE[0], self.FRAME_SIZE[1], 3),
        )
        base_model.trainable = False

        model = models.Sequential([
            layers.Input(shape=(self.FRAME_COUNT, self.FRAME_SIZE[0], self.FRAME_SIZE[1], 3)),
            layers.TimeDistributed(base_model),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.GRU(256, return_sequences=True, dropout=0.3, unroll=True),
            layers.GRU(128, dropout=0.3, unroll=True),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    def _load_cached_model(self):
        cache_path = os.path.join(self.video_model_dir, "model_complete.keras")
        if not os.path.exists(cache_path):
            self._log("Cached model file not found")
            return None

        try:
            self._log("Loading cached compiled model")
            model = keras.models.load_model(cache_path)
            print("✓ Loaded pre-compiled model from cache (skipped rebuild)")
            return model
        except Exception as e:
            print(f"⚠ Cache load failed ({e}), rebuilding from weights...")
            return None

    def _load_weights(self):
        # Fallback: load weights into rebuilt architecture.
        best_path = os.path.join(self.video_model_dir, "best_model.weights.h5")
        model_path = os.path.join(self.video_model_dir, "model.weights.h5")

        selected_path = best_path if os.path.exists(best_path) else model_path
        if not os.path.exists(selected_path):
            raise FileNotFoundError(
                f"No EfficientNet weights found in {self.video_model_dir}. "
                "Expected best_model.weights.h5 or model.weights.h5"
            )

        self._log(f"Loading weights from {selected_path}")

        try:
            self.model.load_weights(selected_path)
            self._log("Weights loaded successfully")
            return
        except ValueError:
            # Keras 3 may treat .weights.h5 as non-legacy format.
            # Create a temporary .h5 copy and load by_name for legacy compatibility.
            self._log("Retrying weight load through legacy .h5 compatibility path")
            with tempfile.TemporaryDirectory() as tmp_dir:
                legacy_path = os.path.join(tmp_dir, "legacy_weights.h5")
                shutil.copyfile(selected_path, legacy_path)
                self.model.load_weights(legacy_path, by_name=True, skip_mismatch=True)
            self._log("Legacy-compatible weights loaded successfully")

    def _ensure_model_built(self):
        """Ensure the loaded Sequential model has a defined input tensor."""
        expected_shape = (
            None,
            self.FRAME_COUNT,
            self.FRAME_SIZE[0],
            self.FRAME_SIZE[1],
            3,
        )

        try:
            if not self.model.built:
                self.model.build(expected_shape)
        except Exception:
            pass

        if not getattr(self.model, "inputs", None):
            dummy_input = tf.zeros((1, self.FRAME_COUNT, self.FRAME_SIZE[0], self.FRAME_SIZE[1], 3), dtype=tf.float32)
            _ = self.model(dummy_input, training=False)

    def _load_prototypes(self) -> Dict[str, Dict]:
        path = os.path.join(self.video_model_dir, "shot_prototypes.pkl")
        if not os.path.exists(path):
            return {}

        raw = joblib.load(path)
        if not raw:
            return {}

        # Training artifacts may be keyed by numeric class id. Normalize to shot names.
        sample_key = next(iter(raw.keys()))
        if isinstance(sample_key, str):
            return raw

        mapped = {}
        for key, value in raw.items():
            class_idx = int(key)
            if 0 <= class_idx < len(self.shot_types):
                mapped[self.shot_types[class_idx]] = value
        return mapped

    def _load_feature_importance(self) -> Optional[np.ndarray]:
        path = os.path.join(self.video_model_dir, "feature_importance.pkl")
        if not os.path.exists(path):
            return None

        importance = joblib.load(path)
        if isinstance(importance, np.ndarray):
            return importance.astype(np.float32)
        return None

    def extract_30_frames(self, video_path: str) -> np.ndarray:
        self._log(f"Extracting 30 frames from {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.MIN_FRAME_COUNT:
            cap.release()
            raise ValueError(
                f"Video has only {total_frames} frames, need at least {self.MIN_FRAME_COUNT}"
            )

        frames = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.FRAME_SIZE)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No decodable frames in video: {video_path}")

        frames = np.array(frames, dtype=np.uint8)
        if len(frames) > self.FRAME_COUNT:
            indices = np.linspace(0, len(frames) - 1, self.FRAME_COUNT, dtype=int)
            frames = frames[indices]
        elif len(frames) < self.FRAME_COUNT:
            padding = np.tile(frames[-1:], (self.FRAME_COUNT - len(frames), 1, 1, 1))
            frames = np.vstack([frames, padding])

        return frames

    def _prepare_input(self, video_path: str) -> np.ndarray:
        self._log("Preparing normalized model input")
        frames = self.extract_30_frames(video_path).astype(np.float32)
        return preprocess_input(frames)

    def predict(self, video_path: str) -> Dict:
        self._log("Starting shot prediction")
        video_tensor = self._prepare_input(video_path)
        self._log("Running model inference")
        probs = self.model.predict(np.expand_dims(video_tensor, axis=0), verbose=0)[0]

        probs = probs.astype(np.float64)
        shot_probabilities = {
            shot: float(prob) for shot, prob in zip(self.shot_types, probs)
        }

        pred_idx = int(np.argmax(probs))
        predicted_shot = self.shot_types[pred_idx]

        self._log(f"Prediction complete: {predicted_shot}")

        return {
            'final_prediction': predicted_shot,
            'ensemble_probabilities': shot_probabilities,
            'individual_predictions': {
                'efficientnet_b4_gru': predicted_shot
            },
            'individual_probabilities': {
                'efficientnet_b4_gru': shot_probabilities
            }
        }

    def extract_embedding(self, video_path: str) -> np.ndarray:
        self._log("Extracting embedding from GRU stack")
        video_tensor = self._prepare_input(video_path)
        embedding = self.feature_extractor.predict(
            np.expand_dims(video_tensor, axis=0),
            verbose=0,
        )[0]
        self._log("Embedding extraction complete")
        return embedding.astype(np.float32)


def _normalize_mode(mode: Optional[object]) -> int:
    if mode is None:
        return ANALYZE_SHOT_MODE_DEFAULT

    if isinstance(mode, int):
        return mode

    mode_str = str(mode).strip().lower()
    if mode_str in {"1", "new", "efficient", "efficientnet", "efficientnetb4", "gru"}:
        return ANALYZE_SHOT_MODE_NEW
    if mode_str in {"2", "legacy", "old"}:
        return ANALYZE_SHOT_MODE_LEGACY

    raise ValueError(f"Invalid mode '{mode}'. Use 1/new or 2/legacy.")


class BattingService:
    """Advanced batting analysis with 3D avatar support"""
    
    # Joint name mapping for frontend
    JOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Severity color mapping
    SEVERITY_COLORS = {
        'critical': '#e74c3c',
        'major': '#f39c12',
        'minor': '#f1c40f',
        'negligible': '#95a5a6'
    }
    
    def __init__(self, model_dir: str = MODEL_FOLDER_PATH, mode: Optional[object] = None):
        self.model_dir = model_dir
        self.mode = _normalize_mode(mode)

        # Shared assets used by response rendering and feedback.
        self.prototypes = joblib.load(f"{model_dir}/prototypes/shot_prototypes.pkl")
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.frame_extractor = None
        self.pose_estimator = None
        self.feature_engineer = None
        self.mistake_analyzer = None
        self.video_classifier = None

        if self.mode == ANALYZE_SHOT_MODE_LEGACY:
            # Legacy path: full RTMPose + engineered-feature ensemble stack.
            from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
            from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
            from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.temporal_feature_engineer import TemporalFeatureEngineer
            from features.SHOT_CLASSIFICATION_SYSTEM.utils.model_based_mistake_analyzer import ModelBasedMistakeAnalyzer

            self.models = self._load_models()
            self.scaler = joblib.load(f"{model_dir}/ensemble/scaler.pkl")
            self.label_encoder = joblib.load(f"{model_dir}/ensemble/label_encoder.pkl")

            with open(f"{model_dir}/ensemble/feature_names.json", 'r') as f:
                self.feature_names = json.load(f)

            self.frame_extractor = FrameExtractor(fps=10)
            self.pose_estimator = PoseEstimator()
            self.feature_engineer = TemporalFeatureEngineer()

            self.mistake_analyzer = ModelBasedMistakeAnalyzer(
                prototypes_path=f"{model_dir}/prototypes/shot_prototypes.pkl",
                feature_importance_path=f"{model_dir}/prototypes/feature_importance.pkl",
                feature_names=self.feature_names,
            )
        else:
            # New path: EfficientNetB4 + GRU only; no YOLO/RTMPose initialization.
            print("[EfficientNetMode] BattingService running in mode 1 (EfficientNetB4 + GRU)", flush=True)
            self.video_classifier = EfficientNetVideoClassifier(model_dir=model_dir)
        
        # AI feedback
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.ai_client = genai.Client(api_key=api_key) if api_key else None

        print(f"✓ Batting service initialized (mode={self.mode})")
    
    def _load_models(self) -> Dict:
        """Load ensemble models"""
        models = {}
        for model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
            model_path = f"{self.model_dir}/{model_name}/model_latest.pkl"
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        return models
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process video and extract features with YOLO ball-bat detection
        """
        # Extract frames
        frames, fps = self.frame_extractor.extract_frames(video_path)
        
        # Get pose sequence
        pose_sequence = self.pose_estimator.estimate_pose_batch(frames)
        
        # Extract temporal features
        features, metadata = self.feature_engineer.extract_temporal_features(pose_sequence, frames)
        
        # Store frames and poses for visual feedback
        contact_frame_idx = metadata['contact_frame']
        contact_frame = frames[contact_frame_idx]
        contact_pose = pose_sequence[contact_frame_idx]
        
        return {
            'features': features,
            'metadata': metadata,
            'contact_frame': contact_frame,
            'contact_keypoints': contact_pose['keypoints'],
            'contact_scores': contact_pose['scores'],
            'yolo_detection': metadata.get('contact_detection', {})
        }

    def process_video_new(self, video_path: str) -> Dict:
        """Process video using EfficientNetB4 + GRU pipeline only."""
        print("[EfficientNetMode] Processing video for mode 1 analysis", flush=True)
        prediction = self.video_classifier.predict(video_path)
        print("[EfficientNetMode] Prediction done, extracting embedding", flush=True)
        embedding = self.video_classifier.extract_embedding(video_path)

        return {
            'features': embedding,
            'prediction': prediction,
            'metadata': {
                'contact_frame': -1,
                'contact_detection': {},
            },
        }
    
    def ensemble_predict(self, features: np.ndarray) -> Dict:
        """
        Get predictions from ensemble with voting
        
        Returns:
            Dictionary with ensemble results
        """
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from each model
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            predictions[name] = self.label_encoder.inverse_transform([pred])[0]
            probabilities[name] = {
                shot: float(prob) 
                for shot, prob in zip(self.label_encoder.classes_, proba)
            }
        
        # Ensemble voting (average probabilities)
        ensemble_proba = {}
        for shot_class in self.label_encoder.classes_:
            avg_prob = np.mean([probabilities[model][shot_class] for model in self.models.keys()])
            ensemble_proba[shot_class] = float(avg_prob)
        
        # Final prediction (highest probability)
        final_prediction = max(ensemble_proba, key=ensemble_proba.get)
        
        return {
            'final_prediction': final_prediction,
            'ensemble_probabilities': ensemble_proba,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }
    
    def calculate_intent_score(self, intended_shot: str, ensemble_proba: Dict[str, float]) -> float:
        """Calculate intent score"""
        intended_prob = ensemble_proba.get(intended_shot, 0.0)
        max_prob = max(ensemble_proba.values())
        
        if max_prob > 0:
            score = (intended_prob / max_prob) * 100
        else:
            score = 0.0
        
        return round(score, 2)

    def _build_fallback_keypoints(self, intended_shot: str) -> np.ndarray:
        """Return keypoints even for model-only flow so response shape remains stable."""
        if intended_shot in self.prototypes:
            keypoints = self.prototypes[intended_shot].get('keypoints', {}).get('mean')
            if keypoints is not None:
                return np.array(keypoints, dtype=np.float32)
        return np.zeros((17, 2), dtype=np.float32)

    def analyze_execution_new(self, intended_shot: str, predicted_shot: str,
                              embedding: np.ndarray) -> List[Dict]:
        """Embedding-based mistake analysis for EfficientNetB4 + GRU mode."""
        print("[EfficientNetMode] Running embedding-based mistake analysis", flush=True)
        prototypes = self.video_classifier.prototypes
        if intended_shot not in prototypes:
            print("[EfficientNetMode] No prototype found for intended shot", flush=True)
            return []

        target = prototypes[intended_shot].get('features', {}).get('mean')
        std = prototypes[intended_shot].get('features', {}).get('std')
        if target is None:
            return []

        target = np.array(target, dtype=np.float32)
        std = np.array(std, dtype=np.float32) if std is not None else np.ones_like(target)

        diffs = embedding - target
        normalized = np.abs(diffs) / np.maximum(std, 1e-6)

        if self.video_classifier.feature_importance is not None:
            importance = self.video_classifier.feature_importance
            if len(importance) == len(normalized):
                severity_scores = normalized * importance
            else:
                severity_scores = normalized
        else:
            severity_scores = normalized

        top_indices = np.argsort(severity_scores)[-5:][::-1]
        mistakes: List[Dict] = []

        for idx in top_indices:
            score = float(severity_scores[idx])
            if score <= 1e-6:
                continue

            if score > 1.5:
                severity = 'critical'
            elif score > 0.7:
                severity = 'major'
            elif score > 0.2:
                severity = 'minor'
            else:
                severity = 'negligible'

            direction = 'higher' if diffs[idx] > 0 else 'lower'
            mistakes.append({
                'body_part': 'Body Position',
                'joint_id': 'torso_bend',
                'feature_name': f'embedding_{int(idx):03d}',
                'severity': severity,
                'severity_score': score,
                'actual_value': float(embedding[idx]),
                'expected_value': float(target[idx]),
                'deviation': float(diffs[idx]),
                'importance': float(severity_scores[idx]),
                'explanation': (
                    f"Your movement embedding component {int(idx)} was {direction} "
                    f"than expected for a {intended_shot}, causing similarity drift toward {predicted_shot}."
                ),
                'recommendation': (
                    f"Repeat {intended_shot} drills with controlled bat path and balance "
                    "to align full-body sequencing with the learned prototype."
                ),
            })

        return mistakes

    def _generate_correction_summary_new(self, mistakes: List[Dict], intended_shot: str) -> str:
        if not mistakes:
            return f"Excellent technique! Your {intended_shot} execution was optimal."

        critical = [m for m in mistakes if m['severity'] == 'critical']
        major = [m for m in mistakes if m['severity'] == 'major']
        minor = [m for m in mistakes if m['severity'] == 'minor']

        summary_parts = []
        if critical:
            summary_parts.append(f"Critical ({len(critical)})")
        if major:
            summary_parts.append(f"Major ({len(major)})")
        if minor:
            summary_parts.append(f"Minor ({len(minor)})")

        summary = " | ".join(summary_parts) if summary_parts else "Minor deviations detected"
        summary += "\n\nPriority: focus on timing, balance, and bat-path consistency first."
        return summary
    
    def convert_to_3d_keypoints(self, keypoints_2d: np.ndarray) -> List[Dict]:
        """
        Convert 2D keypoints to 3D with proper structure for frontend
        
        Returns:
            List of dicts with joint names and 3D coordinates
        """
        keypoints_3d = []
        
        # Estimate depth (Z-axis) from biomechanical structure
        for i, joint_name in enumerate(self.JOINT_NAMES):
            x, y = keypoints_2d[i]
            
            # Estimate Z (depth) based on body structure
            # Head/shoulders: forward
            if i < 7:  # Head and shoulders
                z = 50
            # Elbows: slight forward
            elif i in [7, 8]:
                z = 30
            # Hands: further forward (bat contact plane)
            elif i in [9, 10]:
                z = 80
            # Hips: center
            elif i in [11, 12]:
                z = 0
            # Legs: slightly back
            else:
                z = -20
            
            keypoints_3d.append({
                'joint': joint_name,
                'index': i,
                'position': {
                    'x': float(x),
                    'y': float(-y),  # Flip Y for 3D coordinate system
                    'z': float(z)
                }
            })
        
        return keypoints_3d
    
    def prepare_mistake_visualization(self, mistakes: List[Dict]) -> List[Dict]:
        """
        Prepare mistakes for frontend visualization
        
        Returns:
            List of mistakes with severity colors and glow intensity
        """
        visualization_data = []
        
        for mistake in mistakes:
            joint_id = mistake.get('joint_id')
            if not joint_id:
                continue
            
            severity = mistake['severity']
            severity_score = mistake['severity_score']
            
            # Map severity to color and intensity
            color = self.SEVERITY_COLORS.get(severity, '#95a5a6')
            
            # Glow intensity based on severity score (0-1)
            intensity = min(1.0, severity_score / 2.0)
            
            visualization_data.append({
                'joint_id': joint_id,
                'body_part': mistake['body_part'],
                'severity': severity,
                'severity_color': color,
                'glow_intensity': float(intensity),
                'explanation': mistake['explanation'],
                'recommendation': mistake['recommendation']
            })
        
        return visualization_data
    
    def generate_visual_feedback_for_frontend(self, actual_keypoints: np.ndarray,
                                             intended_shot: str,
                                             mistakes: List[Dict]) -> Dict:
        """
        Generate visual feedback optimized for frontend 3D rendering
        """
        # Convert to 3D keypoints
        actual_keypoints_3d = self.convert_to_3d_keypoints(actual_keypoints)
        
        # Get prototype keypoints
        if intended_shot in self.prototypes:
            prototype_keypoints_2d = self.prototypes[intended_shot]['keypoints']['mean']
            prototype_keypoints_3d = self.convert_to_3d_keypoints(prototype_keypoints_2d)
        else:
            prototype_keypoints_3d = actual_keypoints_3d  # Fallback
        
        # Prepare mistake visualization
        mistake_viz = self.prepare_mistake_visualization(mistakes)
        
        # Generate images (optional, for backward compatibility)
        # skeleton_3d = self.skeleton_animator.generate_3d_skeleton(
        #     actual_keypoints,
        #     mistakes,
        #     view_angle=(30, 45)
        # )
        
        # comparison_view = self.skeleton_animator.generate_comparison_view(
        #     actual_keypoints,
        #     prototype_keypoints_2d,
        #     mistakes
        # )

        # # 3. Generate 360° animation
        # animation_360 = self.skeleton_animator.generate_multi_angle_animation(
        #     actual_keypoints,
        #     mistakes
        # )
        
        return {
            # For 3D Avatar Frontend (PRIMARY)
            'keypoints_3d': {
                'actual': actual_keypoints_3d,
                'prototype': prototype_keypoints_3d,
                'format': 'three_js_compatible'
            },
            'mistakes': mistake_viz,
            'joint_connections': self._get_skeleton_connections(),
            
            # For backward compatibility (OPTIONAL)
            # 'legacy_images': {
            #     'skeleton_3d': skeleton_3d,
            #     'comparison_view': comparison_view,
            #     'animation_360': animation_360
            # },
            
            # Metadata
            'prototype_used': intended_shot,
            'prototype_samples': self.prototypes.get(intended_shot, {}).get('n_samples', 0)
        }
    
    def _get_skeleton_connections(self) -> List[Dict]:
        """
        Get skeleton bone connections for frontend rendering
        """
        connections = [
            # Head
            {'from': 'nose', 'to': 'left_eye', 'label': 'head'},
            {'from': 'nose', 'to': 'right_eye', 'label': 'head'},
            {'from': 'left_eye', 'to': 'left_ear', 'label': 'head'},
            {'from': 'right_eye', 'to': 'right_ear', 'label': 'head'},
            
            # Torso
            {'from': 'left_shoulder', 'to': 'right_shoulder', 'label': 'torso'},
            {'from': 'left_shoulder', 'to': 'left_hip', 'label': 'torso'},
            {'from': 'right_shoulder', 'to': 'right_hip', 'label': 'torso'},
            {'from': 'left_hip', 'to': 'right_hip', 'label': 'torso'},
            
            # Right arm
            {'from': 'right_shoulder', 'to': 'right_elbow', 'label': 'right_arm'},
            {'from': 'right_elbow', 'to': 'right_wrist', 'label': 'right_arm'},
            
            # Left arm
            {'from': 'left_shoulder', 'to': 'left_elbow', 'label': 'left_arm'},
            {'from': 'left_elbow', 'to': 'left_wrist', 'label': 'left_arm'},
            
            # Right leg
            {'from': 'right_hip', 'to': 'right_knee', 'label': 'right_leg'},
            {'from': 'right_knee', 'to': 'right_ankle', 'label': 'right_leg'},
            
            # Left leg
            {'from': 'left_hip', 'to': 'left_knee', 'label': 'left_leg'},
            {'from': 'left_knee', 'to': 'left_ankle', 'label': 'left_leg'}
        ]
        
        return connections
    
    def generate_ai_feedback(self, intended_shot: str, predicted_shot: str,
                           intent_score: float, mistakes: List[Dict]) -> str:
        """Generate AI feedback"""
        if not self.ai_client:
            return self._generate_rule_based_feedback(intended_shot, predicted_shot, 
                                                      intent_score, mistakes)
        
        try:
            # Prepare mistake summary
            mistake_summary = "\n".join([
                f"- {m['body_part']}: {m['explanation']}"
                for m in mistakes[:3]
            ])
            
            prompt = f"""You are an expert cricket coach analyzing a player's {intended_shot} execution.

Player's Intent: {intended_shot.upper()}
Actual Execution: {predicted_shot.upper()}
Intent Score: {intent_score}% (similarity to {self.prototypes[intended_shot]['n_samples']} correct {intended_shot} examples)

Key Technical Issues (detected by biomechanical analysis):
{mistake_summary if mistakes else "No major technical issues detected."}

Provide coaching feedback in 2-3 concise sentences:
1. Start with acknowledgment
2. Point out the main biomechanical issue
3. Give ONE specific, actionable correction

Be direct, supportive, coaching-focused and technically accurate. No bullet points."""

            response = self.ai_client.models.generate_content(
                model="gemini-2.5-flash",  
                contents=prompt
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"AI feedback failed: {e}")
            return self._generate_rule_based_feedback(intended_shot, predicted_shot, 
                                                      intent_score, mistakes)
    
    def _generate_rule_based_feedback(self, intended_shot: str, predicted_shot: str,
                                     intent_score: float, mistakes: List[Dict]) -> str:
        """Fallback feedback"""
        n_samples = self.prototypes.get(intended_shot, {}).get('n_samples', 0)
        
        if intent_score >= 85:
            base = f"Excellent {intended_shot}! Your biomechanics match our {n_samples} reference examples."
        elif intent_score >= 70:
            base = f"Good {intended_shot} attempt. Your execution is close to the learned prototype."
        elif intent_score >= 50:
            base = f"Your {intended_shot} deviates from the {n_samples} training examples."
        else:
            base = f"Significant deviation from correct {intended_shot} form (appeared as {predicted_shot})."
        
        if mistakes:
            top_issue = mistakes[0]
            return f"{base} Main issue: {top_issue['explanation']} {top_issue['recommendation']}"
        
        return base
    
    def analyze_shot(self, video_path: str, intended_shot: str) -> Dict:
        """
        Complete shot analysis with visual feedback
        
        Args:
            video_path: Path to video file
            intended_shot: User's intended shot
            
        Returns:
            Comprehensive analysis with images
        """
        # Process video and classify based on selected mode.
        if self.mode == ANALYZE_SHOT_MODE_LEGACY:
            video_data = self.process_video(video_path)
            ensemble_result = self.ensemble_predict(video_data['features'])
            mistakes = self.mistake_analyzer.analyze_execution(
                intended_shot,
                ensemble_result['final_prediction'],
                video_data['features'],
            )
            actual_keypoints = video_data['contact_keypoints']
            correction_summary = self.mistake_analyzer.generate_correction_summary(
                mistakes, intended_shot
            )
            analysis_method = 'prototype_comparison'
        else:
            video_data = self.process_video_new(video_path)
            ensemble_result = video_data['prediction']
            mistakes = self.analyze_execution_new(
                intended_shot,
                ensemble_result['final_prediction'],
                video_data['features'],
            )
            actual_keypoints = self._build_fallback_keypoints(intended_shot)
            correction_summary = self._generate_correction_summary_new(
                mistakes, intended_shot
            )
            analysis_method = 'efficientnetb4_gru_embedding'

        if self.mode == ANALYZE_SHOT_MODE_NEW:
            print("[EfficientNetMode] Finalizing analysis response", flush=True)
        
        # Calculate intent score
        intent_score = self.calculate_intent_score(
            intended_shot, 
            ensemble_result['ensemble_probabilities']
        )

        # Generate visual feedback (3D avatar ready)
        visual_feedback = self.generate_visual_feedback_for_frontend(
            actual_keypoints,
            intended_shot,
            mistakes
        )
        
        # Generate coaching feedback
        coaching_feedback = self.generate_ai_feedback(
            intended_shot,
            ensemble_result['final_prediction'],
            intent_score,
            mistakes
        )
        
        # Compile results
        result = {
            'intended_shot': intended_shot,
            'predicted_shot': ensemble_result['final_prediction'],
            'intent_score': intent_score,
            'is_correct': ensemble_result['final_prediction'] == intended_shot,
            
            # 3D Avatar Visual Feedback (PRIMARY)
            'visual_feedback': visual_feedback,
            
            # Mistake Analysis
            'mistake_analysis': mistakes,
            'correction_summary': correction_summary,
            'coaching_feedback': coaching_feedback,
            
            # Technical Details
            'ensemble_probabilities': ensemble_result['ensemble_probabilities'],
            'model_predictions': ensemble_result['individual_predictions'],
            
            # Metadata
            'analysis_metadata': {
                'contact_frame': video_data['metadata']['contact_frame'],
                'contact_detection': video_data.get('yolo_detection', {}),
                'prototype_samples': self.prototypes.get(intended_shot, {}).get('n_samples', 0),
                'analysis_method': analysis_method,
                'mode': self.mode,
            }
        }
        
        return to_json_safe(result)
    
    def get_shot_types(self) -> List[str]:
        """Get available shot types"""
        if self.mode == ANALYZE_SHOT_MODE_LEGACY:
            return self.label_encoder.classes_.tolist()
        return list(self.video_classifier.shot_types)


# Global service instances by mode
_batting_services: Dict[int, BattingService] = {}


def get_batting_service(mode: Optional[object] = None) -> BattingService:
    """Get or create service for the requested mode."""
    normalized_mode = _normalize_mode(mode)
    if normalized_mode not in _batting_services:
        _batting_services[normalized_mode] = BattingService(mode=normalized_mode)
    return _batting_services[normalized_mode]