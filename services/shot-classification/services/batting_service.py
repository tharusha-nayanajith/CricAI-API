"""
Batting Service  — with Shot Outcome Prediction
================================================
Extends the existing BattingService to:
  1. Accept an optional field_setting in analyze_shot()
  2. Extract pre-contact ball positions from contact_metadata
  3. Call ShotOutcomePredictor and attach results to the response
"""

import numpy as np
import joblib
import os
import json
from typing import Dict, List, Optional
from google import genai

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.temporal_feature_engineer import TemporalFeatureEngineer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.model_based_mistake_analyzer import ModelBasedMistakeAnalyzer
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.skeleton_animator import SkeletonAnimator
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import MODEL_FOLDER_PATH
from features.SHOT_CLASSIFICATION_SYSTEM.utils.json_utils import to_json_safe

# ── NEW imports ───────────────────────────────────────────────────────────────
from features.SHOT_CLASSIFICATION_SYSTEM.utils.shot_outcome.shot_outcome_predictor import ShotOutcomePredictor
from features.SHOT_CLASSIFICATION_SYSTEM.utils.shot_outcome.field_schemas import FieldSetting


class BattingService:
    """Advanced batting analysis with 3D avatar support and shot outcome prediction."""

    JOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    SEVERITY_COLORS = {
        'critical':   '#e74c3c',
        'major':      '#f39c12',
        'minor':      '#f1c40f',
        'negligible': '#95a5a6'
    }

    def __init__(self, model_dir: str = MODEL_FOLDER_PATH):
        self.model_dir = model_dir

        self.models        = self._load_models()
        self.scaler        = joblib.load(f"{model_dir}/ensemble/scaler.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/ensemble/label_encoder.pkl")

        with open(f"{model_dir}/ensemble/feature_names.json", 'r') as f:
            self.feature_names = json.load(f)

        self.prototypes = joblib.load(f"{model_dir}/prototypes/shot_prototypes.pkl")

        self.frame_extractor  = FrameExtractor(fps=10)
        self.pose_estimator   = PoseEstimator()
        self.feature_engineer = TemporalFeatureEngineer()

        self.mistake_analyzer = ModelBasedMistakeAnalyzer(
            prototypes_path=f"{model_dir}/prototypes/shot_prototypes.pkl",
            feature_importance_path=f"{model_dir}/prototypes/feature_importance.pkl",
            feature_names=self.feature_names
        )
        self.skeleton_animator = SkeletonAnimator()

        # ── Shot outcome predictor (rules-based, no extra model needed) ───────
        self.outcome_predictor = ShotOutcomePredictor()

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.ai_client = genai.Client(api_key=api_key) if api_key else None

        print("✓ Batting service initialized (with shot outcome predictor)")

    # ──────────────────────────────────────────────────────────────────────────
    # Existing methods — unchanged
    # ──────────────────────────────────────────────────────────────────────────

    def _load_models(self) -> Dict:
        models = {}
        for model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
            model_path = f"{self.model_dir}/{model_name}/model_latest.pkl"
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        return models

    def process_video(self, video_path: str) -> Dict:
        frames, fps   = self.frame_extractor.extract_frames(video_path)
        pose_sequence = self.pose_estimator.estimate_pose_batch(frames)
        features, metadata = self.feature_engineer.extract_temporal_features(
            pose_sequence, frames
        )
        contact_frame_idx = metadata['contact_frame']
        contact_frame     = frames[contact_frame_idx]
        contact_pose      = pose_sequence[contact_frame_idx]

        return {
            'features':          features,
            'metadata':          metadata,
            'frames':            frames,           # ← kept for trajectory extraction
            'pose_sequence':     pose_sequence,    # ← kept for pose at contact
            'contact_frame':     contact_frame,
            'contact_keypoints': contact_pose['keypoints'],
            'contact_scores':    contact_pose['scores'],
            'contact_pose':      contact_pose,     # ← full pose dict at contact
            'yolo_detection':    metadata.get('contact_detection', {})
        }

    def ensemble_predict(self, features: np.ndarray) -> Dict:
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        predictions  = {}
        probabilities = {}
        for name, model in self.models.items():
            pred  = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            predictions[name]   = self.label_encoder.inverse_transform([pred])[0]
            probabilities[name] = {
                shot: float(prob)
                for shot, prob in zip(self.label_encoder.classes_, proba)
            }
        ensemble_proba = {}
        for shot_class in self.label_encoder.classes_:
            avg_prob = np.mean([probabilities[m][shot_class] for m in self.models])
            ensemble_proba[shot_class] = float(avg_prob)
        final_prediction = max(ensemble_proba, key=ensemble_proba.get)
        return {
            'final_prediction':       final_prediction,
            'ensemble_probabilities': ensemble_proba,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
        }

    def calculate_intent_score(self, intended_shot: str, ensemble_proba: Dict) -> float:
        intended_prob = ensemble_proba.get(intended_shot, 0.0)
        max_prob      = max(ensemble_proba.values())
        score = (intended_prob / max_prob) * 100 if max_prob > 0 else 0.0
        return round(score, 2)

    def convert_to_3d_keypoints(self, keypoints_2d: np.ndarray) -> List[Dict]:
        keypoints_3d = []
        for i, joint_name in enumerate(self.JOINT_NAMES):
            x, y = keypoints_2d[i]
            if   i < 7:          z = 50
            elif i in [7, 8]:    z = 30
            elif i in [9, 10]:   z = 80
            elif i in [11, 12]:  z = 0
            else:                z = -20
            keypoints_3d.append({
                'joint': joint_name, 'index': i,
                'position': {'x': float(x), 'y': float(-y), 'z': float(z)}
            })
        return keypoints_3d

    def prepare_mistake_visualization(self, mistakes: List[Dict]) -> List[Dict]:
        viz = []
        for m in mistakes:
            if not m.get('joint_id'):
                continue
            severity = m['severity']
            color    = self.SEVERITY_COLORS.get(severity, '#95a5a6')
            intensity = min(1.0, m['severity_score'] / 2.0)
            viz.append({
                'joint_id':       m['joint_id'],
                'body_part':      m['body_part'],
                'severity':       severity,
                'severity_color': color,
                'glow_intensity': float(intensity),
                'explanation':    m['explanation'],
                'recommendation': m['recommendation'],
            })
        return viz

    def generate_visual_feedback_for_frontend(
        self,
        actual_keypoints: np.ndarray,
        intended_shot: str,
        mistakes: List[Dict],
    ) -> Dict:
        actual_3d    = self.convert_to_3d_keypoints(actual_keypoints)
        if intended_shot in self.prototypes:
            proto_kp = self.prototypes[intended_shot]['keypoints']['mean']
            proto_3d = self.convert_to_3d_keypoints(proto_kp)
        else:
            proto_3d = actual_3d
        mistake_viz = self.prepare_mistake_visualization(mistakes)
        return {
            'keypoints_3d': {
                'actual':    actual_3d,
                'prototype': proto_3d,
                'format':    'three_js_compatible',
            },
            'mistakes':          mistake_viz,
            'joint_connections': self._get_skeleton_connections(),
            'prototype_used':    intended_shot,
            'prototype_samples': self.prototypes.get(intended_shot, {}).get('n_samples', 0),
        }

    def _get_skeleton_connections(self) -> List[Dict]:
        return [
            {'from': 'nose',           'to': 'left_eye',       'label': 'head'},
            {'from': 'nose',           'to': 'right_eye',      'label': 'head'},
            {'from': 'left_eye',       'to': 'left_ear',       'label': 'head'},
            {'from': 'right_eye',      'to': 'right_ear',      'label': 'head'},
            {'from': 'left_shoulder',  'to': 'right_shoulder', 'label': 'torso'},
            {'from': 'left_shoulder',  'to': 'left_hip',       'label': 'torso'},
            {'from': 'right_shoulder', 'to': 'right_hip',      'label': 'torso'},
            {'from': 'left_hip',       'to': 'right_hip',      'label': 'torso'},
            {'from': 'right_shoulder', 'to': 'right_elbow',    'label': 'right_arm'},
            {'from': 'right_elbow',    'to': 'right_wrist',    'label': 'right_arm'},
            {'from': 'left_shoulder',  'to': 'left_elbow',     'label': 'left_arm'},
            {'from': 'left_elbow',     'to': 'left_wrist',     'label': 'left_arm'},
            {'from': 'right_hip',      'to': 'right_knee',     'label': 'right_leg'},
            {'from': 'right_knee',     'to': 'right_ankle',    'label': 'right_leg'},
            {'from': 'left_hip',       'to': 'left_knee',      'label': 'left_leg'},
            {'from': 'left_knee',      'to': 'left_ankle',     'label': 'left_leg'},
        ]

    def generate_ai_feedback(
        self,
        intended_shot: str,
        predicted_shot: str,
        intent_score: float,
        mistakes: List[Dict],
        outcome: Optional[Dict] = None,   # ← NEW: outcome data
    ) -> str:
        if not self.ai_client:
            return self._generate_rule_based_feedback(
                intended_shot, predicted_shot, intent_score, mistakes
            )
        try:
            mistake_summary = "\n".join([
                f"- {m['body_part']}: {m['explanation']}"
                for m in mistakes[:3]
            ])

            # Add outcome context to prompt if available
            outcome_context = ""
            if outcome:
                outcome_context = (
                    f"\nPredicted Shot Outcome: {outcome.get('outcome', 'unknown')} "
                    f"({outcome.get('runs', 0)} runs), "
                    f"Power: {outcome.get('power_rating', 0):.0%}, "
                    f"Timing: {outcome.get('timing_rating', 0):.0%}"
                )

            prompt = f"""You are an expert cricket coach analyzing a player's {intended_shot} execution.

Player's Intent: {intended_shot.upper()}
Actual Execution: {predicted_shot.upper()}
Intent Score: {intent_score}% (similarity to {self.prototypes[intended_shot]['n_samples']} correct {intended_shot} examples)
{outcome_context}

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
            return self._generate_rule_based_feedback(
                intended_shot, predicted_shot, intent_score, mistakes
            )

    def _generate_rule_based_feedback(
        self, intended_shot, predicted_shot, intent_score, mistakes
    ) -> str:
        n = self.prototypes.get(intended_shot, {}).get('n_samples', 0)
        if intent_score >= 85:
            base = f"Excellent {intended_shot}! Your biomechanics match our {n} reference examples."
        elif intent_score >= 70:
            base = f"Good {intended_shot} attempt. Your execution is close to the learned prototype."
        elif intent_score >= 50:
            base = f"Your {intended_shot} deviates from the {n} training examples."
        else:
            base = f"Significant deviation from correct {intended_shot} form (appeared as {predicted_shot})."
        if mistakes:
            top = mistakes[0]
            return f"{base} Main issue: {top['explanation']} {top['recommendation']}"
        return base

    # ──────────────────────────────────────────────────────────────────────────
    # Main analysis — updated to accept field_setting
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_shot(
        self,
        video_path: str,
        intended_shot: str,
        field_setting: Optional[FieldSetting] = None,   # ← NEW parameter
    ) -> Dict:
        """
        Complete shot analysis with visual feedback and outcome prediction.

        Args:
            video_path:    Path to video file.
            intended_shot: User's intended shot.
            field_setting: Optional FieldSetting object from the API request.
                           If None, a default field is used for outcome prediction.

        Returns:
            Comprehensive analysis dict including 'shot_outcome'.
        """
        # ── 1. Process video ─────────────────────────────────────────────────
        video_data = self.process_video(video_path)

        # ── 2. Ensemble prediction ───────────────────────────────────────────
        ensemble_result = self.ensemble_predict(video_data['features'])

        # ── 3. Intent score ──────────────────────────────────────────────────
        intent_score = self.calculate_intent_score(
            intended_shot,
            ensemble_result['ensemble_probabilities']
        )

        # ── 4. Mistake analysis ──────────────────────────────────────────────
        mistakes = self.mistake_analyzer.analyze_execution(
            intended_shot,
            ensemble_result['final_prediction'],
            video_data['features']
        )

        # ── 5. Shot outcome prediction ───────────────────────────────────────
        contact_metadata = video_data['metadata'].get('contact_detection', {})

        # Mirror field for left-handers if needed
        field_for_predictor = None
        if field_setting is not None:
            adjusted = field_setting.mirror_for_left_hander()
            field_for_predictor = adjusted.to_dict_list()

        # Extract pre-contact ball positions from Kalman tracker history
        # (stored as pixel coords in metadata by TemporalFeatureEngineer)
        pre_contact_positions = video_data['metadata'].get('ball_positions_before_contact', None)

        shot_outcome = self.outcome_predictor.predict(
            shot_type=ensemble_result['final_prediction'],
            contact_metadata=contact_metadata,
            pose_data=video_data['contact_pose'],
            field_setting=field_for_predictor,
            pre_contact_ball_positions=pre_contact_positions,
            intent_score=intent_score,
        )

        # ── 6. Visual feedback ───────────────────────────────────────────────
        visual_feedback = self.generate_visual_feedback_for_frontend(
            video_data['contact_keypoints'],
            intended_shot,
            mistakes
        )

        # ── 7. AI coaching feedback (now includes outcome) ───────────────────
        coaching_feedback = self.generate_ai_feedback(
            intended_shot,
            ensemble_result['final_prediction'],
            intent_score,
            mistakes,
            outcome=shot_outcome,   # ← pass outcome for richer prompt
        )

        correction_summary = self.mistake_analyzer.generate_correction_summary(
            mistakes, intended_shot
        )

        # ── 8. Compile result ────────────────────────────────────────────────
        result = {
            'intended_shot':   intended_shot,
            'predicted_shot':  ensemble_result['final_prediction'],
            'intent_score':    intent_score,
            'is_correct':      ensemble_result['final_prediction'] == intended_shot,

            # ── Shot outcome (NEW) ────────────────────────────────────────────
            'shot_outcome': shot_outcome,

            # ── 3D Avatar Visual Feedback ─────────────────────────────────────
            'visual_feedback':    visual_feedback,

            # ── Mistake Analysis ──────────────────────────────────────────────
            'mistake_analysis':   mistakes,
            'correction_summary': correction_summary,
            'coaching_feedback':  coaching_feedback,

            # ── Technical Details ─────────────────────────────────────────────
            'ensemble_probabilities': ensemble_result['ensemble_probabilities'],
            'model_predictions':      ensemble_result['individual_predictions'],

            # ── Metadata ─────────────────────────────────────────────────────
            'analysis_metadata': {
                'contact_frame':    video_data['metadata']['contact_frame'],
                'contact_detection': video_data.get('yolo_detection', {}),
                'prototype_samples': self.prototypes.get(intended_shot, {}).get('n_samples', 0),
                'analysis_method':  'prototype_comparison',
                'field_setting_used': field_for_predictor is not None,
            }
        }

        return to_json_safe(result)

    def get_shot_types(self) -> List[str]:
        return self.label_encoder.classes_.tolist()


# ── Global singleton ──────────────────────────────────────────────────────────
_batting_service = None

def get_batting_service() -> BattingService:
    global _batting_service
    if _batting_service is None:
        _batting_service = BattingService()
    return _batting_service