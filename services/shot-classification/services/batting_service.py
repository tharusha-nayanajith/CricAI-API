"""
Batting Service
"""

import numpy as np
import joblib
import os
import json
from typing import Dict, List
from google import genai

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.temporal_feature_engineer import TemporalFeatureEngineer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.model_based_mistake_analyzer import ModelBasedMistakeAnalyzer
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.skeleton_animator import SkeletonAnimator
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    MODEL_FOLDER_PATH
)
from features.SHOT_CLASSIFICATION_SYSTEM.utils.json_utils import to_json_safe


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
    
    def __init__(self, model_dir: str = MODEL_FOLDER_PATH):
        self.model_dir = model_dir
        
        # Load models
        self.models = self._load_models()
        self.scaler = joblib.load(f"{model_dir}/ensemble/scaler.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/ensemble/label_encoder.pkl")
        
        with open(f"{model_dir}/ensemble/feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
        
        # Load prototypes
        self.prototypes = joblib.load(f"{model_dir}/prototypes/shot_prototypes.pkl")
        
        # Initialize components
        self.frame_extractor = FrameExtractor(fps=10)
        self.pose_estimator = PoseEstimator()
        self.feature_engineer = TemporalFeatureEngineer()
        
        # Initialize analyzers
        self.mistake_analyzer = ModelBasedMistakeAnalyzer(
            prototypes_path=f"{model_dir}/prototypes/shot_prototypes.pkl",
            feature_importance_path=f"{model_dir}/prototypes/feature_importance.pkl",
            feature_names=self.feature_names
        )
        
        self.skeleton_animator = SkeletonAnimator()
        
        # AI feedback
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.ai_client = genai.Client(api_key=api_key) if api_key else None
        
        print("✓ Batting service initialized with prototype keypoints")
    
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
        # Process video
        video_data = self.process_video(video_path)
        
        # Get predictions
        ensemble_result = self.ensemble_predict(video_data['features'])
        
        # Calculate intent score
        intent_score = self.calculate_intent_score(
            intended_shot, 
            ensemble_result['ensemble_probabilities']
        )
        
        # Analyze mistakes
        mistakes = self.mistake_analyzer.analyze_execution(
            intended_shot,
            ensemble_result['final_prediction'],
            video_data['features']
        )
        
        # Generate visual feedback (3D avatar ready)
        visual_feedback = self.generate_visual_feedback_for_frontend(
            video_data['contact_keypoints'],
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
        # Get correction summary
        correction_summary = self.mistake_analyzer.generate_correction_summary(
            mistakes, intended_shot
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
                'analysis_method': 'prototype_comparison'
            }
        }
        
        return to_json_safe(result)
    
    def get_shot_types(self) -> List[str]:
        """Get available shot types"""
        return self.label_encoder.classes_.tolist()


# Global service instance
_batting_service = None

def get_batting_service() -> BattingService:
    """Get or create service"""
    global _batting_service
    if _batting_service is None:
        _batting_service = BattingService()
    return _batting_service