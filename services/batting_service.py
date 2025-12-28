"""
Complete Batting Service - Research Ready
Uses actual prototype keypoints for visual feedback generation
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
    """Advanced batting analysis service with ensemble models and visual feedback"""
    
    def __init__(self, model_dir: str = MODEL_FOLDER_PATH):
        self.model_dir = model_dir
        
        # Load ensemble models
        self.models = self._load_models()
        self.scaler = joblib.load(f"{model_dir}/ensemble/scaler.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/ensemble/label_encoder.pkl")
        
        with open(f"{model_dir}/ensemble/feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
        
        # Load prototypes (includes keypoints)
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
        
        # Initialize AI feedback (optional)
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.ai_client = genai.Client(api_key=api_key) if api_key else None
        
        print("✓ Batting service initialized with prototype keypoints")
    
    def _load_models(self) -> Dict:
        """Load all trained models"""
        models = {}
        for model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
            model_path = f"{self.model_dir}/{model_name}/model_latest.pkl"
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"✓ Loaded {model_name}")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
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
            'contact_scores': contact_pose['scores']
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
        """Calculate intent execution score"""
        intended_prob = ensemble_proba.get(intended_shot, 0.0)
        max_prob = max(ensemble_proba.values())
        
        if max_prob > 0:
            score = (intended_prob / max_prob) * 100
        else:
            score = 0.0
        
        return round(score, 2)
    
    def generate_visual_feedback(self, actual_keypoints: np.ndarray, 
                                actual_scores: np.ndarray,
                                intended_shot: str,
                                mistakes: List[Dict]) -> Dict:
        """
        Generate complete visual feedback with ACTUAL prototype keypoints
        """
        # Get prototype keypoints for intended shot
        if intended_shot in self.prototypes:
            prototype_keypoints = self.prototypes[intended_shot]['keypoints']['mean']
        else:
            # Fallback: use actual keypoints
            prototype_keypoints = actual_keypoints
            print(f"⚠️  No prototype found for {intended_shot}, using actual pose")
        
        # 1. Generate 3D skeleton (actual execution with errors)
        skeleton_3d = self.skeleton_animator.generate_3d_skeleton(
            actual_keypoints,
            mistakes,
            view_angle=(30, 45)
        )
        
        # 2. Generate comparison view (ACTUAL vs PROTOTYPE)
        comparison_view = self.skeleton_animator.generate_comparison_view(
            actual_keypoints,      # User's actual pose
            prototype_keypoints,   # CORRECT: Learned prototype from training data
            mistakes
        )
        
        # 3. Generate 360° animation
        animation_360 = self.skeleton_animator.generate_multi_angle_animation(
            actual_keypoints,
            mistakes
        )
        
        return {
            'skeleton_3d': skeleton_3d,
            'comparison_view': comparison_view,
            'animation_360': animation_360,
            'prototype_used': intended_shot,
            'prototype_samples': self.prototypes[intended_shot]['n_samples'] if intended_shot in self.prototypes else 0
        }
    
    def generate_ai_feedback(self, intended_shot: str, predicted_shot: str,
                           intent_score: float, mistakes: List[Dict]) -> str:
        """Generate AI-powered coaching feedback"""
        if not self.ai_client:
            return self._generate_rule_based_feedback(intended_shot, predicted_shot, 
                                                      intent_score, mistakes)
        
        try:
            # Prepare mistake summary
            mistake_summary = "\n".join([
                f"- {m['body_part']}: {m['explanation']}"
                for m in mistakes[:3]
            ])
            
            prompt = f"""You are an expert cricket batting coach analyzing a player's shot execution.

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
        
        # Get ensemble predictions
        ensemble_result = self.ensemble_predict(video_data['features'])
        
        # Calculate intent score
        intent_score = self.calculate_intent_score(
            intended_shot, 
            ensemble_result['ensemble_probabilities']
        )
        
        # Analyze mistakes (model-based)
        mistakes = self.mistake_analyzer.analyze_execution(
            intended_shot,
            ensemble_result['final_prediction'],
            video_data['features']
        )
        
        # Generate visual feedback (with ACTUAL prototypes)
        visual_feedback = self.generate_visual_feedback(
            video_data['contact_keypoints'],
            video_data['contact_scores'],
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
            
            # Visual feedback (3D skeletons, NOT real frames)
            'visual_feedback': visual_feedback,
            
            # Model-based analysis (NOT hardcoded angles)
            'mistake_analysis': mistakes,
            'correction_summary': correction_summary,
            'coaching_feedback': coaching_feedback,
            
            # Technical details
            'ensemble_probabilities': ensemble_result['ensemble_probabilities'],
            'model_predictions': ensemble_result['individual_predictions'],
            
            # Transparency
            'analysis_metadata': {
                'contact_frame': video_data['metadata']['contact_frame'],
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
    """Get or create batting service instance"""
    global _batting_service
    if _batting_service is None:
        _batting_service = BattingService()
    return _batting_service