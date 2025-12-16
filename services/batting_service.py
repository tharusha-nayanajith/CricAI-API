"""
Batting Service
Implements intent-based shot scoring using Random Forest
"""

import numpy as np
import joblib
import os
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.feature_engineer import FeatureEngineer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import SHOT_TYPES, MODEL_PATH, SCALER_PATH


class BattingService:
    """Service for cricket shot analysis"""
    
    def __init__(self):
        """Initialize service with trained model"""
        # Load model and scaler
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        
        # Initialize processing components
        self.frame_extractor = FrameExtractor(fps=10)
        self.pose_estimator = PoseEstimator()
        self.feature_engineer = FeatureEngineer()
        
        print("Batting service initialized successfully")
    
    def process_video(self, video_path: str) -> np.ndarray:
        """
        Process video and extract features
        
        Args:
            video_path: Path to video file
            
        Returns:
            Feature vector
        """
        # Extract frames
        frames, _ = self.frame_extractor.extract_frames(video_path)
        
        # Estimate poses
        pose_sequence = self.pose_estimator.estimate_pose_batch(frames)
        
        # Extract features (detects shot moment)
        features = self.feature_engineer.extract_features_from_video(pose_sequence)
        
        return features
    
    def calculate_intent_score(self, intended_shot: str, probability_distribution: Dict[str, float]) -> float:
        """
        Calculate intent execution score
        
        Formula: Score = Probability of intended shot / Max probability
        
        Args:
            intended_shot: User's intended shot type
            probability_distribution: Model's probability predictions for all shots
            
        Returns:
            Intent score (0-100)
        """
        intended_prob = probability_distribution.get(intended_shot, 0.0)
        max_prob = max(probability_distribution.values())
        
        # Calculate score
        if max_prob > 0:
            score = (intended_prob / max_prob) * 100
        else:
            score = 0.0
        
        return round(score, 2)
    
    def analyze_shot(self, video_path: str, intended_shot: str) -> Dict:
        """
        Analyze cricket shot with intent-based scoring
        
        Args:
            video_path: Path to video file
            intended_shot: User's intended shot (one of SHOT_TYPES)
            
        Returns:
            Analysis results dictionary
        """
        # Validate intended shot
        if intended_shot not in SHOT_TYPES:
            raise ValueError(f"Invalid shot type. Must be one of: {SHOT_TYPES}")
        
        # Process video
        features = self.process_video(video_path)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        predicted_shot = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Create probability distribution
        probability_distribution = {
            shot_type: float(prob) 
            for shot_type, prob in zip(self.model.classes_, probabilities)
        }
        
        # Calculate intent score
        intent_score = self.calculate_intent_score(intended_shot, probability_distribution)
        
        # Determine feedback
        if intent_score >= 80:
            feedback = "Excellent execution! Your shot matched your intent very well."
        elif intent_score >= 60:
            feedback = "Good execution. Minor adjustments needed to perfect your intended shot."
        elif intent_score >= 40:
            feedback = "Moderate execution. The shot differed from your intent. Focus on technique."
        else:
            feedback = "Poor execution. The shot significantly differed from your intent. Review fundamentals."
        
        # Prepare result
        result = {
            'intended_shot': intended_shot,
            'predicted_shot': predicted_shot,
            'intent_score': intent_score,
            'probability_distribution': probability_distribution,
            'feedback': feedback,
            'is_correct': predicted_shot == intended_shot
        }
        
        return result
    
    def get_shot_types(self) -> list:
        """Get list of available shot types"""
        return SHOT_TYPES


# Global service instance
_batting_service = None

def get_batting_service() -> BattingService:
    """Get or create batting service instance"""
    global _batting_service
    if _batting_service is None:
        _batting_service = BattingService()
    return _batting_service