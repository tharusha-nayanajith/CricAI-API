"""
Model-Based Mistake Analyzer
"""

import numpy as np
from typing import Dict, List, Tuple
import joblib


class ModelBasedMistakeAnalyzer:
    """Analyze mistakes using MODEL'S learned knowledge"""
    
    def __init__(self, prototypes_path: str, feature_importance_path: str,
                 feature_names: List[str]):
        """
        Args:
            prototypes_path: Path to learned shot prototypes
            feature_importance_path: Path to feature importance scores
            feature_names: List of feature names
        """
        # Load learned prototypes (average correct execution for each shot)
        self.prototypes = joblib.load(prototypes_path)
        
        # Load feature importance from model
        self.feature_importance = joblib.load(feature_importance_path)
        
        self.feature_names = feature_names
        
        # Get top features (most important for classification)
        self.top_features = self._get_top_features(top_k=15)
        
        # Map features to body parts
        self.feature_to_bodypart = self._map_features_to_bodyparts()
        
        # Map body parts to joint IDs for visualization
        self.bodypart_to_joint_id = {
            'Front Elbow': 'front_elbow',
            'Back Elbow': 'back_elbow',
            'Front Knee': 'front_knee',
            'Back Knee': 'back_knee',
            'Torso': 'torso_bend',
            'Shoulders': 'shoulder_rotation',
            'Front Wrist': 'front_wrist',
            'Back Wrist': 'back_wrist',
            'Body Position': 'body_position'
        }
    
    def _get_top_features(self, top_k: int = 15) -> List[Tuple[str, float]]:
        """Get most important features from model"""
        # Average importance across all models
        avg_importance = np.mean(list(self.feature_importance.values()), axis=0)
        
        # Get top K
        top_indices = np.argsort(avg_importance)[-top_k:][::-1]
        return [(self.feature_names[i], avg_importance[i]) for i in top_indices]
    
    def _map_features_to_bodyparts(self) -> Dict[str, str]:
        """Map feature names to body parts"""
        mapping = {}
        for feat_name in self.feature_names:
            if 'front_elbow' in feat_name:
                mapping[feat_name] = 'Front Elbow'
            elif 'back_elbow' in feat_name:
                mapping[feat_name] = 'Back Elbow'
            elif 'front_knee' in feat_name:
                mapping[feat_name] = 'Front Knee'
            elif 'back_knee' in feat_name:
                mapping[feat_name] = 'Back Knee'
            elif 'torso' in feat_name:
                mapping[feat_name] = 'Torso'
            elif 'shoulder_rotation' in feat_name or 'shoulder' in feat_name:
                mapping[feat_name] = 'Shoulders'
            elif 'wrist' in feat_name:
                if 'right' in feat_name:
                    mapping[feat_name] = 'Front Wrist'
                else:
                    mapping[feat_name] = 'Back Wrist'
            else:
                mapping[feat_name] = 'Body Position'
        return mapping
    
    def analyze_execution(self, intended_shot: str, actual_shot: str,
                         actual_features: np.ndarray) -> List[Dict]:
        """
        Analyze mistakes using model's learned prototypes
        
        Args:
            intended_shot: User's intended shot
            actual_shot: Model's prediction
            actual_features: User's actual feature vector
            
        Returns:
            List of mistakes with model-based reasoning
        """
        if intended_shot not in self.prototypes:
            return []
        
        # Get correct prototype for intended shot
        prototype = self.prototypes[intended_shot]["features"]["mean"]
        
        # Calculate feature-wise deviations
        feature_diffs = actual_features - prototype
        
        # Get absolute deviations weighted by importance
        mistakes = []
        
        for feat_name, importance in self.top_features:
            feat_idx = self.feature_names.index(feat_name)
            actual_value = actual_features[feat_idx]
            expected_value = prototype[feat_idx]
            deviation = feature_diffs[feat_idx]
            
            # Better severity calculation
            std_dev = self._get_feature_std(intended_shot, feat_idx)
            
            # Normalized deviation (how many standard deviations away)
            if std_dev > 1e-6:
                normalized_deviation = abs(deviation) / std_dev
            else:
                normalized_deviation = abs(deviation)
            
            # Severity score with adjusted scaling
            # Scale importance by 10x for better discrimination
            severity_score = (importance * 10) * normalized_deviation
            
            # LOWERED threshold from 0.1 to 0.01 (more sensitive)
            if severity_score > 0.01:
                severity = self._calculate_severity(severity_score)
                body_part = self.feature_to_bodypart.get(feat_name, 'Body Position')
                
                # Add joint_id for visualization
                joint_id = self.bodypart_to_joint_id.get(body_part, None)
                
                mistake = {
                    'body_part': body_part,
                    'joint_id': joint_id, 
                    'feature_name': feat_name,
                    'severity': severity,
                    'severity_score': float(severity_score),
                    'actual_value': float(actual_value),
                    'expected_value': float(expected_value),
                    'deviation': float(deviation),
                    'importance': float(importance),
                    'explanation': self._generate_explanation(
                        feat_name, deviation, intended_shot, actual_shot
                    ),
                    'recommendation': self._generate_recommendation(
                        feat_name, deviation, intended_shot
                    )
                }
                
                mistakes.append(mistake)
        
        # Sort by severity score (most important first)
        mistakes.sort(key=lambda x: x['severity_score'], reverse=True)
        
        # Return top 5 most significant mistakes
        return mistakes[:5]
    
    def _get_feature_std(self, shot_type: str, feature_idx: int) -> float:
        """Get std dev of feature"""
        return float(self.prototypes[shot_type]["features"]["std"][feature_idx])
    
    def _calculate_severity(self, severity_score: float) -> str:
        """ADJUSTED thresholds for better categorization"""
        if severity_score > 1.5:
            return 'critical'
        elif severity_score > 0.7:
            return 'major'
        elif severity_score > 0.2:
            return 'minor'
        else:
            return 'negligible'
    
    def _generate_explanation(self, feature_name: str, deviation: float,
                             intended_shot: str, actual_shot: str) -> str:
        """Generate explanation"""
        direction = "higher" if deviation > 0 else "lower"
        body_part = self.feature_to_bodypart.get(feature_name, feature_name)
        
        # Context-aware explanations
        if 'angular_change' in feature_name:
            return f"Your {body_part} rotation was {direction} than expected for a {intended_shot}, causing the shot to resemble a {actual_shot}"
        elif 'velocity' in feature_name:
            return f"Your {body_part} moved too {'fast' if deviation > 0 else 'slow'} during execution"
        elif 'contact_' in feature_name:
            if 'elbow' in feature_name or 'knee' in feature_name:
                return f"At contact, your {body_part} angle was too {'straight' if deviation > 0 else 'bent'} for a {intended_shot}"
            else:
                return f"Your {body_part} positioning at contact deviated from ideal {intended_shot} form"
        else:
            return f"Your {body_part} positioning differed from optimal {intended_shot} execution"
    
    def _generate_recommendation(self, feature_name: str, deviation: float, intended_shot: str) -> str:
        """Generate actionable advice"""
        body_part = self.feature_to_bodypart.get(feature_name, feature_name)
        
        recommendations = {
            'Front Elbow': {
                'positive': 'Bend your front arm more at contact. Practice with a mirror.',
                'negative': 'Straighten your front arm. Focus on extending through the ball.'
            },
            'Back Elbow': {
                'positive': 'Keep back elbow higher during backswing for more power.',
                'negative': 'Lower your back elbow slightly. Over-extension reduces control.'
            },
            'Front Knee': {
                'positive': 'Transfer weight forward with a more bent front leg.',
                'negative': 'Straighten front leg more at contact for a firmer base.'
            },
            'Torso': {
                'positive': 'Stay more upright. Excessive bending affects timing.',
                'negative': 'Get lower into your stance for better reach.'
            },
            'Shoulders': {
                'positive': 'Increase shoulder rotation for more bat speed.',
                'negative': 'Control shoulder rotation to avoid losing balance.'
            }
        }
        
        # Find matching recommendation
        for key, recs in recommendations.items():
            if key in body_part:
                return recs['negative'] if deviation > 0 else recs['positive']
        
        # Generic recommendation
        if 'velocity' in feature_name:
            return f"Practice smoother movement. {'Slow down' if deviation > 0 else 'Accelerate'} your {body_part}."
        else:
            return f"Adjust your {body_part} to match correct form. Watch professional {intended_shot} examples."
    
    def generate_correction_summary(self, mistakes: List[Dict], intended_shot: str) -> str:
        """Generate summary"""
        if not mistakes:
            return f"Excellent technique! Your {intended_shot} execution was optimal."
        
        critical = [m for m in mistakes if m['severity'] == 'critical']
        major = [m for m in mistakes if m['severity'] == 'major']
        minor = [m for m in mistakes if m['severity'] == 'minor']
        
        summary_parts = []
        
        if critical:
            summary_parts.append(f"🔴 Critical ({len(critical)}): " + 
                               ", ".join([m['body_part'] for m in critical]))
        if major:
            summary_parts.append(f"🟡 Major ({len(major)}): " + 
                               ", ".join([m['body_part'] for m in major]))
        if minor:
            summary_parts.append(f"🟢 Minor ({len(minor)}): " + 
                               ", ".join([m['body_part'] for m in minor]))
        
        summary = " | ".join(summary_parts)
        
        # Add overall advice
        if critical or major:
            summary += f"\n\nPriority: Focus on {mistakes[0]['body_part']} first."
        
        return summary
    
    def calculate_overall_score(self, intended_shot: str, 
                               actual_features: np.ndarray) -> float:
        """
        Calculate overall execution quality score (0-100)
        Based on distance to learned prototype
        """
        if intended_shot not in self.prototypes:
            return 0.0
        
        prototype = self.prototypes[intended_shot]["features"]["mean"]
        
        # Calculate weighted Euclidean distance
        # Weight by feature importance
        importance_weights = np.array([
            self.feature_importance['random_forest'][self.feature_names.index(f)]
            if f in self.feature_names else 0.0
            for f in self.feature_names
        ])
        
        weighted_diff = (actual_features - prototype) * importance_weights
        distance = np.linalg.norm(weighted_diff)
        
        max_expected_distance = 50.0
        similarity = max(0, 1 - (distance / max_expected_distance))
        score = similarity * 100
        
        return round(score, 2)