"""
Prototype Extractor
Extracts "correct execution" prototypes from training data
This is what the MODEL learned as correct - not our assumptions
"""

import numpy as np
import joblib
from typing import Dict
from sklearn.preprocessing import LabelEncoder


class PrototypeExtractor:
    """Extract and save shot execution prototypes"""
    
    def extract_prototypes(self, X: np.ndarray, y: np.ndarray, 
                          label_encoder: LabelEncoder) -> Dict[str, np.ndarray]:
        """
        Calculate prototype (average) for each shot type
        
        Args:
            X: Feature matrix (all training data)
            y: Encoded labels
            label_encoder: Label encoder to get class names
            
        Returns:
            Dictionary: {shot_type: prototype_vector}
        """
        print("\n" + "="*70)
        print("EXTRACTING SHOT PROTOTYPES FROM TRAINING DATA")
        print("="*70)
        
        prototypes = {}
        shot_types = label_encoder.classes_
        
        for shot_idx, shot_type in enumerate(shot_types):
            # Get all samples of this shot type
            mask = (y == shot_idx)
            shot_samples = X[mask]
            
            if len(shot_samples) == 0:
                print(f"⚠️  No samples for {shot_type}")
                continue
            
            # Calculate mean (prototype)
            prototype = np.mean(shot_samples, axis=0)
            
            # Calculate std (for later use in error detection)
            std = np.std(shot_samples, axis=0)
            
            prototypes[shot_type] = {
                'mean': prototype,
                'std': std,
                'n_samples': len(shot_samples)
            }
            
            print(f"✓ {shot_type}: prototype from {len(shot_samples)} samples")
        
        print("="*70)
        return prototypes
    
    def extract_feature_importance(self, models: Dict) -> Dict[str, np.ndarray]:
        """
        Extract feature importance from all models
        
        Args:
            models: Dictionary of trained models
            
        Returns:
            Dictionary: {model_name: importance_array}
        """
        print("\nExtracting feature importance from models...")
        
        importance_dict = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_dict[model_name] = importance
                print(f"✓ {model_name}: {len(importance)} features")
        
        return importance_dict
    
    def save_prototypes(self, prototypes: Dict, importance: Dict, save_dir: str):
        """Save prototypes and feature importance"""
        import os
        os.makedirs(f"{save_dir}/prototypes", exist_ok=True)
        
        # Save prototypes
        joblib.dump(prototypes, f"{save_dir}/prototypes/shot_prototypes.pkl")
        print(f"✓ Saved prototypes with keypoints to {save_dir}/prototypes/shot_prototypes.pkl")
        
        # Save feature importance
        joblib.dump(importance, f"{save_dir}/prototypes/feature_importance.pkl")
        print(f"✓ Saved feature importance to {save_dir}/prototypes/feature_importance.pkl")