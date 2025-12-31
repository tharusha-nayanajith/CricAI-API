"""
Keypoint Prototype Extractor
Saves BOTH keypoints AND features for visualization
"""

import numpy as np
import joblib
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder


class KeypointPrototypeExtractor:
    """Extract keypoint prototypes for visualization"""
    
    def extract_prototypes(self, X_features: np.ndarray, y: np.ndarray,
                          keypoints_list: List[np.ndarray],
                          metadata_list: List[Dict],
                          label_encoder: LabelEncoder) -> Dict:
        """
        Extract prototypes with BOTH features and keypoints
        
        Args:
            X_features: Feature matrix (n_samples, n_features)
            y: Encoded labels
            keypoints_list: List of keypoint arrays (n_samples, 17, 2)
            metadata_list: List of metadata dicts with angles, positions, etc.
            label_encoder: Label encoder
            
        Returns:
            Dictionary with prototypes for each shot
        """
        print("\n" + "="*70)
        print("EXTRACTING SHOT PROTOTYPES (Features + Keypoints)")
        print("="*70)
        
        prototypes = {}
        shot_types = label_encoder.classes_
        
        for shot_idx, shot_type in enumerate(shot_types):
            mask = (y == shot_idx)
            
            # Feature prototype
            shot_features = X_features[mask]
            feature_mean = np.mean(shot_features, axis=0)
            feature_std = np.std(shot_features, axis=0)
            
            # Keypoint prototype (average pose at contact)
            shot_keypoints = [keypoints_list[i] for i in range(len(y)) if mask[i]]
            keypoint_mean = np.mean(shot_keypoints, axis=0)
            keypoint_std = np.std(shot_keypoints, axis=0)
            
            # Metadata averages (for angles, velocities, etc.)
            shot_metadata = [metadata_list[i] for i in range(len(y)) if mask[i]]
            avg_metadata = self._average_metadata(shot_metadata)
            
            prototypes[shot_type] = {
                'features': {
                    'mean': feature_mean,
                    'std': feature_std
                },
                'keypoints': {
                    'mean': keypoint_mean,  # (17, 2) - VISUALIZATION-READY
                    'std': keypoint_std
                },
                'metadata': avg_metadata,
                'n_samples': len(shot_keypoints)
            }
            
            print(f"✓ {shot_type}: {len(shot_keypoints)} samples")
            print(f"    Feature dim: {feature_mean.shape}")
            print(f"    Keypoints: {keypoint_mean.shape}")
        
        print("="*70)
        return prototypes
    
    def _average_metadata(self, metadata_list: List[Dict]) -> Dict:
        """Average all angles and velocities"""
        if not metadata_list:
            return {}
        
        # Get all keys from first metadata
        keys = []
        for meta in metadata_list:
            if 'angles' in meta:
                keys.extend([f"angle_{k}" for k in meta['angles'].keys()])
            if 'velocities' in meta:
                keys.extend([f"velocity_{k}" for k in meta['velocities'].keys()])
        
        keys = list(set(keys))
        
        avg_meta = {}
        for key in keys:
            values = []
            for meta in metadata_list:
                if key.startswith('angle_'):
                    angle_key = key.replace('angle_', '')
                    if 'angles' in meta and angle_key in meta['angles']:
                        values.append(meta['angles'][angle_key])
                elif key.startswith('velocity_'):
                    vel_key = key.replace('velocity_', '')
                    if 'velocities' in meta and vel_key in meta['velocities']:
                        values.append(meta['velocities'][vel_key])
            
            if values:
                avg_meta[key] = float(np.mean(values))
        
        return avg_meta