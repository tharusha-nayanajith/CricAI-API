"""
Data Augmentation for Pose-Based Features
Reduces overfitting by introducing controlled variations
"""

import numpy as np
from typing import List, Tuple


class PoseAugmenter:
    """Augment pose data to improve generalization"""
    
    def __init__(self, augmentation_factor: int = 3):
        """
        Args:
            augmentation_factor: Number of augmented copies per sample
        """
        self.augmentation_factor = augmentation_factor
    
    def add_noise(self, features: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to features"""
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    def scale_features(self, features: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """Randomly scale features slightly"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return features * scale
    
    def temporal_shift(self, features: np.ndarray, shift_std: float = 0.01) -> np.ndarray:
        """Simulate slight temporal variations"""
        # Add small random shifts to velocity features
        # (assuming velocity features are in specific indices)
        shifted = features.copy()
        velocity_indices = list(range(len(features) // 2, len(features)))  # Rough estimate
        
        for idx in velocity_indices:
            shifted[idx] += np.random.normal(0, shift_std)
        
        return shifted
    
    def augment_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment entire dataset
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            (X_augmented, y_augmented)
        """
        augmented_X = [X]  # Original data
        augmented_y = [y]
        
        for _ in range(self.augmentation_factor):
            X_aug = X.copy()
            
            # Apply random combination of augmentations
            for i in range(len(X_aug)):
                # Random choice of augmentations
                if np.random.rand() > 0.5:
                    X_aug[i] = self.add_noise(X_aug[i])
                
                if np.random.rand() > 0.5:
                    X_aug[i] = self.scale_features(X_aug[i])
                
                if np.random.rand() > 0.5:
                    X_aug[i] = self.temporal_shift(X_aug[i])
            
            augmented_X.append(X_aug)
            augmented_y.append(y)
        
        # Concatenate all
        X_final = np.vstack(augmented_X)
        y_final = np.concatenate(augmented_y)
        
        # Shuffle
        indices = np.random.permutation(len(X_final))
        X_final = X_final[indices]
        y_final = y_final[indices]
        
        print(f"Augmentation: {len(X)} → {len(X_final)} samples")
        
        return X_final, y_final