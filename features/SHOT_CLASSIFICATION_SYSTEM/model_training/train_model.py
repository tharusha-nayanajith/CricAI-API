"""
Model Training Script
Trains Random Forest model for cricket shot classification
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.feature_engineer import FeatureEngineer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    SHOT_TYPES, RANDOM_FOREST_PARAMS, MODEL_PATH, SCALER_PATH
)


class ModelTrainer:
    """Train Random Forest model for shot classification"""
    
    def __init__(self):
        self.frame_extractor = FrameExtractor(fps=10)
        self.pose_estimator = PoseEstimator()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = None
    
    def process_video(self, video_path: str) -> np.ndarray:
        """
        Process a single video and extract features
        
        Args:
            video_path: Path to video file
            
        Returns:
            Feature vector
        """
        # Step 1: Extract frames
        frames, _ = self.frame_extractor.extract_frames(video_path)
        
        # Step 2: Estimate pose for each frame
        pose_sequence = self.pose_estimator.estimate_pose_batch(frames)
        
        # Step 3: Extract features (detects shot moment automatically)
        features = self.feature_engineer.extract_features_from_video(pose_sequence)
        
        return features
    
    def prepare_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all videos and create training dataset
        
        Args:
            dataset_path: Path to dataset directory (contains shot type folders)
            
        Returns:
            Tuple of (features, labels)
        """
        X = []  # Features
        y = []  # Labels
        
        for shot_type in SHOT_TYPES:
            shot_dir = os.path.join(dataset_path, shot_type)
            
            if not os.path.exists(shot_dir):
                print(f"Warning: Directory not found: {shot_dir}")
                continue
            
            video_files = [f for f in os.listdir(shot_dir) if f.endswith('.mp4')]
            print(f"\nProcessing {len(video_files)} videos for shot type: {shot_type}")
            
            for idx, video_file in enumerate(video_files):
                video_path = os.path.join(shot_dir, video_file)
                
                try:
                    print(f"Processing {idx + 1}/{len(video_files)}: {video_file}")
                    features = self.process_video(video_path)
                    X.append(features)
                    y.append(shot_type)
                    
                except Exception as e:
                    print(f"Error processing {video_file}: {str(e)}")
                    continue
        
        return np.array(X), np.array(y)
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """
        Train Random Forest model
        
        Args:
            X: Feature matrix
            y: Labels
        """
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save processed data for evaluation
        np.savez('processed_dataset.npz',
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)
        print("Processed dataset saved for evaluation")

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"\nTraining Accuracy: {train_score * 100:.2f}%")
        print(f"Testing Accuracy: {test_score * 100:.2f}%")
        
        # Detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=SHOT_TYPES))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        
        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Scaler saved to: {SCALER_PATH}")


def main():
    """Main training function"""
    from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import DATASET_PATH
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path not found: {DATASET_PATH}")
        print("Please update DATASET_PATH in this script")
        return
    
    trainer = ModelTrainer()
    
    # Process all videos and create dataset
    print("Step 1: Processing videos and extracting features...")
    X, y = trainer.prepare_dataset(DATASET_PATH)
    
    print(f"\nDataset prepared:")
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Shot type distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for shot_type, count in zip(unique, counts):
        print(f"  {shot_type}: {count}")
    
    # Train model
    print("\nStep 2: Training model...")
    trainer.train_model(X, y)
    
    # Save model
    print("\nStep 3: Saving model...")
    trainer.save_model()
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    main()