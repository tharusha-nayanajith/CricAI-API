"""
Ensemble Model Trainer
Trains RF + XGBoost + GradientBoosting with proper retraining support
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import Dict, Tuple, List
import json
from datetime import datetime
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    SHOT_TYPES, RANDOM_FOREST_PARAMS, MODEL_FOLDER_PATH, DATASET_PATH, SUPPORTED_VIDEO_EXTENSIONS
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.temporal_feature_engineer import TemporalFeatureEngineer
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.data_augmentation import PoseAugmenter
from features.SHOT_CLASSIFICATION_SYSTEM.utils.prototype_extractor import PrototypeExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.utils.keypoint_prototype_extractor import KeypointPrototypeExtractor


class EnsembleTrainer:
    """Train and manage ensemble of models"""
    
    def __init__(self, model_dir: str = MODEL_FOLDER_PATH):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        # Storage for keypoints during dataset preparation
        self.keypoints_cache = []
        self.metadata_cache = []
        
        # Create model directories
        os.makedirs(f"{model_dir}/random_forest", exist_ok=True)
        os.makedirs(f"{model_dir}/xgboost", exist_ok=True)
        os.makedirs(f"{model_dir}/gradient_boosting", exist_ok=True)
        os.makedirs(f"{model_dir}/ensemble", exist_ok=True)
        
        # Initialize components
        self.frame_extractor = FrameExtractor(fps=10)
        self.pose_estimator = PoseEstimator()
        self.feature_engineer = TemporalFeatureEngineer()
        self.augmenter = PoseAugmenter(augmentation_factor=2)
        self.prototype_extractor = PrototypeExtractor()
        self.keypoint_extractor = KeypointPrototypeExtractor()
    
    def process_video(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Process video and extract temporal features with YOLO
        
        Returns:
            (features, metadata, contact_keypoints)
        """
        # Extract frames
        frames, _ = self.frame_extractor.extract_frames(video_path)
        
        # Get poses
        pose_sequence = self.pose_estimator.estimate_pose_batch(frames)
        
        # Extract temporal features
        features, metadata = self.feature_engineer.extract_temporal_features(
            pose_sequence,
            frames 
        )
        
        # Get contact frame keypoints for prototype extraction
        contact_idx = metadata['contact_frame']
        contact_keypoints = pose_sequence[contact_idx]['keypoints']
        
        return features, metadata, contact_keypoints
    
    def prepare_dataset(self, dataset_path: str, shot_types: List[str], 
                       use_augmentation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all videos and create dataset
        
        Args:
            dataset_path: Path to dataset directory
            shot_types: List of shot type names
            use_augmentation: Whether to augment data
            
        Returns:
            (X, y) features and labels
        """
        X = []
        y = []

        # Clear caches
        self.keypoints_cache = []
        self.metadata_cache = []
        
        for shot_type in shot_types:
            shot_dir = os.path.join(dataset_path, shot_type)
            
            if not os.path.exists(shot_dir):
                print(f"Warning: Directory not found: {shot_dir}")
                continue
            
            video_files = [
                f for f in os.listdir(shot_dir)
                if f.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS)
            ]
            print(f"\nProcessing {len(video_files)} videos for {shot_type}...")
            
            for idx, video_file in enumerate(video_files):
                video_path = os.path.join(shot_dir, video_file)
                
                try:
                    if (idx + 1) % 10 == 0:
                        print(f"  {idx + 1}/{len(video_files)} completed")
                    
                    features, metadata, contact_keypoints = self.process_video(video_path)
                    X.append(features)
                    y.append(shot_type)

                    # Cache keypoints and metadata
                    self.keypoints_cache.append(contact_keypoints)
                    self.metadata_cache.append(metadata)
                    
                except Exception as e:
                    print(f"  Error processing {video_file}: {str(e)}")
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Apply data augmentation if enabled
        if use_augmentation:
            print(f"\nApplying data augmentation...")
            X, y = self.augmenter.augment_batch(X, y)

            # Replicate keypoints for augmented samples
            original_count = len(self.keypoints_cache)
            for _ in range(self.augmenter.augmentation_factor):
                self.keypoints_cache.extend(self.keypoints_cache[:original_count])
                self.metadata_cache.extend(self.metadata_cache[:original_count])
        
        return X, y
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("\n" + "="*70)
        print("Training Random Forest Model")
        print("="*70)
        
        rf_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        
        rf_model.fit(X_train, y_train)
        
        return rf_model
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        print("\n" + "="*70)
        print("Training XGBoost Model")
        print("="*70)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(X_train, y_train)
        
        return xgb_model
    
    def train_gradient_boosting(self, X_train, y_train) -> GradientBoostingClassifier:
        """Train Gradient Boosting model"""
        print("\n" + "="*70)
        print("Training Gradient Boosting Model")
        print("="*70)
        
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        
        return gb_model
    
    def train_all_models(self, X, y, test_size=0.2):
        """Train all models in ensemble"""
        print(f"\n{'='*70}")
        print("ENSEMBLE TRAINING PIPELINE")
        print(f"{'='*70}")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")
        print(f"Classes: {np.unique(y)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature names
        self.feature_names = self.feature_engineer.get_feature_names()
        
        # Train models
        self.models['random_forest'] = self.train_random_forest(X_train_scaled, y_train)
        self.models['xgboost'] = self.train_xgboost(X_train_scaled, y_train)
        self.models['gradient_boosting'] = self.train_gradient_boosting(X_train_scaled, y_train)
        
        # Evaluate each model
        results = {}
        for name, model in self.models.items():
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
            results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
            print(f"\n{name.upper()}:")
            print(f"  Train Accuracy: {train_acc*100:.2f}%")
            print(f"  Test Accuracy: {test_acc*100:.2f}%")
        
        # Extract and save prototypes WITH KEYPOINTS
        print("\nExtracting prototypes with keypoints...")
        # Combine train and test indices
        all_indices = np.concatenate([
            np.where(np.isin(np.arange(len(X)), np.arange(len(X_train))))[0],
            np.where(np.isin(np.arange(len(X)), np.arange(len(X_train), len(X))))[0]
        ])
        
        # Get corresponding keypoints and metadata
        all_keypoints = [self.keypoints_cache[i] for i in all_indices]
        all_metadata = [self.metadata_cache[i] for i in all_indices]
        all_labels    = self.label_encoder.transform(y[all_indices])
        
        # Extract prototypes
        prototypes = self.keypoint_extractor.extract_prototypes(
            X[all_indices],
            all_labels,
            all_keypoints,
            all_metadata,
            self.label_encoder
        )
        importance = self.prototype_extractor.extract_feature_importance(self.models)
        self.prototype_extractor.save_prototypes(prototypes, importance, self.model_dir)

        # Save models
        self.save_models()
        
        # Save dataset for future retraining
        self._save_processed_data(X_train, X_test, y_train, y_test)
        
        return results
    
    def save_models(self):
        """Save all models and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each model
        for name, model in self.models.items():
            model_path = f"{self.model_dir}/{name}/model_{timestamp}.pkl"
            joblib.dump(model, model_path)
            
            # Also save as "latest"
            latest_path = f"{self.model_dir}/{name}/model_latest.pkl"
            joblib.dump(model, latest_path)
            
            print(f"✓ Saved {name} model to {model_path}")
        
        # Save scaler
        scaler_path = f"{self.model_dir}/ensemble/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save label encoder
        encoder_path = f"{self.model_dir}/ensemble/label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save feature names
        feature_path = f"{self.model_dir}/ensemble/feature_names.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'num_features': len(self.feature_names),
            'classes': self.label_encoder.classes_.tolist(),
            'models': list(self.models.keys())
        }
        
        metadata_path = f"{self.model_dir}/ensemble/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ All models and metadata saved to {self.model_dir}")
    
    def _save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data for retraining"""
        data_path = f"{self.model_dir}/ensemble/processed_dataset.npz"
        np.savez(data_path,
                 X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test)
        print(f"✓ Processed dataset saved for retraining")
    
    def retrain_models(self, new_data_path: str = None, shot_types: List[str] = None):
        """
        Retrain models with new data
        
        Args:
            new_data_path: Path to new dataset (optional, uses existing if None)
            shot_types: Shot types to include
        """
        print("\n" + "="*70)
        print("RETRAINING ENSEMBLE MODELS")
        print("="*70)
        
        # Load existing processed data
        existing_data_path = f"{self.model_dir}/ensemble/processed_dataset.npz"
        
        if os.path.exists(existing_data_path):
            print("Loading existing processed data...")
            data = np.load(existing_data_path)
            X_existing = np.vstack([data['X_train'], data['X_test']])
            y_existing = np.concatenate([data['y_train'], data['y_test']])
            print(f"Existing data: {len(X_existing)} samples")
        else:
            X_existing = np.array([])
            y_existing = np.array([])
            print("No existing data found, starting fresh")
        
        # Process new data if provided
        if new_data_path and shot_types:
            print(f"\nProcessing new data from {new_data_path}...")
            X_new, y_new = self.prepare_dataset(new_data_path, shot_types, use_augmentation=True)
            
            # Combine with existing data
            if len(X_existing) > 0:
                X_combined = np.vstack([X_existing, X_new])
                y_combined = np.concatenate([y_existing, y_new])
            else:
                X_combined = X_new
                y_combined = y_new
            
            print(f"\nCombined dataset: {len(X_combined)} samples")
        else:
            X_combined = X_existing
            y_combined = y_existing
        
        # Train with combined data
        self.train_all_models(X_combined, y_combined)
        
        print("\n✓ Retraining complete!")


def main():
    """Main training function"""
    
    trainer = EnsembleTrainer()
    
    # Prepare dataset
    print("Preparing dataset...")
    X, y = trainer.prepare_dataset(DATASET_PATH, SHOT_TYPES, use_augmentation=True)
    
    # Train models
    results = trainer.train_all_models(X, y)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()