"""
Video-based Shot Classification Trainer
=======================================
Proven approach: EfficientNetB4 + TimeDistributed + GRU
- Processes 30 frames per video (NO contact frame detection)
- TimeDistributed CNN applied to each frame
- GRU captures temporal dependencies
- Extracts features for similarity/prototype analysis
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import re
import glob
import ctypes
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import Dict, Tuple, List
from datetime import datetime
import pathlib

from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    SHOT_TYPES, MODEL_FOLDER_PATH, DATASET_PATH, SUPPORTED_VIDEO_EXTENSIONS
)

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class VideoClassifierTrainer:
    """
    Training pipeline for video-based shot classification
    Uses EfficientNetB4 with temporal modeling via GRU
    """
    
    # ════════════════════════════════════════════════════════════════
    # FRAME EXTRACTION PARAMETERS
    # ════════════════════════════════════════════════════════════════
    FRAME_COUNT = 30  # Target: 30 frames per video
    MIN_FRAME_COUNT = 20  # Minimum frames required (will be padded to 30)
    FRAME_SIZE = (224, 224)  # EfficientNet input size
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0003
    VALIDATION_SPLIT = 0.2
    
    def __init__(self, model_dir: str = MODEL_FOLDER_PATH):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Caches for prototype extraction
        self.features_cache = []
        self.labels_cache = []
        self.keypoints_cache = []  # Video metadata
        
        # Create directories
        os.makedirs(f"{model_dir}/video_classifier", exist_ok=True)
        os.makedirs(f"{model_dir}/prototypes", exist_ok=True)
        
        print("✓ VideoClassifierTrainer initialized")
    
    # ════════════════════════════════════════════════════════════════
    # FRAME EXTRACTION
    # ════════════════════════════════════════════════════════════════
    
    def extract_30_frames(self, video_path: str) -> np.ndarray:
        """
        Extract frames from video and pad to exactly 30 frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of shape (30, 224, 224, 3) - normalized uint8
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.MIN_FRAME_COUNT:
            cap.release()
            raise ValueError(
                f"Video has only {total_frames} frames, need at least {self.MIN_FRAME_COUNT}"
            )
        
        # Extract all frames from video
        frames = []
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if ret:
                # Resize to 224x224 and convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.FRAME_SIZE)
                frames.append(frame)
        
        cap.release()
        
        frames = np.array(frames, dtype=np.uint8)
        
        # If more frames than needed, sample uniformly
        if len(frames) > self.FRAME_COUNT:
            indices = np.linspace(0, len(frames) - 1, self.FRAME_COUNT, dtype=int)
            frames = frames[indices]
        
        # If fewer frames than needed, pad by repeating last frame
        elif len(frames) < self.FRAME_COUNT:
            num_to_pad = self.FRAME_COUNT - len(frames)
            # Repeat last frame to reach target count
            padding = np.tile(frames[-1:], (num_to_pad, 1, 1, 1))
            frames = np.vstack([frames, padding])
        
        return frames
    
    # ════════════════════════════════════════════════════════════════
    # DATASET PREPARATION
    # ════════════════════════════════════════════════════════════════
    
    def prepare_dataset(self, dataset_path: str, shot_types: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all videos and extract 30 frames from each
        
        Args:
            dataset_path: Path to dataset root directory
            shot_types: List of shot type folder names
            
        Returns:
            (video_paths, y) where video_paths contains valid video file paths and y is labels
        """
        video_paths = []
        y = []
        
        print("\n" + "="*70)
        print("PREPARING VIDEO DATASET")
        print("="*70)
        
        for shot_idx, shot_type in enumerate(shot_types):
            shot_dir = os.path.join(dataset_path, shot_type)
            if not os.path.exists(shot_dir):
                print(f"⚠ Shot directory not found: {shot_dir}")
                continue
            
            # Find all video files
            video_files = []
            for ext in SUPPORTED_VIDEO_EXTENSIONS:
                video_files.extend(pathlib.Path(shot_dir).glob(f"*{ext}"))
            
            print(f"\n📹 {shot_type}: Found {len(video_files)} videos")
            
            for video_path in video_files:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"   ⚠ Skipped {video_path.name}: Cannot open video")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames < self.MIN_FRAME_COUNT:
                    print(
                        f"   ⚠ Skipped {video_path.name}: "
                        f"Video has only {total_frames} frames, need at least {self.MIN_FRAME_COUNT}"
                    )
                    continue

                video_paths.append(str(video_path))
                y.append(shot_idx)
            
            print(f"   ✓ Successfully loaded {len([label for label in y if label == shot_idx])} videos")
        
        video_paths = np.array(video_paths)
        y = np.array(y, dtype=np.int32)
        
        print(f"\n✓ Dataset prepared:")
        print(f"  Shape: video_paths={video_paths.shape}, y={y.shape}")
        print(f"  Data type: y={y.dtype}")
        
        # Distribution
        for shot_idx, shot_type in enumerate(shot_types):
            count = np.sum(y == shot_idx)
            print(f"  {shot_type}: {count} videos")
        
        return video_paths, y

    def _load_video_numpy(self, path_bytes: bytes) -> np.ndarray:
        """Load and normalize a single video into (30, 224, 224, 3) float32."""
        video_path = path_bytes.decode("utf-8")
        frames = self.extract_30_frames(video_path)
        # Match ImageNet preprocessing expected by EfficientNet weights.
        return preprocess_input(frames.astype(np.float32))

    def _get_available_memory_gb(self) -> float:
        """Return currently available system RAM in GB (best effort)."""
        try:
            if os.name == "nt":
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return stat.ullAvailPhys / (1024 ** 3)
        except Exception:
            pass

        # Fallback default when memory cannot be queried.
        return 8.0

    def _latest_checkpoint_info(self, checkpoint_dir: str) -> Tuple[str, int]:
        """Find latest intermediate checkpoint and extract its epoch number."""
        pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.weights.h5")
        candidates = glob.glob(pattern)
        if not candidates:
            return "", 0

        best_path = ""
        best_epoch = 0
        for path in candidates:
            match = re.search(r"checkpoint_epoch_(\d+)\.weights\.h5$", os.path.basename(path))
            if not match:
                continue
            epoch_num = int(match.group(1))
            if epoch_num > best_epoch:
                best_epoch = epoch_num
                best_path = path

        return best_path, best_epoch

    def build_tf_dataset(self, video_paths: np.ndarray, labels: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        """Build tf.data pipeline that streams videos from disk."""
        available_gb = self._get_available_memory_gb()
        low_memory_mode = available_gb < 6.0

        shuffle_buffer = min(len(video_paths), 256 if low_memory_mode else 1024)
        num_parallel_calls = 1 if low_memory_mode else tf.data.AUTOTUNE
        prefetch_count = 1 if low_memory_mode else 2

        if low_memory_mode:
            print(f"⚠ Low-memory mode enabled (available RAM: {available_gb:.2f} GB)")

        ds = tf.data.Dataset.from_tensor_slices((video_paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

        def _map_fn(path, label):
            frames = tf.numpy_function(self._load_video_numpy, [path], tf.float32)
            frames.set_shape((self.FRAME_COUNT, self.FRAME_SIZE[0], self.FRAME_SIZE[1], 3))
            return frames, tf.cast(label, tf.int32)

        ds = ds.map(_map_fn, num_parallel_calls=num_parallel_calls)
        ds = ds.batch(self.BATCH_SIZE)
        ds = ds.prefetch(prefetch_count)
        return ds
    
    # ════════════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE
    # ════════════════════════════════════════════════════════════════
    
    def build_model(self, num_classes: int) -> keras.Model:
        """
        Build EfficientNetB4 + TimeDistributed + GRU model
        
        Architecture:
        - TimeDistributed EfficientNetB4 (extract CNN features from each frame)
        - TimeDistributed GlobalAveragePooling2D (condense per-frame features)
        - GRU layers (capture temporal dependencies)
        - Dense layers (classification)
        
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*70)
        print("BUILDING MODEL ARCHITECTURE")
        print("="*70)
        
        # Load pre-trained EfficientNetB4
        base_model = EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=(self.FRAME_SIZE[0], self.FRAME_SIZE[1], 3)
        )
        base_model.trainable = False  # Freeze pretrained weights
        
        # Build model with TimeDistributed processing
        model = models.Sequential([
            layers.Input(shape=(self.FRAME_COUNT, self.FRAME_SIZE[0], self.FRAME_SIZE[1], 3)),
            # TimeDistributed: Apply EfficientNetB4 to each of 30 frames
            # Input: (batch, 30, 224, 224, 3) -> Output: (batch, 30, feature_maps)
            layers.TimeDistributed(base_model),
            
            # Condense per-frame features
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            # Output: (batch, 30, 1280) - per-frame feature vectors
            
            # GRU layers to capture temporal dependencies
            layers.GRU(256, return_sequences=True, dropout=0.3, unroll=True),
            layers.GRU(128, dropout=0.3, unroll=True),
            # Output: (batch, 128) - aggregated temporal features
            
            # Dense layers for classification
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile
        optimizer = Adam(learning_rate=self.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model compiled successfully")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    # ════════════════════════════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════════════════════════════
    
    def train(self, video_paths: np.ndarray, y: np.ndarray, use_augmentation: bool = False):
        """
        Train the video classification model
        
        Args:
            video_paths: Training video paths (n_samples,)
            y: Training labels (n_samples,)
            use_augmentation: Whether to apply data augmentation
        """
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)

        class_weights_np = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weight = {int(i): float(w) for i, w in enumerate(class_weights_np)}
        
        # Split dataset
        paths_train, paths_val, y_train, y_val = train_test_split(
            video_paths, y_encoded,
            test_size=self.VALIDATION_SPLIT,
            random_state=42,
            stratify=y_encoded
        )
        
        print(f"Training set: {paths_train.shape}")
        print(f"Validation set: {paths_val.shape}")

        train_ds = self.build_tf_dataset(paths_train, y_train, shuffle=True)
        val_ds = self.build_tf_dataset(paths_val, y_val, shuffle=False)
        
        # Build model
        model = self.build_model(num_classes)
        self.model = model
        
        # Create checkpoint directories
        best_model_dir = f"{self.model_dir}/video_classifier"
        temp_checkpoint_dir = f"{self.model_dir}/video_classifier/checkpoints"
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(temp_checkpoint_dir, exist_ok=True)

        # Resume support: continue from latest intermediate checkpoint if present.
        resume_ckpt_path, resume_epoch = self._latest_checkpoint_info(temp_checkpoint_dir)
        if resume_ckpt_path:
            self.model.load_weights(resume_ckpt_path)
            print(f"✓ Resuming from checkpoint: {resume_ckpt_path} (epoch {resume_epoch})")
        
        # Callbacks
        # Best model checkpoint (preserved after training)
        checkpoint = keras.callbacks.ModelCheckpoint(
            f"{best_model_dir}/best_model.weights.h5",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        
        # Intermediate checkpoints (will be deleted after training)
        intermediate_checkpoint = keras.callbacks.ModelCheckpoint(
            f"{temp_checkpoint_dir}/checkpoint_epoch_{{epoch:02d}}.weights.h5",
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            verbose=0
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        history = model.fit(
            train_ds,
            epochs=self.EPOCHS,
            initial_epoch=resume_epoch,
            validation_data=val_ds,
            class_weight=class_weight,
            callbacks=[checkpoint, intermediate_checkpoint, reduce_lr, early_stop],
            verbose=1
        )
        
        print("\n✓ Training completed")
        
        return history, paths_train, paths_val, y_train, y_val, temp_checkpoint_dir
    
    # ════════════════════════════════════════════════════════════════
    # PROTOTYPE EXTRACTION
    # ════════════════════════════════════════════════════════════════
    
    def extract_prototypes(self, video_paths: np.ndarray, y: np.ndarray) -> Dict:
        """
        Extract prototype keypoints and features for mistake analysis
        
        Args:
            video_paths: Video paths (n_samples,)
            y: Labels
            
        Returns:
            Dictionary with prototypes for each shot type
        """
        print("\n" + "="*70)
        print("EXTRACTING PROTOTYPES")
        print("="*70)
        
        # Create intermediate model to extract CNN features
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[4].output  # After GRU(128), before Dense
        )
        
        # Stream features in mini-batches to avoid loading all videos into memory.
        features_per_class = {idx: [] for idx in range(len(self.label_encoder.classes_))}
        batch_size = max(1, self.BATCH_SIZE)

        total = len(video_paths)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_paths = video_paths[start_idx:end_idx]
            batch_labels = y[start_idx:end_idx]

            batch_frames = []
            for p in batch_paths:
                frames = preprocess_input(self.extract_30_frames(str(p)).astype(np.float32))
                batch_frames.append(frames)

            batch_frames = np.array(batch_frames, dtype=np.float32)
            batch_features = feature_extractor.predict(batch_frames, verbose=0)

            for i, label in enumerate(batch_labels):
                features_per_class[int(label)].append(batch_features[i])
        
        prototypes = {}
        
        for shot_idx, shot_type in enumerate(self.label_encoder.classes_):
            shot_feature_list = features_per_class.get(shot_idx, [])
            if not shot_feature_list:
                continue

            shot_features = np.vstack(shot_feature_list)
            
            # Average prototype
            prototype_feature = np.mean(shot_features, axis=0)
            prototype_std = np.std(shot_features, axis=0)
            
            prototypes[shot_type] = {
                'features': {
                    'mean': prototype_feature,
                    'std': prototype_std
                },
                'n_samples': len(shot_feature_list)
            }
            
            print(f"✓ {shot_type}: {len(shot_feature_list)} samples, feature shape: {prototype_feature.shape}")
        
        return prototypes
    
    # ════════════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE (for mistake analyzer)
    # ════════════════════════════════════════════════════════════════
    
    def calculate_feature_importance(self, num_classes: int) -> np.ndarray:
        """
        Calculate feature importance from model weights
        
        Returns:
            Feature importance array
        """
        # Get weights from GRU layer
        gru_layer = self.model.layers[4]  # First GRU
        if hasattr(gru_layer, 'weights') and len(gru_layer.weights) > 0:
            weight_matrix = gru_layer.weights[0].numpy()
            importance = np.abs(weight_matrix).mean(axis=1)
            # Normalize
            importance = importance / importance.sum()
        else:
            # Fallback: uniform importance
            importance = np.ones(128) / 128
        
        return importance
    
    # ════════════════════════════════════════════════════════════════
    # MODEL CACHING (save/load complete model with architecture+weights)
    # ════════════════════════════════════════════════════════════════
    
    def save_compiled_model(self, num_classes: int):
        """
        Save the complete compiled model (architecture + weights) to .keras format.
        This allows future loads to skip architecture rebuild and directly load.
        
        Call this once after build_model() and load_weights().
        Use load_compiled_model() to load it back without rebuilding.
        """
        model_cache_dir = f"{self.model_dir}/video_classifier"
        model_cache_path = f"{model_cache_dir}/model_complete.keras"
        
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("No model to save. Call build_model() first.")
        
        os.makedirs(model_cache_dir, exist_ok=True)
        self.model.save(model_cache_path)
        print(f"✓ Complete model saved (architecture + weights): {model_cache_path}")
        return model_cache_path
    
    def load_compiled_model(self) -> keras.Model:
        """
        Load a pre-built, pre-compiled model from cache.
        Returns None if cache doesn't exist (caller should then build_model + load_weights).
        """
        model_cache_path = f"{self.model_dir}/video_classifier/model_complete.keras"
        
        if not os.path.exists(model_cache_path):
            return None
        
        try:
            model = keras.models.load_model(model_cache_path)
            self.model = model
            print(f"✓ Loaded pre-compiled model from cache: {model_cache_path}")
            return model
        except Exception as e:
            print(f"⚠ Failed to load cached model ({e}). Will rebuild.")
            return None
    
    # ════════════════════════════════════════════════════════════════
    # SAVE/LOAD MODELS
    # ════════════════════════════════════════════════════════════════
    
    def save_models(self, shot_types: List[str]):
        """Save trained model and metadata"""
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        model_path = f"{self.model_dir}/video_classifier/model.weights.h5"
        self.model.save_weights(model_path)
        # Save model in native Keras format for safer serialization.
        # model_path = f"{self.model_dir}/video_classifier/model.keras"
        # self.model.save(model_path)
        # Verify the saved model can be loaded before considering save successful.
        # keras.models.load_model(model_path)
        # print(f"✓ Model saved: {model_path}")
        
        # Save label encoder
        le_path = f"{self.model_dir}/video_classifier/label_encoder.pkl"
        joblib.dump(self.label_encoder, le_path)
        print(f"✓ Label encoder saved: {le_path}")
        
        # Save metadata
        metadata = {
            'shot_types': shot_types,
            'frame_count': self.FRAME_COUNT,
            'frame_size': self.FRAME_SIZE,
            'model_type': 'video_classifier_efficientnetb4_gru',
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = f"{self.model_dir}/video_classifier/metadata.pkl"
        joblib.dump(metadata, metadata_path)
        print(f"✓ Metadata saved: {metadata_path}")
    
    # ════════════════════════════════════════════════════════════════
    # MAIN TRAINING PIPELINE
    # ════════════════════════════════════════════════════════════════
    
    def train_pipeline(self, dataset_path: str, shot_types: List[str]):
        """
        Complete training pipeline
        """
        print("\n" + "="*80)
        print(" CRICKET SHOT CLASSIFICATION - VIDEO CLASSIFIER TRAINING PIPELINE")
        print("="*80)
        
        # Step 1: Prepare dataset
        video_paths, y = self.prepare_dataset(dataset_path, shot_types)
        
        # Step 2: Train model
        history, paths_train, paths_val, y_train, y_val, temp_checkpoint_dir = self.train(video_paths, y)
        
        # Step 3: Load best weights into current model
        self.model.load_weights(
            f"{self.model_dir}/video_classifier/best_model.weights.h5"
        )
        print("\n✓ Loaded best model weights from checkpoint")
        self.save_compiled_model(len(shot_types))
        
        # Step 4: Extract prototypes
        prototypes = self.extract_prototypes(video_paths, y)
        proto_path = f"{self.model_dir}/video_classifier/shot_prototypes.pkl"
        os.makedirs(os.path.dirname(proto_path), exist_ok=True)
        joblib.dump(prototypes, proto_path)
        print(f"✓ Prototypes saved: {proto_path}")
        
        # Step 5: Calculate feature importance
        feature_importance = self.calculate_feature_importance(len(shot_types))
        fi_path = f"{self.model_dir}/video_classifier/feature_importance.pkl"
        joblib.dump(feature_importance, fi_path)
        print(f"✓ Feature importance saved: {fi_path}")
        
        # Step 6: Save models
        self.save_models(shot_types)

        # Step 7: Clean up intermediate checkpoints only after successful final save
        import shutil
        if os.path.exists(temp_checkpoint_dir):
            shutil.rmtree(temp_checkpoint_dir)
            print("✓ Cleaned up temporary checkpoints")
        
        print("\n" + "="*80)
        print(" ✓ TRAINING COMPLETE")
        print("="*80)
        
        return history


def main():
    """Main entry point"""
    trainer = VideoClassifierTrainer()
    
    # Train
    trainer.train_pipeline(
        dataset_path=DATASET_PATH,
        shot_types=SHOT_TYPES
    )


if __name__ == "__main__":
    main()