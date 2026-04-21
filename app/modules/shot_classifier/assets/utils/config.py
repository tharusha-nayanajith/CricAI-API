"""
Configuration file for Shot Classification System
"""

import os

# Shot types from your Kaggle dataset
SHOT_TYPES = ['cut', 'drive', 'flick', 'misc', 'pull', 'slog', 'sweep']

# RTMPose keypoint indices (COCO format)
KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Feature extraction parameters
FRAME_EXTRACTION_FPS = 10  # Extract 10 frames per second
MIN_CONFIDENCE = 0.3  # Minimum confidence for pose keypoints

# Weights for shot moment detection
ALPHA = 0.6  # Weight for angle change
BETA = 0.4   # Weight for hand velocity

# Model training parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# File paths
DATASET_PATH = "C:\\Users\\User\\Downloads\\Cricket-Shots" 

# Mode 1 trained models - self-contained in assets directory
MODEL_FOLDER_PATH = "app/modules/shot_classifier/assets/trained_models"
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

def supported_extensions_str() -> str:
    """Human-readable extensions for error messages"""
    return ", ".join(ext.upper() for ext in SUPPORTED_VIDEO_EXTENSIONS)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))