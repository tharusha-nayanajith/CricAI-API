from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import json
from typing import Dict, List
import io
import os
import tempfile
from pydantic import BaseModel

# Impact detection pipeline imports
try:
    from impact_detector_pipeline import run_impact_detection_pipeline
    from ball_detector import load_model as load_yolo_model
    _IMPACT_DETECTION_AVAILABLE = True
except ImportError:
    _IMPACT_DETECTION_AVAILABLE = False
    print("Warning: impact detection modules not found or dependencies missing.")

app = FastAPI(title="Cricket Shot Analyzer API")

# Pre-load YOLO model at startup to avoid per-request cold start
_yolo_model = None
if _IMPACT_DETECTION_AVAILABLE:
    try:
        import os
        # Use custom trained cricket ball model if available, else fall back to COCO
        _custom_weights = "runs/detect/train/weights/best.pt"
        _model_to_load = _custom_weights if os.path.exists(_custom_weights) else "yolov8n.pt"
        _yolo_model = load_yolo_model(_model_to_load)
        print(f"YOLO model loaded: {_model_to_load}")
    except Exception as e:
        print(f"Warning: Could not pre-load YOLO model: {e}")


# CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Pose Landmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_detector = vision.PoseLandmarker.create_from_options(options)

# Load golden frame data
try:
    with open('golden_frame_drive.json', 'r') as f:
        raw_data = json.load(f)
        
    # Normalize the data structure
    GOLDEN_FRAMES = {}
    
    # Check if it's already in the expected format (dict with player names)
    if isinstance(raw_data, dict) and all(isinstance(v, dict) for v in raw_data.values()):
        GOLDEN_FRAMES = raw_data
    # Check if it's a list of keypoints (single shot)
    elif isinstance(raw_data, list):
        GOLDEN_FRAMES = {
            "Professional Player": {
                "drive_shot": {
                    "keypoints": raw_data
                }
            }
        }
    # Check if it's a dict with direct keypoints
    elif isinstance(raw_data, dict) and 'keypoints' in raw_data:
        GOLDEN_FRAMES = {
            "Professional Player": {
                "drive_shot": raw_data
            }
        }
    else:
        print(f"Warning: Unexpected golden frame format. Structure: {type(raw_data)}")
        GOLDEN_FRAMES = {}
        
except FileNotFoundError:
    GOLDEN_FRAMES = {}
    print("Warning: golden_frame_drive.json not found")
except json.JSONDecodeError as e:
    GOLDEN_FRAMES = {}
    print(f"Warning: golden_frame_drive.json is not valid JSON: {e}")


class ComparisonResult(BaseModel):
    similarity_percentage: float
    matched_player: str
    shot_type: str
    keypoints_detected: int
    confidence: float
    feedback: List[str]


def extract_keypoints(image_bytes: bytes) -> Dict:
    """Extract pose keypoints from image using BlazePose"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image format")
        
        # Resize if image is too large (helps with processing)
        height, width = image.shape[:2]
        max_dimension = 1024
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose
        detection_result = pose_detector.detect(mp_image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            raise ValueError("No pose detected in image. Make sure the full body is visible.")
        
        # Extract keypoints from first pose detected
        pose_landmarks = detection_result.pose_landmarks[0]
        
        # Extract keypoints as list of [x, y, z, visibility]
        keypoints = []
        for landmark in pose_landmarks:
            keypoints.append({
                'x': float(landmark.x),
                'y': float(landmark.y),
                'z': float(landmark.z),
                'visibility': float(landmark.visibility)
            })
        
        return {
            'keypoints': keypoints,
            'image_height': image.shape[0],
            'image_width': image.shape[1]
        }
    except Exception as e:
        raise ValueError(f"Keypoint extraction failed: {str(e)}")


def normalize_keypoints(keypoints: List[Dict]) -> np.ndarray:
    """Normalize keypoints for comparison"""
    # Convert to numpy array
    kp_array = np.array([[kp['x'], kp['y'], kp['z']] for kp in keypoints])
    
    # Normalize using hip center (landmarks 23, 24)
    hip_center = (kp_array[23] + kp_array[24]) / 2
    kp_normalized = kp_array - hip_center
    
    # Scale normalization
    scale = np.max(np.abs(kp_normalized))
    if scale > 0:
        kp_normalized = kp_normalized / scale
    
    return kp_normalized


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
    
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def calculate_similarity(user_kp: List[Dict], golden_kp: List[Dict]) -> Dict:
    """Calculate similarity between user and golden keypoints"""
    user_norm = normalize_keypoints(user_kp)
    golden_norm = normalize_keypoints(golden_kp)
    
    # Important joints for cricket shots (weighted)
    joint_weights = {
        11: 2.0, 12: 2.0, 13: 2.5, 14: 2.5, 15: 3.0, 16: 3.0,
        23: 2.0, 24: 2.0, 25: 1.5, 26: 1.5,
    }
    
    total_similarity = 0
    total_weight = 0
    feedback = []
    
    # 1. Positional Similarity
    for idx in range(len(user_norm)):
        weight = joint_weights.get(idx, 1.0)
        distance = np.linalg.norm(user_norm[idx] - golden_norm[idx])
        similarity = max(0, 1 - distance)
        total_similarity += similarity * weight
        total_weight += weight

    overall_similarity = (total_similarity / total_weight) * 100
    
    # 2. Angular Similarity and Feedback
    key_angles = {
        'left_elbow': (11, 13, 15),
        'right_elbow': (12, 14, 16),
        'left_shoulder': (13, 11, 23),
        'right_shoulder': (14, 12, 24),
        'left_hip': (11, 23, 25),
        'right_hip': (12, 24, 26),
        'left_knee': (23, 25, 27),
        'right_knee': (24, 26, 28),
    }
    
    angle_feedback_messages = {
        'left_elbow': "Bend your left elbow more.",
        'right_elbow': "Keep your right arm straighter.",
        'left_shoulder': "Open up your left shoulder.",
        'right_shoulder': "Rotate your right shoulder more.",
        'left_hip': "Engage your left hip.",
        'right_hip': "Drive through with your right hip.",
        'left_knee': "Bend your left knee more for stability.",
        'right_knee': "Ensure your right knee is stable.",
    }

    for name, (p1_idx, p2_idx, p3_idx) in key_angles.items():
        user_angle = calculate_angle(user_kp[p1_idx], user_kp[p2_idx], user_kp[p3_idx])
        golden_angle = calculate_angle(golden_kp[p1_idx], golden_kp[p2_idx], golden_kp[p3_idx])
        
        angle_diff = abs(user_angle - golden_angle)
        
        # Add to overall similarity, weighted by importance of angle
        angle_similarity = max(0, 1 - (angle_diff / 180)) # Normalize to 0-1
        total_similarity += angle_similarity * 1.5 # Angles are important
        total_weight += 1.5
        
        if angle_diff > 20: # Threshold for significant difference
            feedback.append(angle_feedback_messages.get(name, f"Check your {name.replace('_', ' ')} angle."))

    overall_similarity = (total_similarity / total_weight) * 100

    return {
        'similarity': overall_similarity,
        'feedback': feedback
    }


def find_best_match(user_keypoints: List[Dict]) -> Dict:
    """Find best matching professional shot"""
    best_match = None
    best_similarity = 0
    
    for player_name, shots in GOLDEN_FRAMES.items():
        for shot_type, shot_data in shots.items():
            # Handle different data structures
            if isinstance(shot_data, dict) and 'keypoints' in shot_data:
                golden_kp = shot_data['keypoints']
            elif isinstance(shot_data, list):
                golden_kp = shot_data
            else:
                print(f"Skipping invalid shot data for {player_name} - {shot_type}")
                continue
                
            result = calculate_similarity(user_keypoints, golden_kp)
            
            if result['similarity'] > best_similarity:
                best_similarity = result['similarity']
                best_match = {
                    'player': player_name,
                    'shot': shot_type,
                    'similarity': result['similarity'],
                    'feedback': result['feedback']
                }
    
    return best_match


@app.get("/")
async def root():
    return {"message": "Cricket Shot Analyzer API", "status": "running"}


@app.post("/analyze-shot", response_model=ComparisonResult)
async def analyze_shot(image: UploadFile = File(...)):
    """
    Analyze cricket shot from uploaded image
    Returns similarity comparison with professional player
    """
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")
        
        # Read image
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Extract keypoints
        try:
            user_data = extract_keypoints(image_bytes)
            user_keypoints = user_data['keypoints']
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        # Find best match
        if not GOLDEN_FRAMES:
            raise HTTPException(
                status_code=500, 
                detail="No reference data available. Please ensure golden_frame_drive.json is properly configured."
            )
        
        match_result = find_best_match(user_keypoints)
        
        if not match_result:
            raise HTTPException(
                status_code=404, 
                detail="No suitable match found. The pose might be too different from reference shots."
            )
        
        # Calculate confidence based on visibility
        avg_visibility = np.mean([kp['visibility'] for kp in user_keypoints])
        
        return ComparisonResult(
            similarity_percentage=round(match_result['similarity'], 2),
            matched_player=match_result['player'],
            shot_type=match_result['shot'],
            keypoints_detected=len(user_keypoints),
            confidence=round(avg_visibility * 100, 2),
            feedback=match_result['feedback'][:5]  # Top 5 feedback points
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Processing error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/available-shots")
async def get_available_shots():
    """Get list of available professional shots"""
    shots = []
    for player, player_shots in GOLDEN_FRAMES.items():
        for shot_type in player_shots.keys():
            shots.append({
                'player': player,
                'shot': shot_type
            })
    return {"shots": shots}


@app.get("/debug-golden-frames")
async def debug_golden_frames():
    """Debug endpoint to check golden frames structure"""
    return {
        "loaded": len(GOLDEN_FRAMES) > 0,
        "num_players": len(GOLDEN_FRAMES),
        "structure": {
            player: {
                shot: {
                    "has_keypoints": 'keypoints' in data if isinstance(data, dict) else isinstance(data, list),
                    "num_keypoints": len(data.get('keypoints', [])) if isinstance(data, dict) else len(data) if isinstance(data, list) else 0,
                    "data_type": str(type(data))
                }
                for shot, data in shots.items()
            }
            for player, shots in GOLDEN_FRAMES.items()
        }
    }


class ImpactDetectionResult(BaseModel):
    impact_frame: int
    impact_time: float
    method: str
    ball_detections: int
    audio_estimate: int


@app.post("/detect-impact", response_model=ImpactDetectionResult)
async def detect_impact(video: UploadFile = File(...)):
    """
    Detect the bat-ball impact frame in a cricket shot video.

    Upload a cricket shot video (mp4, mov, avi, etc.) and receive:
    - impact_frame: The exact frame where bat hits ball (0-based index)
    - impact_time: The timestamp in seconds
    - method: Detection method used (ball_velocity | audio_fallback)
    - ball_detections: Number of frames where YOLO detected the ball
    - audio_estimate: Initial audio-based frame estimate (before refinement)
    """
    if not _IMPACT_DETECTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Impact detection modules are not available. "
                   "Install: moviepy librosa scipy ultralytics",
        )

    # Validate file type
    allowed_types = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/mpeg"}
    if video.content_type and video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{video.content_type}'. "
                   "Please upload mp4, mov, avi, webm, or mpeg.",
        )

    # Save to a temp file (OpenCV and moviepy need a path, not a stream)
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        content = await video.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()

        result = run_impact_detection_pipeline(
            video_path=tmp_file.name,
            frame_window=5,
            preloaded_model=_yolo_model,
        )

        return ImpactDetectionResult(**result)

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Impact detection failed: {str(e)}")
    finally:
        # Always clean up the temp file to avoid disk leaks
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)