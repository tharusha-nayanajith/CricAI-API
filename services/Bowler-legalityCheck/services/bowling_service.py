import threading
from features.bowlingActionsChecker.bowlingactions import (
    load_model_and_scaler, 
    infer_image,
    infer_video,
    extract_release_frame_and_analyze
)

_model = None
_scaler = None
_meta = None
_lock = threading.Lock()

def _ensure_loaded():
    """Lazy load model and scaler (singleton pattern)"""
    global _model, _scaler, _meta
    if _model is None:
        with _lock:
            if _model is None:
                print("🔄 Loading bowling action model...")
                _model, _scaler, _meta = load_model_and_scaler()
                print("✅ Model loaded!")

def get_model_and_infer(img_path: str):
    """Infer from image"""
    _ensure_loaded()
    return infer_image(img_path, _model, _scaler)

def get_model_and_infer_video(video_path: str, detection_method: str = 'wrist_velocity'):
    """Infer from video - detects release frame automatically"""
    _ensure_loaded()
    return infer_video(video_path, _model, _scaler, detection_method)

def get_model_and_infer_video_with_frame(video_path: str, detection_method: str = 'wrist_velocity'):
    """
    Infer from video and return release frame image
    Returns: dict with prob_illegal, keypoints, frame_index, release_frame (numpy array)
    """
    _ensure_loaded()
    
    # First extract the release frame and analyze
    extract_result = extract_release_frame_and_analyze(video_path, detection_method)
    
    if extract_result["status"] != "success":
        return {
            "status": "error",
            "error": extract_result.get("message", "Failed to process video"),
            "prob_illegal": None
        }
    
    # Get the keypoints and release frame
    keypoints = extract_result["keypoints"]
    release_frame = extract_result.get("release_frame")
    frame_index = extract_result["frame_index"]
    
    # Run inference
    Xs = _scaler.transform(keypoints.reshape(1, -1))
    pred = _model.predict(Xs, verbose=0)[0][0]
    
    return {
        "status": "success",
        "video": video_path,
        "frame_index": frame_index,
        "prob_illegal": float(pred),
        "keypoints": keypoints,
        "release_frame": release_frame  # Include the actual frame
    }