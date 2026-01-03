"""
Vision Engine Service
Handles YOLO model loading and inference
"""

from typing import Any, Optional, List, Dict
import numpy as np
from pathlib import Path

from app.core.config import settings


class VisionEngine:
    """
    Singleton class for managing YOLO model and performing object detection
    
    This class will:
    - Load and manage the YOLO model
    - Perform inference on frames
    - Track detected objects across frames
    - Extract ball and player positions
    """
    
    _instance: Optional['VisionEngine'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """
        Implement Singleton pattern to ensure only one model instance
        """
        if cls._instance is None:
            cls._instance = super(VisionEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the vision engine (only once due to Singleton)
        """
        if not self._initialized:
            self.model = None
            self.model_path = settings.YOLO_MODEL_PATH  # Use YOLO model path
            self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
            self._initialized = True
    
    def load_model(self) -> None:
        """
        Load the YOLO model from disk
        
        Loads the trained YOLO model for stump detection
        """
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run object detection on a single frame
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detections with bounding boxes, classes, and confidence scores
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            for r in results:
                # Handle both detection and segmentation models
                # Segmentation models have boxes in r.boxes
                # Detection models also have boxes in r.boxes
                boxes = r.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                    
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = self.model.names[cls_id]
                    
                    detection = {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "center_x": (x1 + x2) // 2,
                            "center_y": (y1 + y2) // 2
                        }
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in predict: {error_details}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame and extract relevant objects
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Dictionary containing detected objects (ball, stumps, players, etc.)
            
        TODO: Implement frame processing
        - Run detection
        - Filter relevant classes (ball, stumps, batsman, etc.)
        - Extract coordinates
        """
        pass
    
    def detect_ball(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Detect cricket ball in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            Ball position (x, y) or None if not detected
        """
        pass
    
    def detect_stumps(self, frame: np.ndarray) -> List[tuple]:
        """
        Detect stump positions for calibration
        
        Args:
            frame: Input frame
            
        Returns:
            List of stump detections with bounding boxes
        """
        detections = self.predict(frame)
        
        # Debug: Log all detections
        print(f"\n=== DEBUG: All detections ({len(detections)}) ===")
        for i, d in enumerate(detections):
            print(f"  [{i}] Class: {d['class_name']} (ID: {d['class_id']}), Confidence: {d['confidence']:.3f}")
        
        print(f"\n=== Filtering for STUMP_CLASS_ID={settings.STUMP_CLASS_ID}, MIN_CONFIDENCE={settings.MIN_STUMP_CONFIDENCE} ===")
        
        # Filter for stump class (class_id = 0 based on user's code)
        stumps = [
            d for d in detections 
            if d["class_id"] == settings.STUMP_CLASS_ID 
            and d["confidence"] >= settings.MIN_STUMP_CONFIDENCE
        ]
        
        print(f"=== Filtered stumps: {len(stumps)} ===\n")
        
        return stumps
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded and ready
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None


# Global instance
vision_engine = VisionEngine()
