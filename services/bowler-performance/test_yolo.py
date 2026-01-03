"""
Test script to verify YOLO model works independently
"""

from ultralytics import YOLO
import cv2
import numpy as np

# Load model
print("Loading model...")
model = YOLO('app/models/ml/best.pt')
print("Model loaded successfully!")

# Create a dummy image (or load a real one)
# Using a dummy image for testing
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print("Running inference...")
results = model(dummy_image, conf=0.5)

print(f"Results type: {type(results)}")
print(f"Number of results: {len(results)}")

for r in results:
    print(f"Result type: {type(r)}")
    print(f"Has boxes: {hasattr(r, 'boxes')}")
    if hasattr(r, 'boxes'):
        print(f"Boxes: {r.boxes}")
        if r.boxes is not None:
            print(f"Number of detections: {len(r.boxes)}")

print("\n✅ Test completed successfully!")
