"""
Simple script to test YOLO stump detection model
Loads an image, runs detection, and displays results with bounding boxes
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

# Configuration
MODEL_PATH = "app/models/ml/best.pt"
CONFIDENCE_THRESHOLD = 0.25  # Adjust as needed
STUMP_CLASS_ID = 0  # Assuming stumps are class 0

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels on the image
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries
        
    Returns:
        Image with drawn bounding boxes
    """
    img_copy = image.copy()
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        conf = det['confidence']
        class_name = det['class_name']
        
        # Choose color based on class
        if det['class_id'] == STUMP_CLASS_ID:
            color = (0, 255, 0)  # Green for stumps
        else:
            color = (0, 165, 255)  # Orange for other objects
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        center_x, center_y = bbox['center_x'], bbox['center_y']
        cv2.circle(img_copy, (center_x, center_y), 5, color, -1)
        
        # Draw label with confidence
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw label background
        cv2.rectangle(
            img_copy,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        # Draw detection number
        cv2.putText(
            img_copy,
            f"#{i+1}",
            (center_x + 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    
    return img_copy


def main(image_path):
    """
    Main function to test stump detection
    
    Args:
        image_path: Path to the test image
    """
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        print("\nUsage: python test_stump_detection.py <path_to_image>")
        return
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    
    print("=" * 60)
    print("🏏 YOLO Stump Detection Test")
    print("=" * 60)
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not read image from {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"   Image size: {width}x{height}")
    
    # Load YOLO model
    print(f"\n🤖 Loading YOLO model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Run detection
    print(f"\n🔍 Running detection (confidence threshold: {CONFIDENCE_THRESHOLD})")
    try:
        results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    except Exception as e:
        print(f"   ❌ Error during detection: {e}")
        return
    
    # Parse detections
    detections = []
    for r in results:
        boxes = r.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = model.names[cls_id]
            
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
    
    # Display results
    print(f"\n📊 Detection Results:")
    print(f"   Total detections: {len(detections)}")
    
    if len(detections) == 0:
        print("\n   ⚠️  No objects detected!")
        print("   Try:")
        print("   - Lowering the confidence threshold")
        print("   - Using a different image with clearer stumps")
        print("   - Checking if the model is trained correctly")
        return
    
    # Group by class
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n   Detections by class:")
    for class_name, count in class_counts.items():
        print(f"   - {class_name}: {count}")
    
    # Filter stumps
    stumps = [d for d in detections if d['class_id'] == STUMP_CLASS_ID]
    print(f"\n   🎯 Stumps detected: {len(stumps)}")
    
    if len(stumps) > 0:
        print("\n   Stump details:")
        for i, stump in enumerate(stumps, 1):
            bbox = stump['bbox']
            print(f"   [{i}] Confidence: {stump['confidence']:.3f}, "
                  f"Center: ({bbox['center_x']}, {bbox['center_y']}), "
                  f"Size: {bbox['width']}x{bbox['height']}")
    
    # Draw detections on image
    print("\n🎨 Drawing detections...")
    result_image = draw_detections(image, detections)
    
    # Add summary text to image
    summary_text = f"Total: {len(detections)} | Stumps: {len(stumps)}"
    cv2.putText(
        result_image,
        summary_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Display image
    print("\n👁️  Displaying result (press any key to close)...")
    
    # Resize if image is too large
    max_display_width = 1280
    if width > max_display_width:
        scale = max_display_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        result_image = cv2.resize(result_image, (new_width, new_height))
        print(f"   (Resized for display: {new_width}x{new_height})")
    
    cv2.imshow('Stump Detection Results', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 Result saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_stump_detection.py <path_to_image>")
        print("\nExample:")
        print("  python test_stump_detection.py test_images/pitch.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)
