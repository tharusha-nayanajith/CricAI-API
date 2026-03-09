"""
Script to extract pose keypoints from a professional cricket shot image
and create/update golden_frame_drive.json
"""

import cv2
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

def extract_keypoints_from_image(image_path):
    """Extract pose keypoints from an image"""
    
    # Initialize MediaPipe Pose Landmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect pose
    result = detector.detect(mp_image)
    
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        raise ValueError("No pose detected in image. Make sure full body is visible.")
    
    # Extract keypoints
    pose_landmarks = result.pose_landmarks[0]
    keypoints = []
    
    for landmark in pose_landmarks:
        keypoints.append({
            'x': float(landmark.x),
            'y': float(landmark.y),
            'z': float(landmark.z),
            'visibility': float(landmark.visibility)
        })
    
    print(f"✅ Extracted {len(keypoints)} keypoints from {image_path}")
    return keypoints


def add_to_golden_frames(player_name, shot_type, keypoints, json_file='golden_frame_drive.json'):
    """Add or update golden frame data"""
    
    # Load existing data or create new
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️  Existing file is invalid JSON, creating new file")
                data = {}
    else:
        data = {}
    
    # Add player if not exists
    if player_name not in data:
        data[player_name] = {}
    
    # Add shot
    data[player_name][shot_type] = {
        'keypoints': keypoints
    }
    
    # Save to file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved to {json_file}")
    print(f"   Player: {player_name}")
    print(f"   Shot: {shot_type}")
    print(f"   Keypoints: {len(keypoints)}")


def main():
    """Main function to extract and save golden frames"""
    
    print("🏏 Cricket Shot Keypoint Extractor")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists('pose_landmarker_lite.task'):
        print("❌ Error: pose_landmarker_lite.task not found!")
        print("Download it first with:")
        print("curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        return
    
    # Get input from user
    print("\nEnter details for the cricket shot:")
    image_path = input("Image path (e.g., virat_cover_drive.jpg): ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    player_name = input("Player name (e.g., Virat Kohli): ").strip()
    shot_type = input("Shot type (e.g., cover_drive, pull_shot, straight_drive): ").strip()
    
    try:
        # Extract keypoints
        print("\n🔍 Extracting keypoints...")
        keypoints = extract_keypoints_from_image(image_path)
        
        # Save to JSON
        print("\n💾 Saving to golden_frame_drive.json...")
        add_to_golden_frames(player_name, shot_type, keypoints)
        
        print("\n✅ Success! You can now use this in your app.")
        
        # Show current data structure
        with open('golden_frame_drive.json', 'r') as f:
            data = json.load(f)
        
        print("\n📊 Current golden frames:")
        for player, shots in data.items():
            print(f"  👤 {player}")
            for shot in shots.keys():
                print(f"     🏏 {shot}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()