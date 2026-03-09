# # test_infer.py

# from features.bowlingActionsChecker.bowlingactions import load_model_and_scaler, infer_image

# # Load model, scaler
# model, scaler, meta = load_model_and_scaler()

# # Inference on single image
# img_path = "data/practice/pracimg11.jpeg"
# res = infer_image(img_path, model, scaler)

# prob_illegal = res["prob_illegal"]

# if prob_illegal is None:
#     print(f"{img_path}: Keypoints not detected!")
# elif prob_illegal >= 0.5:  # Threshold can be adjusted
#     print(f"{img_path}: Illegal (prob={prob_illegal:.2f})")
# else:
#     print(f"{img_path}: Legal (prob={prob_illegal:.2f})")

# test_infer_video.py
import cv2
from features.bowlingActionsChecker.bowlingactions import (
    load_model_and_scaler, 
    infer_image,
    infer_video,
    extract_release_frame_and_analyze
)

def test_image(img_path):
    """Test single image"""
    print("\n" + "="*60)
    print("🖼️  TESTING IMAGE")
    print("="*60)
    
    model, scaler, meta = load_model_and_scaler()
    res = infer_image(img_path, model, scaler)
    prob_illegal = res["prob_illegal"]
    
    if prob_illegal is None:
        print(f"❌ {img_path}: Keypoints not detected!")
    elif prob_illegal >= 0.5:
        print(f"🚫 {img_path}: ILLEGAL (prob={prob_illegal:.2f})")
    else:
        print(f"✅ {img_path}: LEGAL (prob={prob_illegal:.2f})")

def test_video(video_path, detection_method='wrist_velocity', save_release_frame=False):
    """Test video with ball release detection"""
    print("\n" + "="*60)
    print("🎥 TESTING VIDEO")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Detection method: {detection_method}")
    
    # Load model
    model, scaler, meta = load_model_and_scaler()
    
    # Method 1: Full inference
    res = infer_video(video_path, model, scaler, detection_method)
    
    if res.get("status") != "success":
        print(f"❌ Error: {res.get('error', 'Unknown error')}")
        return
    
    frame_idx = res.get("frame_index")
    prob_illegal = res.get("prob_illegal")
    
    print(f"\n📍 Ball release detected at frame: {frame_idx}")
    
    if prob_illegal is None:
        print(f"❌ {video_path}: Keypoints not detected in release frame!")
    elif prob_illegal >= 0.5:
        print(f"🚫 {video_path}: ILLEGAL bowling action (prob={prob_illegal:.2f})")
    else:
        print(f"✅ {video_path}: LEGAL bowling action (prob={prob_illegal:.2f})")
    
    # Optional: Save the release frame for inspection
    if save_release_frame:
        result = extract_release_frame_and_analyze(video_path, detection_method)
        if result["status"] == "success" and result["release_frame"] is not None:
            # Get the base path without extension
            import os
            base_path = os.path.splitext(video_path)[0]
            output_path = base_path + '_release_frame.jpg'
            
            success = cv2.imwrite(output_path, result["release_frame"])
            if success:
                print(f"💾 Release frame saved to: {output_path}")
            else:
                print(f"⚠️  Failed to save release frame to: {output_path}")

def test_video_all_methods(video_path):
    """Test video with all detection methods"""
    print("\n" + "="*60)
    print("🔬 TESTING ALL DETECTION METHODS")
    print("="*60)
    
    model, scaler, meta = load_model_and_scaler()
    methods = ['wrist_velocity', 'wrist_deceleration', 'arm_extension']
    
    for method in methods:
        print(f"\n--- Method: {method} ---")
        res = infer_video(video_path, model, scaler, method)
        
        if res.get("status") == "success":
            frame_idx = res.get("frame_index")
            prob = res.get("prob_illegal")
            label = "ILLEGAL" if prob >= 0.5 else "LEGAL"
            print(f"Frame {frame_idx}: {label} (prob={prob:.2f})")
        else:
            print(f"Error: {res.get('error')}")

if __name__ == "__main__":
    # =====================================
    # TEST IMAGE
    # =====================================
    img_path = "data/practice/pracimg11.jpeg"
    test_image(img_path)
    
    # =====================================
    # TEST VIDEO (Single method)
    # =====================================
    video_path = "data/practice/pracvideo3.mp4"  # Change to your video path
    test_video(video_path, detection_method='wrist_velocity', save_release_frame=True)
    
    # =====================================
    # TEST VIDEO (All methods comparison)
    # =====================================
    # Uncomment to compare all detection methods:
    # test_video_all_methods(video_path)
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)