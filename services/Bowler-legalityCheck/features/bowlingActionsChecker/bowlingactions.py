import os
import cv2
import numpy as np
import pandas as pd
import json
import joblib
import math
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta
import mediapipe as mp

mp_pose = mp.solutions.pose

# CONFIG
PROF_DIR = "data/professional"
PRACTICE_DIR = "data/practice"
FEATURES_DIR = "features"
MODEL_PATH = "models/bowler_model.h5"
SCALER_PATH = "models/scaler.pkl"
META_PATH = "models/meta.json"
SELECT_LANDMARKS = [11,13,15,12,14,16,23,25,27]

# Utilities
def extract_keypoints_from_frame(frame):
    """Extract keypoints from a video frame"""
    if frame is None:
        return None
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        features = []
        for idx in SELECT_LANDMARKS:
            l = lm[idx]
            features.extend([l.x, l.y, l.z])
        return np.array(features, dtype=np.float32)

def extract_keypoints_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    return extract_keypoints_from_frame(img)

def normalize_keypoints_by_torso(features):
    try:
        left_sh_x = features[0]
        left_sh_y = features[1]
        right_sh_x = features[3]
        right_sh_y = features[4]
        shoulder_dist = math.hypot(right_sh_x-left_sh_x, right_sh_y-left_sh_y)
        if shoulder_dist < 1e-6:
            shoulder_dist = 1e-6
        norm = features.copy()
        for i in range(0,len(features),3):
            norm[i+0] = (norm[i+0]-left_sh_x)/shoulder_dist
            norm[i+1] = (norm[i+1]-left_sh_y)/shoulder_dist
            norm[i+2] = norm[i+2]/shoulder_dist
        return norm
    except:
        return features

def detect_ball_release_frame(video_path, method='wrist_velocity'):
    """
    Detect the ball release point in a bowling video
    Methods:
    - 'wrist_velocity': Detects peak in wrist velocity
    - 'arm_extension': Detects maximum arm extension
    - 'wrist_deceleration': Detects when wrist suddenly decelerates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    frames = []
    wrist_positions = []
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frames.append(frame.copy())
            
            # Process frame
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get right wrist position (index 16)
                right_wrist = landmarks[16]
                wrist_positions.append([right_wrist.x, right_wrist.y, right_wrist.z])
            else:
                wrist_positions.append([0, 0, 0])
    
    cap.release()
    
    if len(wrist_positions) < 3:
        return None, None
    
    wrist_positions = np.array(wrist_positions)
    
    # Calculate velocities
    velocities = []
    for i in range(1, len(wrist_positions)):
        vel = np.linalg.norm(wrist_positions[i] - wrist_positions[i-1])
        velocities.append(vel)
    
    velocities = np.array(velocities)
    
    if method == 'wrist_velocity':
        # Find peak velocity (ball release usually happens at peak wrist speed)
        if len(velocities) > 0:
            release_frame_idx = np.argmax(velocities)
        else:
            release_frame_idx = len(frames) // 2
    
    elif method == 'wrist_deceleration':
        # Find sudden deceleration (ball leaves hand)
        accelerations = np.diff(velocities)
        if len(accelerations) > 0:
            # Find largest negative acceleration (deceleration)
            release_frame_idx = np.argmin(accelerations) + 1
        else:
            release_frame_idx = len(frames) // 2
    
    elif method == 'arm_extension':
        # Find maximum arm extension
        arm_lengths = []
        for i, lm_data in enumerate(wrist_positions):
            # Would need shoulder position too - simplified here
            arm_lengths.append(lm_data[1])  # y-position as proxy
        
        release_frame_idx = np.argmax(arm_lengths)
    
    else:
        release_frame_idx = len(frames) // 2
    
    # Ensure index is valid
    release_frame_idx = min(release_frame_idx, len(frames) - 1)
    
    return frames[release_frame_idx], release_frame_idx

def extract_release_frame_and_analyze(video_path, detection_method='wrist_velocity'):
    """
    Main function: Extract release frame from video and get keypoints
    """
    print(f"🎥 Processing video: {video_path}")
    
    release_frame, frame_idx = detect_ball_release_frame(video_path, method=detection_method)
    
    if release_frame is None:
        return {
            "status": "error",
            "message": "Could not detect ball release or process video",
            "frame_index": None,
            "keypoints": None
        }
    
    print(f"✅ Detected release at frame: {frame_idx}")
    
    # Extract keypoints from release frame
    keypoints = extract_keypoints_from_frame(release_frame)
    
    if keypoints is None:
        return {
            "status": "error",
            "message": "Pose not detected in release frame",
            "frame_index": frame_idx,
            "keypoints": None
        }
    
    # Normalize keypoints
    normalized_kp = normalize_keypoints_by_torso(keypoints)
    
    return {
        "status": "success",
        "frame_index": frame_idx,
        "keypoints": normalized_kp,
        "release_frame": release_frame
    }

# Feature extraction from professional images (unchanged)
def process_professional_images(prof_dir=PROF_DIR, out_dir=FEATURES_DIR):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for label_name, lbl in [("legal",0),("illegal",1)]:
        folder = os.path.join(prof_dir,label_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg','.png','.jpeg')):
                continue
            img_path = os.path.join(folder,fname)
            kp = extract_keypoints_from_image(img_path)
            if kp is None:
                continue
            kp = normalize_keypoints_by_torso(kp)
            df = pd.DataFrame([kp])
            df['label'] = lbl
            out_csv = os.path.join(out_dir,f"{label_name}__{fname}.csv")
            df.to_csv(out_csv,index=False)
            rows.append(out_csv)
            print("Saved features:", out_csv)
    return rows

# Build dataset
def build_dataset_from_features(features_dir=FEATURES_DIR):
    files = [os.path.join(features_dir,f) for f in os.listdir(features_dir) if f.endswith('.csv')]
    X_list,y_list=[],[]
    for f in files:
        df = pd.read_csv(f)
        if 'label' not in df.columns:
            continue
        y = df['label'].values
        X = df.drop('label',axis=1).values
        X_list.append(X)
        y_list.append(y)
    if not X_list:
        raise RuntimeError("No feature CSVs found")
    X = np.vstack(X_list)
    y = np.concatenate(y_list).astype(np.int32)
    print("Dataset:", X.shape, y.shape)
    return X, y

# Model
def make_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model(X,y,model_path=MODEL_PATH,scaler_path=SCALER_PATH,meta_path=META_PATH):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train,X_val,y_train,y_val = train_test_split(Xs,y,test_size=0.15,stratify=y,random_state=42)
    model = make_model(X.shape[1])
    es = callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=200,batch_size=16,callbacks=[es],verbose=2)
    model.save(model_path)
    joblib.dump(scaler,scaler_path)
    meta = {"feature_dim": X.shape[1], "select_landmarks": SELECT_LANDMARKS}
    with open(meta_path,'w') as f:
        json.dump(meta,f,indent=2)
    print("Model saved:", model_path)
    return model,scaler,history

# Load & inference
def load_model_and_scaler(model_path=MODEL_PATH,scaler_path=SCALER_PATH,meta_path=META_PATH):
    model = models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path,'r') as f:
        meta = json.load(f)
    return model,scaler,meta

def infer_image(img_path, model, scaler):
    kp = extract_keypoints_from_image(img_path)
    if kp is None:
        return {"img":img_path,"prob_illegal":None}
    kp = normalize_keypoints_by_torso(kp)
    Xs = scaler.transform(kp.reshape(1,-1))
    pred = model.predict(Xs,verbose=0)[0][0]
    return {"img":img_path,"prob_illegal":float(pred)}

def infer_video(video_path, model, scaler, detection_method='wrist_velocity'):
    """
    Infer bowling action from video by analyzing release frame
    """
    result = extract_release_frame_and_analyze(video_path, detection_method)
    
    if result["status"] != "success":
        return {
            "video": video_path,
            "prob_illegal": None,
            "error": result["message"],
            "status": "error"
        }
    
    kp = result["keypoints"]
    Xs = scaler.transform(kp.reshape(1,-1))
    pred = model.predict(Xs, verbose=0)[0][0]
    
    return {
        "video": video_path,
        "frame_index": result["frame_index"],
        "prob_illegal": float(pred),
        "keypoints": kp,  # Include normalized keypoints
        "status": "success"
    }