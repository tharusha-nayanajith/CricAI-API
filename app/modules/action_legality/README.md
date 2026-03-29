# Action Legality

This module ports the legacy bowling-action legality model into the FastAPI
backend's shared-artifact flow.

## Inputs

- `VideoArtifacts.release_frame`
- `VideoArtifacts.release_point`
- optional `video_url` for result metadata

The module does not run its own release-frame detector. It reuses the release
frame already found by the shared preprocessor.

## Runtime flow

1. Take the raw release frame from the preprocessor output.
2. Extract pose landmarks from that frame.
3. Normalize landmarks with the legacy torso-centered feature pipeline.
4. Standardize the feature vector with the exported scaler values in
   `assets/scaler.json`.
5. Run the imported TensorFlow model in `assets/bowler_model.h5`.

## Pose extraction

The service supports two MediaPipe paths:

- Classic API: `mp.solutions.pose.Pose`
- Tasks API fallback: `PoseLandmarker`

For Tasks API deployments, provide a landmarker model at:

- `app/modules/action_legality/assets/pose_landmarker.task`

or set:

- `MEDIAPIPE_POSE_TASK_PATH`

The Tasks path attempts GPU delegate initialization first and falls back to CPU
if GPU support is unavailable.

## Assets

- `assets/bowler_model.h5`: imported legality model
- `assets/scaler.json`: exported scaler statistics
- `assets/meta.json`: model metadata
- `assets/pose_landmarker.task`: optional MediaPipe Tasks model

## Result fields

- `verdict`
- `illegal_probability`
- `legal_probability`
- `confidence`
- `release_frame_index`
- `release_timestamp_s`
- `release_confidence`
- `selected_landmarks`
- `normalized_keypoints`
- `video_url`
