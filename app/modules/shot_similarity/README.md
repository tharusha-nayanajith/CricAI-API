# Shot Similarity

This module compares the batter's shot motion against local reference shot
sequences.

## Inputs

- `VideoArtifacts.standardized_video_path` for sequence extraction
- fallback `VideoArtifacts.bat_contact_frame` when a standardized video is not available
- `shot_classifier.predicted_shot` to constrain the comparison set
- optional `video_url` for result metadata

If the shared preprocessor cannot produce a standardized video or contact-frame
fallback, this module fails with a `FeatureError`.

## Runtime flow

1. Reuse the shared preprocessor's standardized batting video.
2. Extract 30 evenly spaced frames with OpenCV.
3. Extract 33 MediaPipe pose landmarks per frame, using zero-filled frames when
   no pose is detected.
4. Normalize each pose by hip center, shoulder distance, and shoulder rotation.
5. Reuse the shot classifier output as the canonical shot family.
6. Compare the user sequence only against references that belong to the classified
   shot family.
7. Align user/reference timing with Dynamic Time Warping.
8. Return the best matched player, similarity score, coaching feedback, and
   optional artifact URLs.

## Reference library

Reference shots can come from:

- `app/modules/shot_similarity/assets/shots/*.json` for bundled 30-frame shot references
- `SHOT_SIMILARITY_REFERENCE_DIR` for external multi-frame JSON references
- `app/modules/shot_similarity/assets/golden_frames.json` for legacy single-frame references

External JSON files are expected to contain a top-level `frames` array where
 each frame is a list of pose landmarks with `x`, `y`, `z`, and optional
 `visibility`.

## Important design note

This module does not port the old standalone YOLO plus audio impact detector.
Impact detection remains owned by the shared preprocessor, and shot similarity
consumes the existing `bat_contact_frame`.

## Result fields

- `similarity_percentage`
- `matched_player`
- `shot_type`
- `keypoints_detected`
- `confidence`
- `feedback`
- `compared_frame`
- `video_url`
- `ai_feedback`
- `visualization_video_url`
- `normalized_user_url`
- `normalized_reference_url`
