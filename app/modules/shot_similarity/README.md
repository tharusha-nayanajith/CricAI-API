# Shot Similarity

This module compares the batter pose at contact against a local reference
library of golden shots.

## Inputs

- `VideoArtifacts.bat_contact_frame`
- `shot_classifier.predicted_shot` to constrain the comparison set
- optional `video_url` for result metadata

If the shared preprocessor cannot produce `bat_contact_frame`, this module
fails with a `FeatureError`.

## Runtime flow

1. Reuse the shared preprocessor's bat-contact frame.
2. Reuse the shot classifier output as the canonical shot family.
3. Extract pose landmarks from that frame.
4. Compare the user pose only against references that belong to the classified
   shot family.
5. Return the best matched player, similarity score, and coaching feedback.

## Reference library

Reference shots can come from either:

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
