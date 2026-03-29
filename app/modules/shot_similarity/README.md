# Shot Similarity

This module compares the batter pose at contact against a local reference
library of golden shots.

## Inputs

- `VideoArtifacts.bat_contact_frame`
- optional `video_url` for result metadata

If the shared preprocessor cannot produce `bat_contact_frame`, this module
fails with a `FeatureError`.

## Runtime flow

1. Reuse the shared preprocessor's bat-contact frame.
2. Extract pose landmarks from that frame.
3. Normalize the user pose and compare it against the local reference library.
4. Return the best matched player, shot type, similarity score, and coaching
   feedback.

## Reference library

Reference shots live in:

- `app/modules/shot_similarity/assets/golden_frames.json`

The checked-in file is intentionally empty until real player reference poses
are added.

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
