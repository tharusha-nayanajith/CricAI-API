from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.modules.bowler_performance.models import (
    OverlayPoint2D,
    ReleaseVisualAnalysis,
    ReleaseVisualOverlay,
)
from app.modules.preprocessor.models import ReleasePoint

LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26
LANDMARK_LEFT_ANKLE = 27
LANDMARK_RIGHT_ANKLE = 28

RELEASE_VISUAL_LANDMARKS = [
    LANDMARK_NOSE,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_RIGHT_SHOULDER,
    LANDMARK_LEFT_ELBOW,
    LANDMARK_RIGHT_ELBOW,
    LANDMARK_LEFT_WRIST,
    LANDMARK_RIGHT_WRIST,
    LANDMARK_LEFT_HIP,
    LANDMARK_RIGHT_HIP,
    LANDMARK_LEFT_KNEE,
    LANDMARK_RIGHT_KNEE,
    LANDMARK_LEFT_ANKLE,
    LANDMARK_RIGHT_ANKLE,
]


@dataclass(slots=True)
class _Point:
    x: float
    y: float


@dataclass(slots=True)
class _Landmark:
    id: int
    x: float
    y: float
    visibility: float


def build_release_visual_analysis(
    frame_bgr: np.ndarray,
    release_point: ReleasePoint | None,
) -> tuple[ReleaseVisualAnalysis | None, np.ndarray | None]:
    if frame_bgr.size == 0:
        return None, None

    landmarks = _extract_pose_landmarks(frame_bgr)
    if not landmarks:
        return None, None

    landmark_map = {landmark.id: landmark for landmark in landmarks}
    bowling_arm = _infer_bowling_arm(landmark_map, release_point)
    release_xy = _release_point_xy(release_point, landmark_map, bowling_arm)
    shoulder_mid = _midpoint(
        landmark_map.get(LANDMARK_LEFT_SHOULDER),
        landmark_map.get(LANDMARK_RIGHT_SHOULDER),
    )
    hip_mid = _midpoint(
        landmark_map.get(LANDMARK_LEFT_HIP),
        landmark_map.get(LANDMARK_RIGHT_HIP),
    )
    head_point = landmark_map.get(LANDMARK_NOSE)
    neck_point = shoulder_mid if shoulder_mid is not None else None
    torso_anchor = _point_from_landmark(head_point) if head_point is not None else neck_point
    torso_length = _distance(torso_anchor, hip_mid) if torso_anchor is not None and hip_mid is not None else None
    if torso_length is not None and torso_length < 1e-3:
        torso_length = None

    overlays: list[ReleaseVisualOverlay] = []

    arm_overlay = _build_release_arm_overlay(landmark_map, release_xy, bowling_arm)
    if arm_overlay is not None:
        overlays.append(arm_overlay)

    body_lean_overlay = _build_body_lean_overlay(torso_anchor, hip_mid)
    if body_lean_overlay is not None:
        overlays.append(body_lean_overlay)

    front_leg_overlay = _build_front_leg_overlay(landmark_map, bowling_arm)
    if front_leg_overlay is not None:
        overlays.append(front_leg_overlay)

    release_hip_overlay = _build_release_to_hip_overlay(release_xy, hip_mid, torso_length)
    if release_hip_overlay is not None:
        overlays.append(release_hip_overlay)

    if not overlays:
        return None, None

    overlay_image = _render_release_overlay(frame_bgr, overlays)
    summary_notes = [overlay.observation for overlay in overlays]
    height, width = frame_bgr.shape[:2]
    return (
        ReleaseVisualAnalysis(
            release_frame_width=width,
            release_frame_height=height,
            bowling_arm=bowling_arm,
            overlays=overlays,
            summary_notes=summary_notes,
        ),
        overlay_image,
    )


def _extract_pose_landmarks(frame_bgr: np.ndarray) -> list[_Landmark]:
    from app.modules.action_legality.service import _extract_pose_landmarks_2d

    points = _extract_pose_landmarks_2d(frame_bgr, RELEASE_VISUAL_LANDMARKS)
    return [
        _Landmark(id=point.id, x=point.x, y=point.y, visibility=point.visibility)
        for point in points
    ]


def _infer_bowling_arm(landmarks: dict[int, _Landmark], release_point: ReleasePoint | None) -> str | None:
    left = landmarks.get(LANDMARK_LEFT_WRIST)
    right = landmarks.get(LANDMARK_RIGHT_WRIST)
    if left is None and right is None:
        return None
    if release_point is not None:
        release_xy = _Point(*release_point.hand_position)
        if left is not None and right is not None:
            return (
                "left"
                if _distance(_point_from_landmark(left), release_xy)
                <= _distance(_point_from_landmark(right), release_xy)
                else "right"
            )
        if left is not None:
            return "left"
        return "right"
    if left is not None and right is not None:
        return "left" if left.visibility >= right.visibility else "right"
    return "left" if left is not None else "right"


def _release_point_xy(
    release_point: ReleasePoint | None,
    landmarks: dict[int, _Landmark],
    bowling_arm: str | None,
) -> _Point | None:
    if release_point is not None:
        return _Point(*release_point.hand_position)
    wrist_landmark = None
    if bowling_arm == "left":
        wrist_landmark = landmarks.get(LANDMARK_LEFT_WRIST)
    elif bowling_arm == "right":
        wrist_landmark = landmarks.get(LANDMARK_RIGHT_WRIST)
    return _point_from_landmark(wrist_landmark)


def _build_release_arm_overlay(
    landmarks: dict[int, _Landmark],
    release_xy: _Point | None,
    bowling_arm: str | None,
) -> ReleaseVisualOverlay | None:
    if bowling_arm == "left":
        shoulder = _point_from_landmark(landmarks.get(LANDMARK_LEFT_SHOULDER))
        elbow = _point_from_landmark(landmarks.get(LANDMARK_LEFT_ELBOW))
        wrist = _point_from_landmark(landmarks.get(LANDMARK_LEFT_WRIST))
    else:
        shoulder = _point_from_landmark(landmarks.get(LANDMARK_RIGHT_SHOULDER))
        elbow = _point_from_landmark(landmarks.get(LANDMARK_RIGHT_ELBOW))
        wrist = _point_from_landmark(landmarks.get(LANDMARK_RIGHT_WRIST))
    if shoulder is None or elbow is None or wrist is None or release_xy is None:
        return None

    angle = _angle_from_vertical(shoulder, release_xy)
    if angle <= 18.0:
        status, color, observation, reason, recommendation = (
            "ok",
            "#00C853",
            "Your release arm is staying on a high and upright path at release.",
            "A higher release arm usually supports better bounce, seam position, and bowling alignment.",
            "Keep repeating this arm path and hold the shoulder line tall into release.",
        )
    elif angle <= 32.0:
        status, color, observation, reason, recommendation = (
            "warning",
            "#FFB300",
            "Your release arm is dropping slightly lower than the ideal release line.",
            "A lower arm path can flatten the release and make pace and length harder to repeat.",
            "Try to lift the bowling arm earlier into release and feel the hand finishing higher above the shoulder.",
        )
    else:
        status, color, observation, reason, recommendation = (
            "critical",
            "#E53935",
            "Your release arm is falling well away from a high release line.",
            "This can reduce bounce and make the release point drift away from the target line.",
            "Work on a taller arm path, stronger front-side lift, and a more upright shoulder position through release.",
        )
    return ReleaseVisualOverlay(
        overlay_id="release_arm_line",
        label="Release Arm Line",
        status=status,
        color_hex=color,
        points=[_to_overlay_point(point) for point in [shoulder, elbow, wrist, release_xy]],
        observation=observation,
        reason=reason,
        recommendation=recommendation,
        metric_value=round(angle, 1),
        metric_label="arm_angle_from_vertical_deg",
    )


def _build_body_lean_overlay(
    head_or_neck: _Point | None,
    hip_mid: _Point | None,
) -> ReleaseVisualOverlay | None:
    if head_or_neck is None or hip_mid is None:
        return None
    lean = _angle_from_vertical(head_or_neck, hip_mid)
    direction = "right" if hip_mid.x > head_or_neck.x else "left"
    if lean <= 8.0:
        status, color, observation, reason, recommendation = (
            "ok",
            "#00C853",
            "Your body is staying balanced over the hips during release.",
            "A stacked trunk helps you control line more consistently and keeps the release stable.",
            "Keep the head centered and stay tall over the front side through release.",
        )
    elif lean <= 16.0:
        status, color, observation, reason, recommendation = (
            "warning",
            "#FFB300",
            f"Your body is falling slightly to the {direction} during release.",
            "This can reduce control and make it harder to bowl a consistent line.",
            f"Try to keep your head and front shoulder closer to the target line at release. Focus on landing with a stronger front leg and staying tall through the action.",
        )
    else:
        status, color, observation, reason, recommendation = (
            "critical",
            "#E53935",
            f"Your body is falling strongly to the {direction} at release.",
            "Excessive side fall can pull the ball away from the target and make the action harder to repeat under pressure.",
            "Stabilize the trunk, brace the front side earlier, and keep the head and chest driving more directly toward the target line.",
        )
    return ReleaseVisualOverlay(
        overlay_id="body_lean_line",
        label="Body Lean Line",
        status=status,
        color_hex=color,
        points=[_to_overlay_point(head_or_neck), _to_overlay_point(hip_mid)],
        observation=observation,
        reason=reason,
        recommendation=recommendation,
        metric_value=round(lean, 1),
        metric_label="body_lean_deg",
    )


def _build_front_leg_overlay(
    landmarks: dict[int, _Landmark],
    bowling_arm: str | None,
) -> ReleaseVisualOverlay | None:
    is_left_front_leg = bowling_arm != "left"
    hip_id = LANDMARK_LEFT_HIP if is_left_front_leg else LANDMARK_RIGHT_HIP
    knee_id = LANDMARK_LEFT_KNEE if is_left_front_leg else LANDMARK_RIGHT_KNEE
    ankle_id = LANDMARK_LEFT_ANKLE if is_left_front_leg else LANDMARK_RIGHT_ANKLE
    hip = _point_from_landmark(landmarks.get(hip_id))
    knee = _point_from_landmark(landmarks.get(knee_id))
    ankle = _point_from_landmark(landmarks.get(ankle_id))
    if hip is None or knee is None or ankle is None:
        return None

    knee_angle = _joint_angle(hip, knee, ankle)
    if knee_angle >= 160.0:
        status, color, observation, reason, recommendation = (
            "ok",
            "#00C853",
            "Your front leg looks well braced at release.",
            "A firm front leg helps transfer momentum upward and supports a stable release point.",
            "Keep driving up into that front-leg block and hold the knee firm through release.",
        )
    elif knee_angle >= 145.0:
        status, color, observation, reason, recommendation = (
            "warning",
            "#FFB300",
            "Your front leg is soft at release.",
            "A softer front side can leak energy and make pace, bounce, and alignment less repeatable.",
            "Brace the front side more firmly and feel the front knee holding shape as you release the ball.",
        )
    else:
        status, color, observation, reason, recommendation = (
            "critical",
            "#E53935",
            "Your front leg is collapsing through release.",
            "That collapse reduces force transfer and often pulls the body away from a strong, stable release position.",
            "Work on a stronger front-leg block, land more solidly, and keep the knee firmer through the release phase.",
        )
    return ReleaseVisualOverlay(
        overlay_id="front_leg_angle",
        label="Front Leg Angle",
        status=status,
        color_hex=color,
        points=[_to_overlay_point(point) for point in [hip, knee, ankle]],
        observation=observation,
        reason=reason,
        recommendation=recommendation,
        metric_value=round(knee_angle, 1),
        metric_label="front_knee_angle_deg",
    )


def _build_release_to_hip_overlay(
    release_xy: _Point | None,
    hip_mid: _Point | None,
    torso_length: float | None,
) -> ReleaseVisualOverlay | None:
    if release_xy is None or hip_mid is None or torso_length is None:
        return None
    normalized_offset = abs(release_xy.x - hip_mid.x) / max(torso_length, 1e-6)
    if normalized_offset <= 0.18:
        status, color, observation, reason, recommendation = (
            "ok",
            "#00C853",
            "There is a fairly straight line from the release point to the hip.",
            "That stacked alignment usually supports cleaner energy transfer and a more repeatable release path.",
            "Maintain that stacked alignment and keep the release hand working close to the body line.",
        )
    elif normalized_offset <= 0.32:
        status, color, observation, reason, recommendation = (
            "warning",
            "#FFB300",
            "Your release point is drifting a little away from the hip line.",
            "That drift can make the action less efficient and can move the ball away from the intended line.",
            "Stay tighter through the core during release and keep the hand path closer to the hip line.",
        )
    else:
        status, color, observation, reason, recommendation = (
            "critical",
            "#E53935",
            "Your release point is well outside the hip line.",
            "This can break body alignment and make control and repeatability much harder.",
            "Improve trunk control, keep the chest more stacked, and work on bringing the arm path back closer to the body line.",
        )
    return ReleaseVisualOverlay(
        overlay_id="release_to_hip_alignment",
        label="Release To Hip Alignment",
        status=status,
        color_hex=color,
        points=[_to_overlay_point(release_xy), _to_overlay_point(hip_mid)],
        observation=observation,
        reason=reason,
        recommendation=recommendation,
        metric_value=round(normalized_offset, 3),
        metric_label="release_hip_offset_ratio",
    )


def _render_release_overlay(
    frame_bgr: np.ndarray,
    overlays: list[ReleaseVisualOverlay],
) -> np.ndarray:
    rendered = frame_bgr.copy()
    for overlay in overlays:
        color = _hex_to_bgr(overlay.color_hex)
        for start, end in zip(overlay.points, overlay.points[1:], strict=False):
            cv2.line(
                rendered,
                (int(round(start.x)), int(round(start.y))),
                (int(round(end.x)), int(round(end.y))),
                color,
                3,
                lineType=cv2.LINE_AA,
            )
        for point in overlay.points:
            cv2.circle(
                rendered,
                (int(round(point.x)), int(round(point.y))),
                5,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
    return rendered


def _midpoint(left: _Landmark | None, right: _Landmark | None) -> _Point | None:
    if left is None or right is None:
        return None
    return _Point(x=(left.x + right.x) / 2.0, y=(left.y + right.y) / 2.0)


def _to_overlay_point(point: _Point) -> OverlayPoint2D:
    return OverlayPoint2D(x=float(point.x), y=float(point.y))


def _point_from_landmark(landmark: _Landmark | None) -> _Point | None:
    if landmark is None:
        return None
    return _Point(x=landmark.x, y=landmark.y)


def _distance(point_a: _Point | None, point_b: _Point | None) -> float:
    if point_a is None or point_b is None:
        return float("inf")
    return math.hypot(point_b.x - point_a.x, point_b.y - point_a.y)


def _joint_angle(point_a: _Point, point_b: _Point, point_c: _Point) -> float:
    vector_ba = np.asarray([point_a.x - point_b.x, point_a.y - point_b.y], dtype=np.float32)
    vector_bc = np.asarray([point_c.x - point_b.x, point_c.y - point_b.y], dtype=np.float32)
    norm_product = float(np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
    if norm_product < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(vector_ba, vector_bc) / norm_product, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _angle_from_vertical(point_a: _Point, point_b: _Point) -> float:
    delta_x = point_b.x - point_a.x
    delta_y = point_b.y - point_a.y
    return abs(math.degrees(math.atan2(delta_x, delta_y if abs(delta_y) > 1e-6 else 1e-6)))


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    value = color_hex.lstrip("#")
    if len(value) != 6:
        return (0, 255, 0)
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return (blue, green, red)
