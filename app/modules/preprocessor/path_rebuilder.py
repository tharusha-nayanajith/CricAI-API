from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from loguru import logger

from app.modules.preprocessor.constants import (
    TRACK_REBUILDER_ACCELERATION_WEIGHT,
    TRACK_REBUILDER_CONTINUITY_CONF_WEIGHT,
    TRACK_REBUILDER_DISTANCE_WEIGHT,
    TRACK_REBUILDER_INLIER_THRESH,
    TRACK_REBUILDER_MAX_FRAME_GAP,
    TRACK_REBUILDER_MAX_PREDICTION_ERROR,
    TRACK_REBUILDER_MAX_STEP_PIXELS,
    TRACK_REBUILDER_MAX_UNSUPPORTED_RUN,
    TRACK_REBUILDER_MIN_INLIERS,
    TRACK_REBUILDER_MIN_POST_BOUNCE_DETS,
    TRACK_REBUILDER_MIN_PRE_BOUNCE_DETS,
    TRACK_REBUILDER_MIN_SEGMENT_INLIERS,
    TRACK_REBUILDER_MIN_STABLE_SUPPORT,
    TRACK_REBUILDER_MIN_TAIL_START_FRAMES,
    TRACK_REBUILDER_PREDICTION_WEIGHT,
    TRACK_REBUILDER_RANSAC_ITERS,
    TRACK_REBUILDER_START_CONF_WEIGHT,
    TRACK_REBUILDER_START_WINDOW,
    TRACK_REBUILDER_SUPPORT_THRESH,
    TRACK_REBUILDER_TRACK_BEAM_WIDTH,
)
from app.modules.preprocessor.models import BallDetection, FrameBallDetections


@dataclass(slots=True)
class Parabola:
    a: float
    b: float
    c: float

    def predict(self, t_value: float) -> float:
        return self.a * t_value**2 + self.b * t_value + self.c


@dataclass(slots=True)
class SegmentFit:
    inliers: list[BallDetection]
    para_x: Parabola
    para_y: Parabola


@dataclass(slots=True)
class TrackState:
    detections: tuple[BallDetection, ...]
    score: float
    sum_confidence: float


class TransitionScore(NamedTuple):
    score: float
    prediction_error: float


class DeliveryPathRebuilder:
    def rebuild(
        self,
        raw_candidates: list[BallDetection],
        fps: float,
        grouped_candidates: list[FrameBallDetections] | None = None,
        roi_entry_frame_idx: int | None = None,
    ) -> list[BallDetection] | None:
        if len(raw_candidates) < TRACK_REBUILDER_MIN_INLIERS:
            return None

        selected_track = self._select_track(raw_candidates)
        if len(selected_track) < TRACK_REBUILDER_MIN_INLIERS:
            logger.info(
                "Delivery path rebuild skipped: selected track too short ({})",
                len(selected_track),
            )
            return None

        ordered_track = sorted(selected_track, key=lambda detection: detection.frame_idx)
        bounce_frame, _bounce_t = self._detect_bounce(ordered_track, fps)
        pre_track, post_track = self._split_at_bounce(ordered_track, bounce_frame)

        pre_fit = self._fit_segment(pre_track, min_inliers=TRACK_REBUILDER_MIN_INLIERS)
        if pre_fit is None:
            logger.info(
                "Delivery path rebuild skipped: pre-bounce fit failed ({})",
                len(pre_track),
            )
            return None

        post_fit = self._fit_segment(
            post_track,
            min_inliers=TRACK_REBUILDER_MIN_SEGMENT_INLIERS,
        )
        candidate_frames = grouped_candidates or self._grouped_frames_from_raw(raw_candidates)
        end_frame = self._find_supported_end_frame(
            ordered_track,
            candidate_frames,
            pre_fit,
            post_fit,
            bounce_frame,
            roi_entry_frame_idx,
        )
        rebuilt_path = self._sample_piecewise_path(
            ordered_track,
            candidate_frames,
            fps,
            pre_fit,
            post_fit,
            bounce_frame,
            end_frame,
        )
        if len(rebuilt_path) < 3:
            return None

        logger.info(
            "Delivery path rebuilt raw_candidates={} selected_track={} "
            "rebuilt={} bounce_frame={} end_frame={}",
            len(raw_candidates),
            len(selected_track),
            len(rebuilt_path),
            bounce_frame,
            end_frame,
        )
        return rebuilt_path

    def _select_track(self, raw_path: list[BallDetection]) -> list[BallDetection]:
        if not raw_path:
            return []

        frame_groups = self._group_candidates_by_frame(raw_path)
        segments = self._split_frame_groups(frame_groups)
        segment_tracks = [self._select_track_in_segment(segment) for segment in segments]
        earliest_frame = frame_groups[0][0].frame_idx
        return self._choose_track(segment_tracks, earliest_frame)

    def _choose_track(
        self,
        tracks: list[list[BallDetection]],
        earliest_frame: int,
    ) -> list[BallDetection]:
        if not tracks:
            return []

        eligible_tracks = [
            track for track in tracks if len(track) >= TRACK_REBUILDER_MIN_INLIERS
        ]
        candidate_tracks = eligible_tracks or [track for track in tracks if track]
        if not candidate_tracks:
            return []

        preferred_tracks = [
            track
            for track in candidate_tracks
            if track[0].frame_idx - earliest_frame <= TRACK_REBUILDER_START_WINDOW
        ]
        return max(preferred_tracks or candidate_tracks, key=self._track_key)

    def _select_track_in_segment(
        self,
        frame_groups: list[list[BallDetection]],
    ) -> list[BallDetection]:
        active_states: list[TrackState] = []
        seen_states: list[TrackState] = []
        earliest_frame = frame_groups[0][0].frame_idx if frame_groups else 0

        for frame_candidates in frame_groups:
            next_states = [
                TrackState(
                    detections=(candidate,),
                    score=candidate.confidence * TRACK_REBUILDER_START_CONF_WEIGHT,
                    sum_confidence=candidate.confidence,
                )
                for candidate in frame_candidates
            ]

            for state in active_states:
                for candidate in frame_candidates:
                    transition = self._score_transition(state, candidate)
                    if transition is None:
                        continue
                    next_states.append(
                        TrackState(
                            detections=state.detections + (candidate,),
                            score=state.score + transition.score,
                            sum_confidence=state.sum_confidence + candidate.confidence,
                        )
                    )

            active_states = self._prune_states(next_states)
            seen_states.extend(active_states)

        if not seen_states:
            return []
        best_state = self._choose_state(seen_states, earliest_frame)
        return list(best_state.detections) if best_state is not None else []

    def _choose_state(
        self,
        states: list[TrackState],
        earliest_frame: int,
    ) -> TrackState | None:
        if not states:
            return None

        eligible_states = [
            state
            for state in states
            if len(state.detections) >= TRACK_REBUILDER_MIN_INLIERS
        ]
        candidate_states = eligible_states or states
        preferred_states = [
            state
            for state in candidate_states
            if state.detections[0].frame_idx - earliest_frame <= TRACK_REBUILDER_START_WINDOW
        ]
        return max(preferred_states or candidate_states, key=self._track_state_key)

    @staticmethod
    def _group_candidates_by_frame(
        raw_path: list[BallDetection],
    ) -> list[list[BallDetection]]:
        ordered = sorted(
            raw_path,
            key=lambda detection: (
                detection.frame_idx,
                detection.x,
                detection.y,
            ),
        )
        frame_groups: list[list[BallDetection]] = []
        current_group: list[BallDetection] = []
        current_frame: int | None = None

        for detection in ordered:
            if current_frame != detection.frame_idx:
                if current_group:
                    frame_groups.append(
                        sorted(
                            current_group,
                            key=lambda item: item.confidence,
                            reverse=True,
                        )
                    )
                current_group = [detection]
                current_frame = detection.frame_idx
            else:
                current_group.append(detection)

        if current_group:
            frame_groups.append(
                sorted(
                    current_group,
                    key=lambda item: item.confidence,
                    reverse=True,
                )
            )

        return frame_groups

    @staticmethod
    def _grouped_frames_from_raw(
        raw_path: list[BallDetection],
    ) -> list[FrameBallDetections]:
        return [
            FrameBallDetections(
                frame_idx=group[0].frame_idx,
                timestamp_s=group[0].timestamp_s,
                detections=list(group),
            )
            for group in DeliveryPathRebuilder._group_candidates_by_frame(raw_path)
        ]

    @staticmethod
    def _split_frame_groups(
        frame_groups: list[list[BallDetection]],
    ) -> list[list[list[BallDetection]]]:
        if not frame_groups:
            return []

        segments: list[list[list[BallDetection]]] = [[frame_groups[0]]]
        for previous_group, current_group in zip(
            frame_groups,
            frame_groups[1:],
            strict=False,
        ):
            previous_frame = previous_group[0].frame_idx
            current_frame = current_group[0].frame_idx
            if current_frame - previous_frame > TRACK_REBUILDER_MAX_FRAME_GAP:
                segments.append([current_group])
            else:
                segments[-1].append(current_group)
        return segments

    def _score_transition(
        self,
        state: TrackState,
        current: BallDetection,
    ) -> TransitionScore | None:
        last = state.detections[-1]
        frame_gap = current.frame_idx - last.frame_idx
        if frame_gap <= 0 or frame_gap > TRACK_REBUILDER_MAX_FRAME_GAP:
            return None

        step_distance = float(np.hypot(current.x - last.x, current.y - last.y))
        step_speed = step_distance / frame_gap
        if step_speed > TRACK_REBUILDER_MAX_STEP_PIXELS:
            return None

        if len(state.detections) == 1:
            score = (
                current.confidence * TRACK_REBUILDER_CONTINUITY_CONF_WEIGHT
                - step_distance * TRACK_REBUILDER_DISTANCE_WEIGHT
            )
            return TransitionScore(score=score, prediction_error=step_distance)

        prev = state.detections[-2]
        prev_gap = last.frame_idx - prev.frame_idx
        if prev_gap <= 0:
            return None

        prev_dx = (last.x - prev.x) / prev_gap
        prev_dy = (last.y - prev.y) / prev_gap
        predicted_x = last.x + prev_dx * frame_gap
        predicted_y = last.y + prev_dy * frame_gap
        prediction_error = float(
            np.hypot(current.x - predicted_x, current.y - predicted_y)
        )
        if prediction_error > TRACK_REBUILDER_MAX_PREDICTION_ERROR:
            return None

        prev_speed = float(np.hypot(prev_dx, prev_dy))
        acceleration_error = abs(step_speed - prev_speed)
        score = (
            current.confidence * TRACK_REBUILDER_CONTINUITY_CONF_WEIGHT
            - prediction_error * TRACK_REBUILDER_PREDICTION_WEIGHT
            - acceleration_error * TRACK_REBUILDER_ACCELERATION_WEIGHT
        )
        return TransitionScore(score=score, prediction_error=prediction_error)

    @staticmethod
    def _prune_states(states: list[TrackState]) -> list[TrackState]:
        if not states:
            return []
        unique_states: dict[tuple[int, int, int, int], TrackState] = {}
        for state in sorted(
            states,
            key=DeliveryPathRebuilder._track_state_key,
            reverse=True,
        ):
            last = state.detections[-1]
            signature = (
                len(state.detections),
                last.frame_idx,
                int(round(last.x * 10.0)),
                int(round(last.y * 10.0)),
            )
            if signature not in unique_states:
                unique_states[signature] = state
            if len(unique_states) >= TRACK_REBUILDER_TRACK_BEAM_WIDTH:
                break
        return list(unique_states.values())

    @staticmethod
    def _track_key(track: list[BallDetection]) -> tuple[int, float, float, int]:
        if not track:
            return (0, float("-inf"), float("-inf"), 0)
        mean_conf = float(
            np.mean([detection.confidence for detection in track], dtype=np.float64)
        )
        total_conf = float(
            np.sum([detection.confidence for detection in track], dtype=np.float64)
        )
        return (len(track), mean_conf, total_conf, -track[0].frame_idx)

    @staticmethod
    def _track_state_key(state: TrackState) -> tuple[int, float, float, int]:
        mean_conf = state.sum_confidence / len(state.detections)
        return (
            len(state.detections),
            state.score,
            mean_conf,
            -state.detections[0].frame_idx,
        )

    def _fit_3pt(self, pts: list[tuple[float, float]]) -> Parabola | None:
        if len(pts) != 3:
            return None

        matrix = np.array(
            [[t_value**2, t_value, 1.0] for t_value, _ in pts],
            dtype=np.float64,
        )
        values = np.array([coord for _, coord in pts], dtype=np.float64)
        try:
            coeffs = np.linalg.solve(matrix, values)
        except np.linalg.LinAlgError:
            return None
        return Parabola(a=float(coeffs[0]), b=float(coeffs[1]), c=float(coeffs[2]))

    def _fit_lstsq(self, pts: list[tuple[float, float]]) -> Parabola | None:
        if len(pts) < 3:
            return None

        matrix = np.array(
            [[t_value**2, t_value, 1.0] for t_value, _ in pts],
            dtype=np.float64,
        )
        values = np.array([coord for _, coord in pts], dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        return Parabola(a=float(coeffs[0]), b=float(coeffs[1]), c=float(coeffs[2]))

    def _fit_segment(
        self,
        detections: list[BallDetection],
        min_inliers: int,
    ) -> SegmentFit | None:
        if len(detections) < max(3, min_inliers):
            return None

        best_inliers: list[BallDetection] = []
        best_error = float("inf")
        rng = random.Random(0)

        for _ in range(TRACK_REBUILDER_RANSAC_ITERS):
            sample = rng.sample(detections, 3)
            para_x = self._fit_3pt([(point.timestamp_s, point.x) for point in sample])
            para_y = self._fit_3pt([(point.timestamp_s, point.y) for point in sample])
            if para_x is None or para_y is None:
                continue

            inliers, mean_error = self._collect_inliers(detections, para_x, para_y)
            if len(inliers) > len(best_inliers) or (
                len(inliers) == len(best_inliers) and mean_error < best_error
            ):
                best_inliers = inliers
                best_error = mean_error

        if len(best_inliers) < min_inliers:
            return None

        para_x = self._fit_lstsq([(point.timestamp_s, point.x) for point in best_inliers])
        para_y = self._fit_lstsq([(point.timestamp_s, point.y) for point in best_inliers])
        if para_x is None or para_y is None:
            return None

        refined_inliers, _ = self._collect_inliers(detections, para_x, para_y)
        if len(refined_inliers) >= min_inliers:
            best_inliers = refined_inliers

        return SegmentFit(
            inliers=sorted(best_inliers, key=lambda detection: detection.frame_idx),
            para_x=para_x,
            para_y=para_y,
        )

    @staticmethod
    def _split_at_bounce(
        detections: list[BallDetection],
        bounce_frame: int | None,
    ) -> tuple[list[BallDetection], list[BallDetection]]:
        if bounce_frame is None:
            return detections, []

        pre_bounce = [
            detection for detection in detections if detection.frame_idx <= bounce_frame
        ]
        post_bounce = [
            detection for detection in detections if detection.frame_idx >= bounce_frame
        ]
        return pre_bounce, post_bounce

    def _detect_bounce(
        self,
        detections: list[BallDetection],
        fps: float,
    ) -> tuple[int | None, float | None]:
        _ = fps
        min_required = (
            TRACK_REBUILDER_MIN_PRE_BOUNCE_DETS
            + TRACK_REBUILDER_MIN_POST_BOUNCE_DETS
        )
        if len(detections) < min_required:
            return None, None

        ordered = sorted(detections, key=lambda detection: detection.frame_idx)
        candidate_indices = range(
            TRACK_REBUILDER_MIN_PRE_BOUNCE_DETS - 1,
            len(ordered) - TRACK_REBUILDER_MIN_POST_BOUNCE_DETS,
        )
        best_idx: int | None = None
        best_score = float("-inf")

        for idx in candidate_indices:
            pre_start = idx - (TRACK_REBUILDER_MIN_PRE_BOUNCE_DETS - 1)
            pre_points = ordered[pre_start : idx + 1]
            post_points = ordered[idx : idx + TRACK_REBUILDER_MIN_POST_BOUNCE_DETS + 1]
            pre_slope = self._segment_slope(pre_points[0], pre_points[-1])
            post_slope = self._segment_slope(post_points[0], post_points[-1])
            if pre_slope > 0.0 and post_slope < 0.0:
                score = pre_slope - post_slope
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx is None:
            candidate_points = [ordered[idx] for idx in candidate_indices]
            if not candidate_points:
                return None, None
            bounce_point = max(candidate_points, key=lambda detection: detection.y)
            return bounce_point.frame_idx, bounce_point.timestamp_s

        bounce_point = ordered[best_idx]
        return bounce_point.frame_idx, bounce_point.timestamp_s

    def _collect_inliers(
        self,
        detections: list[BallDetection],
        para_x: Parabola,
        para_y: Parabola,
    ) -> tuple[list[BallDetection], float]:
        inliers: list[BallDetection] = []
        errors: list[float] = []
        for detection in detections:
            error = self._residual(detection, para_x, para_y)
            if error <= TRACK_REBUILDER_INLIER_THRESH:
                inliers.append(detection)
                errors.append(error)
        mean_error = float(np.mean(errors)) if errors else float("inf")
        return inliers, mean_error

    @staticmethod
    def _residual(
        detection: BallDetection,
        para_x: Parabola,
        para_y: Parabola,
    ) -> float:
        pred_x = para_x.predict(detection.timestamp_s)
        pred_y = para_y.predict(detection.timestamp_s)
        return float(np.hypot(detection.x - pred_x, detection.y - pred_y))

    @staticmethod
    def _segment_slope(first: BallDetection, second: BallDetection) -> float:
        dt = second.timestamp_s - first.timestamp_s
        if dt <= 0.0:
            return 0.0
        return (second.y - first.y) / dt

    def _find_supported_end_frame(
        self,
        ordered_track: list[BallDetection],
        grouped_candidates: list[FrameBallDetections],
        pre_fit: SegmentFit,
        post_fit: SegmentFit | None,
        bounce_frame: int | None,
        roi_entry_frame_idx: int | None,
    ) -> int:
        grouped_by_frame = {
            frame.frame_idx: frame.detections for frame in grouped_candidates
        }
        first_frame = ordered_track[0].frame_idx
        last_frame = ordered_track[-1].frame_idx
        last_supported_frame = last_frame
        stable_support = 0
        unsupported_run = 0
        stop_gate_frame = max(
            first_frame + TRACK_REBUILDER_MIN_TAIL_START_FRAMES,
            roi_entry_frame_idx if roi_entry_frame_idx is not None else first_frame,
            bounce_frame if bounce_frame is not None else first_frame,
        )

        for frame_idx in range(first_frame, last_frame + 1):
            candidates = grouped_by_frame.get(frame_idx, [])
            supported = self._frame_has_support(
                frame_idx,
                candidates,
                pre_fit,
                post_fit,
                bounce_frame,
            )
            if supported:
                last_supported_frame = frame_idx
                stable_support += 1
                unsupported_run = 0
                continue

            if (
                stable_support < TRACK_REBUILDER_MIN_STABLE_SUPPORT
                or frame_idx < stop_gate_frame
            ):
                continue

            unsupported_run += 1
            if unsupported_run >= TRACK_REBUILDER_MAX_UNSUPPORTED_RUN:
                return last_supported_frame

        return last_supported_frame

    def _frame_has_support(
        self,
        frame_idx: int,
        candidates: list[BallDetection],
        pre_fit: SegmentFit,
        post_fit: SegmentFit | None,
        bounce_frame: int | None,
    ) -> bool:
        if not candidates:
            return False
        predicted_x, predicted_y = self._predict_xy(
            frame_idx,
            candidates[0].timestamp_s,
            pre_fit,
            post_fit,
            bounce_frame,
        )
        return any(
            np.hypot(candidate.x - predicted_x, candidate.y - predicted_y)
            <= TRACK_REBUILDER_SUPPORT_THRESH
            for candidate in candidates
        )

    def _sample_piecewise_path(
        self,
        ordered_track: list[BallDetection],
        grouped_candidates: list[FrameBallDetections],
        fps: float,
        pre_fit: SegmentFit,
        post_fit: SegmentFit | None,
        bounce_frame: int | None,
        end_frame: int,
    ) -> list[BallDetection]:
        grouped_by_frame = {
            frame.frame_idx: frame.detections for frame in grouped_candidates
        }
        start_frame = ordered_track[0].frame_idx
        default_conf = float(
            np.mean([detection.confidence for detection in ordered_track], dtype=np.float64)
        )
        fitted: list[BallDetection] = []

        for frame_idx in range(start_frame, end_frame + 1):
            timestamp_s = frame_idx / fps if fps > 0 else frame_idx / 30.0
            predicted_x, predicted_y = self._predict_xy(
                frame_idx,
                timestamp_s,
                pre_fit,
                post_fit,
                bounce_frame,
            )
            frame_candidates = grouped_by_frame.get(frame_idx, [])
            confidence = default_conf
            if frame_candidates:
                nearest = min(
                    frame_candidates,
                    key=lambda detection: np.hypot(
                        detection.x - predicted_x,
                        detection.y - predicted_y,
                    ),
                )
                confidence = float(nearest.confidence)

            fitted.append(
                BallDetection(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    x=float(predicted_x),
                    y=float(predicted_y),
                    confidence=confidence,
                )
            )
        return fitted

    def _predict_xy(
        self,
        frame_idx: int,
        timestamp_s: float,
        pre_fit: SegmentFit,
        post_fit: SegmentFit | None,
        bounce_frame: int | None,
    ) -> tuple[float, float]:
        active_fit = pre_fit
        if post_fit is not None and bounce_frame is not None and frame_idx > bounce_frame:
            active_fit = post_fit
        return active_fit.para_x.predict(timestamp_s), active_fit.para_y.predict(
            timestamp_s
        )
