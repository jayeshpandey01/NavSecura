"""Per-track visit lifecycle state machine with hysteresis and grace period management.

Enforces the 4-state lifecycle (OUTSIDE -> INSIDE -> GRACE -> CLOSED)
with mathematical rigor:
- Entry hysteresis: eliminates boundary flicker (Failure Mode F4).
- Grace period countdown: absorbs detector dropouts and brief occlusions (F1, F2).
- Duration invariant: duration = t_last_seen_inside - t_enter (never inflated by grace).
- Minimum dwell filtering: discards transient false positives (F6).
- Deterministic EOF flush: force-closes active visits on stream end (F7).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from navsecura.events.models import CloseReason, Visit, VisitState
from navsecura.roi.geometry import PolygonROI
from navsecura.track.base import Tracklet

logger = logging.getLogger("navsecura.events.state_machine")


class TrackVisitSession:
    """Tracks the lifecycle of an individual tracklet with respect to an ROI."""

    def __init__(
        self,
        track_id: int,
        roi_id: str,
        camera_id: str = "camera_0",
        enter_consecutive_frames: int = 3,
        grace_period_s: float = 2.0,
        min_dwell_s: float = 1.0,
    ) -> None:
        """Initialize per-track visit session.

        Args:
            track_id: Temporary multi-object tracker ID.
            roi_id: Target Region of Interest identifier.
            camera_id: Video stream/sensor identifier.
            enter_consecutive_frames: Required consecutive in-ROI frames to confirm entry.
            grace_period_s: Permitted absence duration before closing the visit.
            min_dwell_s: Minimum dwell time in seconds required to emit a visit event.
        """
        self.track_id = track_id
        self.roi_id = roi_id
        self.camera_id = camera_id
        self.enter_consecutive_frames = max(1, enter_consecutive_frames)
        self.grace_period_s = max(0.1, grace_period_s)
        self.min_dwell_s = max(0.0, min_dwell_s)

        self.state: VisitState = VisitState.OUTSIDE
        self.visit_id: str = str(uuid.uuid4())
        self.merged_track_ids: List[int] = [track_id]
        self.parent_visit_id: Optional[str] = None
        self.person_id: Optional[str] = None

        # Hysteresis buffer: [(frame_idx, timestamp_s, box, foot_pt, conf)]
        self._in_roi_candidates: List[
            Tuple[int, float, Tuple[float, float, float, float], Tuple[float, float], float]
        ] = []

        # Committed visit timing & coordinates
        self.frame_enter: int = 0
        self.t_enter_s: float = 0.0
        self.frame_last_seen_inside: int = 0
        self.t_last_seen_inside: float = 0.0

        self.entry_box: Optional[Tuple[float, float, float, float]] = None
        self.entry_foot_point: Optional[Tuple[float, float]] = None
        self.last_box: Optional[Tuple[float, float, float, float]] = None
        self.last_foot_point: Optional[Tuple[float, float]] = None

        # Grace tracking
        self.grace_start_t: float = 0.0
        self.grace_start_frame: int = 0
        self.last_observation_frame: int = 0
        self.last_observation_t: float = 0.0
        self.was_seen_outside: bool = False

        # Statistical metrics
        self._conf_sum: float = 0.0
        self._conf_count: int = 0
        self.peak_conf: float = 0.0
        self.peak_box: Optional[Tuple[float, float, float, float]] = None
        self.peak_frame: int = 0
        self.peak_crop: Optional[np.ndarray] = None
        self.color_histogram: Optional[List[float]] = None

    @property
    def duration_s(self) -> float:
        """Current net dwell duration in seconds."""
        if self.state == VisitState.OUTSIDE:
            return 0.0
        return max(0.0, self.t_last_seen_inside - self.t_enter_s)

    @property
    def mean_conf(self) -> float:
        """Mean detection confidence over all in-ROI observations."""
        if self._conf_count == 0:
            return 0.0
        return self._conf_sum / float(self._conf_count)

    def update(
        self,
        is_inside: bool,
        box: Tuple[float, float, float, float],
        foot_point: Tuple[float, float],
        confidence: float,
        frame_idx: int,
        timestamp_s: float,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> Optional[Visit]:
        """Update session state with current frame observation.

        Args:
            is_inside: True if ground contact foot-point is inside the ROI polygon.
            box: Bounding box [x1, y1, x2, y2].
            foot_point: Contact coordinate (x, y).
            confidence: Detector confidence score.
            frame_idx: Video frame index.
            timestamp_s: Video timestamp in seconds.
            frame_rgb: Optional RGB frame array for appearance feature extraction.

        Returns:
            Emitted Visit if a visit closed and met the min_dwell threshold; None otherwise.
        """
        self.last_observation_frame = frame_idx
        self.last_observation_t = timestamp_s

        if self.state == VisitState.OUTSIDE:
            return self._handle_outside(
                is_inside, box, foot_point, confidence, frame_idx, timestamp_s, frame_rgb
            )
        elif self.state == VisitState.INSIDE:
            return self._handle_inside(
                is_inside, box, foot_point, confidence, frame_idx, timestamp_s, frame_rgb
            )
        elif self.state == VisitState.GRACE:
            return self._handle_grace(
                is_inside, box, foot_point, confidence, frame_idx, timestamp_s, frame_rgb
            )
        return None

    def _handle_outside(
        self,
        is_inside: bool,
        box: Tuple[float, float, float, float],
        foot_point: Tuple[float, float],
        confidence: float,
        frame_idx: int,
        timestamp_s: float,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> Optional[Visit]:
        """Process observation when entity is currently outside the ROI."""
        if is_inside:
            self._in_roi_candidates.append(
                (frame_idx, timestamp_s, box, foot_point, confidence)
            )
            # Check if hysteresis threshold is reached
            if len(self._in_roi_candidates) >= self.enter_consecutive_frames:
                # Commit transition OUTSIDE -> INSIDE
                first_obs = self._in_roi_candidates[0]
                self.frame_enter = first_obs[0]
                self.t_enter_s = first_obs[1]
                self.entry_box = first_obs[2]
                self.entry_foot_point = first_obs[3]

                self.frame_last_seen_inside = frame_idx
                self.t_last_seen_inside = timestamp_s
                self.last_box = box
                self.last_foot_point = foot_point

                # Ingest all buffered candidates into confidence stats
                for cand in self._in_roi_candidates:
                    c_conf = cand[4]
                    self._conf_sum += c_conf
                    self._conf_count += 1
                    if c_conf > self.peak_conf:
                        self.peak_conf = c_conf
                        self.peak_box = cand[2]
                        self.peak_frame = cand[0]

                self._in_roi_candidates.clear()
                self.state = VisitState.INSIDE
                self._extract_appearance(box, frame_rgb)

                logger.debug(
                    "Track %d entered ROI '%s' at frame %d (t=%.3fs, hysteresis=%d)",
                    self.track_id,
                    self.roi_id,
                    self.frame_enter,
                    self.t_enter_s,
                    self.enter_consecutive_frames,
                )
        else:
            # Reset hysteresis counter on any outside frame
            self._in_roi_candidates.clear()
        return None

    def _handle_inside(
        self,
        is_inside: bool,
        box: Tuple[float, float, float, float],
        foot_point: Tuple[float, float],
        confidence: float,
        frame_idx: int,
        timestamp_s: float,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> Optional[Visit]:
        """Process observation when entity is actively inside the ROI."""
        if is_inside:
            # Continuous presence
            self.frame_last_seen_inside = frame_idx
            self.t_last_seen_inside = timestamp_s
            self.last_box = box
            self.last_foot_point = foot_point

            self._conf_sum += confidence
            self._conf_count += 1
            if confidence > self.peak_conf:
                self.peak_conf = confidence
                self.peak_box = box
                self.peak_frame = frame_idx
                self._extract_appearance(box, frame_rgb)
        else:
            # Stepped outside ROI -> transition to GRACE
            self.state = VisitState.GRACE
            self.grace_start_t = timestamp_s
            self.grace_start_frame = frame_idx
            self.was_seen_outside = True
            logger.debug(
                "Track %d stepped outside ROI '%s' at frame %d; grace countdown started",
                self.track_id,
                self.roi_id,
                frame_idx,
            )
        return None

    def _handle_grace(
        self,
        is_inside: bool,
        box: Tuple[float, float, float, float],
        foot_point: Tuple[float, float],
        confidence: float,
        frame_idx: int,
        timestamp_s: float,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> Optional[Visit]:
        """Process observation during grace period."""
        if is_inside:
            # Returned to ROI within grace window -> restore INSIDE
            self.state = VisitState.INSIDE
            self.frame_last_seen_inside = frame_idx
            self.t_last_seen_inside = timestamp_s
            self.last_box = box
            self.last_foot_point = foot_point

            self._conf_sum += confidence
            self._conf_count += 1
            if confidence > self.peak_conf:
                self.peak_conf = confidence
                self.peak_box = box
                self.peak_frame = frame_idx
                self._extract_appearance(box, frame_rgb)

            logger.debug(
                "Track %d returned to ROI '%s' at frame %d within grace period",
                self.track_id,
                self.roi_id,
                frame_idx,
            )
            return None

        # Entity still outside or missing; check grace timer
        elapsed_grace = timestamp_s - self.grace_start_t
        if elapsed_grace >= self.grace_period_s:
            reason = (
                CloseReason.EXITED if self.was_seen_outside else CloseReason.TRACK_LOST
            )
            return self.close(reason)
        return None

    def check_missing_in_frame(
        self, frame_idx: int, timestamp_s: float
    ) -> Optional[Visit]:
        """Check status when entity is completely unobserved in the current frame."""
        if self.state == VisitState.INSIDE:
            # Track lost while inside ROI -> start grace period
            self.state = VisitState.GRACE
            self.grace_start_t = timestamp_s
            self.grace_start_frame = frame_idx
            self.was_seen_outside = False
            logger.debug(
                "Track %d vanished while inside ROI '%s'; grace period started at frame %d",
                self.track_id,
                self.roi_id,
                frame_idx,
            )
            return None
        elif self.state == VisitState.GRACE:
            elapsed_grace = timestamp_s - self.grace_start_t
            if elapsed_grace >= self.grace_period_s:
                reason = (
                    CloseReason.EXITED
                    if self.was_seen_outside
                    else CloseReason.TRACK_LOST
                )
                return self.close(reason)
        elif self.state == VisitState.OUTSIDE:
            self._in_roi_candidates.clear()
        return None

    def close(self, reason: Union[CloseReason, str]) -> Optional[Visit]:
        """Force or finalize closing of the active visit.

        Enforces mathematical invariants:
        - duration = t_last_seen_inside - t_enter (excludes grace time).
        - visits with duration < min_dwell_s are suppressed.

        Args:
            reason: CloseReason enumeration or string reason code.

        Returns:
            Committed Visit object if duration >= min_dwell_s; None otherwise.
        """
        if self.state == VisitState.CLOSED:
            return None

        prev_state = self.state
        self.state = VisitState.CLOSED
        reason_str = reason.value if isinstance(reason, CloseReason) else str(reason)

        if prev_state == VisitState.OUTSIDE:
            return None

        net_duration = max(0.0, self.t_last_seen_inside - self.t_enter_s)

        # Minimum dwell filtering (Failure Mode F6)
        if net_duration < self.min_dwell_s:
            logger.info(
                "Discarding transient visit for Track %d: duration %.3fs < min_dwell %.3fs",
                self.track_id,
                net_duration,
                self.min_dwell_s,
            )
            return None

        visit = Visit(
            visit_id=self.visit_id,
            camera_id=self.camera_id,
            roi_id=self.roi_id,
            track_id=self.track_id,
            merged_track_ids=list(self.merged_track_ids),
            parent_visit_id=self.parent_visit_id,
            person_id=self.person_id,
            video_t_enter_s=round(self.t_enter_s, 3),
            video_t_exit_s=round(self.t_last_seen_inside, 3),
            frame_enter=self.frame_enter,
            frame_exit=self.frame_last_seen_inside,
            duration_s=round(net_duration, 3),
            mean_conf=round(self.mean_conf, 4),
            peak_conf=round(self.peak_conf, 4),
            peak_frame=self.peak_frame,
            close_reason=reason_str,
            entry_box=self.entry_box,
            entry_foot_point=self.entry_foot_point,
            peak_box=self.peak_box,
            last_box=self.last_box,
            last_foot_point=self.last_foot_point,
            color_histogram=self.color_histogram,
        )

        logger.info(
            "Emitted Visit %s (Track %d, %s): frames %d-%d (t=%.3f-%.3fs, dur=%.3fs, reason=%s)",
            visit.visit_id[:8],
            visit.track_id,
            visit.roi_id,
            visit.frame_enter,
            visit.frame_exit,
            visit.video_t_enter_s,
            visit.video_t_exit_s,
            visit.duration_s,
            visit.close_reason,
        )
        return visit

    def _extract_appearance(
        self,
        box: Tuple[float, float, float, float],
        frame_rgb: Optional[np.ndarray],
    ) -> None:
        """Extract a 128-bin normalized HSV torso color histogram for discriminative re-ID."""
        if frame_rgb is None or frame_rgb.size == 0:
            return

        import cv2

        h, w = frame_rgb.shape[:2]
        bx1 = max(0, min(int(box[0]), w - 1))
        by1 = max(0, min(int(box[1]), h - 1))
        bx2 = max(0, min(int(box[2]), w))
        by2 = max(0, min(int(box[3]), h))

        box_w = bx2 - bx1
        box_h = by2 - by1
        if box_w < 8 or box_h < 16:
            return

        # Torso isolation: discard background floor and ceiling
        tx1 = bx1 + int(0.20 * box_w)
        tx2 = bx2 - int(0.20 * box_w)
        ty1 = by1 + int(0.15 * box_h)
        ty2 = by1 + int(0.65 * box_h)

        if tx2 <= tx1 or ty2 <= ty1:
            return

        torso_crop = frame_rgb[ty1:ty2, tx1:tx2]
        self.peak_crop = frame_rgb[by1:by2, bx1:bx2]

        # Convert to HSV color space
        hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2HSV)

        # 2D Histogram: 16 Hue bins (0-180), 8 Saturation bins (0-256) = 128 bins
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist_flat = hist.flatten().astype(np.float32)

        norm = np.linalg.norm(hist_flat)
        if norm > 1e-6:
            hist_flat /= norm
        self.color_histogram = [float(v) for v in hist_flat]


class VisitStateMachine:
    """Manages visit lifecycles across all active tracks and polygon ROIs."""

    def __init__(
        self,
        roi: PolygonROI,
        camera_id: str = "camera_0",
        enter_consecutive_frames: int = 3,
        grace_period_s: float = 2.0,
        min_dwell_s: float = 1.0,
    ) -> None:
        """Initialize the multi-track visit state machine.

        Args:
            roi: Active PolygonROI geometry instance.
            camera_id: Sensor identifier.
            enter_consecutive_frames: Consecutive frames required to enter.
            grace_period_s: Grace duration in seconds before closing lost tracks.
            min_dwell_s: Minimum duration required to emit a visit event.
        """
        self.roi = roi
        self.camera_id = camera_id
        self.enter_consecutive_frames = enter_consecutive_frames
        self.grace_period_s = grace_period_s
        self.min_dwell_s = min_dwell_s

        self.sessions: Dict[int, TrackVisitSession] = {}
        self.closed_visits: List[Visit] = []

    def update(
        self,
        tracklets: List[Tracklet],
        frame_idx: int,
        timestamp_s: float,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> List[Visit]:
        """Update state machine with tracklets from the current video frame.

        Args:
            tracklets: Active tracklets output by the multi-object tracker.
            frame_idx: Current frame index.
            timestamp_s: Current video timestamp in seconds.
            frame_rgb: Optional RGB frame array for appearance modeling.

        Returns:
            List of newly finalized Visit events in this frame.
        """
        newly_closed: List[Visit] = []
        observed_track_ids = set()

        for trk in tracklets:
            track_id = trk.track_id
            observed_track_ids.add(track_id)

            if track_id not in self.sessions:
                self.sessions[track_id] = TrackVisitSession(
                    track_id=track_id,
                    roi_id=self.roi.config.roi_id,
                    camera_id=self.camera_id,
                    enter_consecutive_frames=self.enter_consecutive_frames,
                    grace_period_s=self.grace_period_s,
                    min_dwell_s=self.min_dwell_s,
                )

            session = self.sessions[track_id]
            is_inside, foot_pt = self.roi.contains_box(trk.box)

            emitted = session.update(
                is_inside=is_inside,
                box=trk.box,
                foot_point=foot_pt,
                confidence=trk.confidence,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                frame_rgb=frame_rgb,
            )
            if emitted is not None:
                newly_closed.append(emitted)
                self.closed_visits.append(emitted)

        # Check for unobserved tracks currently in INSIDE or GRACE
        for track_id, session in list(self.sessions.items()):
            if track_id not in observed_track_ids:
                emitted = session.check_missing_in_frame(frame_idx, timestamp_s)
                if emitted is not None:
                    newly_closed.append(emitted)
                    self.closed_visits.append(emitted)

        return newly_closed

    def force_close_all(
        self,
        frame_idx: int,
        timestamp_s: float,
        reason: Union[CloseReason, str] = CloseReason.STREAM_END,
    ) -> List[Visit]:
        """Force-close all pending active visits at stream termination (Failure Mode F7).

        Args:
            frame_idx: Final video frame index.
            timestamp_s: Final video timestamp in seconds.
            reason: Reason code (default: CloseReason.STREAM_END).

        Returns:
            List of finalized Visit events emitted at stream end.
        """
        emitted_visits: List[Visit] = []
        for track_id, session in list(self.sessions.items()):
            if session.state in {VisitState.INSIDE, VisitState.GRACE}:
                emitted = session.close(reason)
                if emitted is not None:
                    emitted_visits.append(emitted)
                    self.closed_visits.append(emitted)
        logger.info(
            "Stream ended at frame %d (t=%.3fs): force-closed %d active visits",
            frame_idx,
            timestamp_s,
            len(emitted_visits),
        )
        return emitted_visits

    def get_active_sessions(self) -> List[TrackVisitSession]:
        """Get all sessions currently in INSIDE or GRACE state."""
        return [
            s
            for s in self.sessions.values()
            if s.state in {VisitState.INSIDE, VisitState.GRACE}
        ]
