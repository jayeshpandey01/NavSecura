"""Pre-roll circular ring buffer and visual evidence generation.

Implements high-throughput in-memory rolling video history and persistent
evidence writers (entry snapshots with HUD annotations, tight peak confidence crops,
and localized pre/post-roll MP4 video clips) for forensic auditability.
"""

from __future__ import annotations

import collections
import logging
import math
from pathlib import Path
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from navsecura.events.models import Visit

logger = logging.getLogger(__name__)


class CircularVideoBuffer:
    """Thread-safe rolling circular in-memory buffer for video frames.

    Retains a continuous temporal window of recent video frames to support
    instantaneous extraction of pre-roll and post-roll video clips when
    business events are finalized.
    """

    def __init__(
        self,
        capacity_frames: Optional[int] = None,
        max_duration_s: float = 30.0,
        fps: float = 24.0,
    ) -> None:
        """Initialize circular frame buffer.

        Args:
            capacity_frames: Fixed maximum number of frames to store.
                If None, calculated from max_duration_s * fps.
            max_duration_s: Rolling history duration in seconds (default: 30.0s).
            fps: Native stream frame rate (default: 24.0).
        """
        self.fps = max(1.0, float(fps))
        if capacity_frames is not None:
            self.max_frames = max(1, int(capacity_frames))
        else:
            self.max_frames = max(1, math.ceil(max_duration_s * self.fps))

        self._buffer: collections.deque[Tuple[int, float, np.ndarray]] = collections.deque(
            maxlen=self.max_frames
        )
        self._lock = threading.Lock()

    def append(
        self,
        frame_idx: int,
        timestamp_s: float,
        frame: np.ndarray,
    ) -> None:
        """Append a new video frame observation to the rolling buffer.

        Args:
            frame_idx: Monotonic zero-indexed video frame counter.
            timestamp_s: Frame presentation timestamp in seconds.
            frame: Video frame array (BGR or RGB). A shallow copy is stored.
        """
        with self._lock:
            self._buffer.append((frame_idx, float(timestamp_s), frame.copy()))

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Retrieve the exact or closest frame array for a specific frame index.

        Args:
            frame_idx: Desired video frame index.

        Returns:
            Frame array if within buffered history, or closest frame if index
            is within 2 frames of availability; None if buffer is empty or outside range.
        """
        with self._lock:
            if not self._buffer:
                return None

            oldest_idx = self._buffer[0][0]
            newest_idx = self._buffer[-1][0]

            # Allow at most 2 frames interpolation tolerance
            if frame_idx < oldest_idx - 2 or frame_idx > newest_idx + 2:
                return None

            best_frame: Optional[np.ndarray] = None
            min_dist = float("inf")

            for f_idx, _, fr in self._buffer:
                dist = abs(f_idx - frame_idx)
                if dist < min_dist and dist <= 2:
                    min_dist = dist
                    best_frame = fr
                if dist == 0:
                    break

            return best_frame.copy() if best_frame is not None else None

    def get_frames_slice(
        self,
        start_frame: int,
        end_frame: int,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """Extract a contiguous sequence of frames spanning [start_frame, end_frame].

        Automatically handles edge boundaries by clamping to currently buffered
        frames without throwing index errors.

        Args:
            start_frame: Beginning frame index (inclusive).
            end_frame: Terminating frame index (inclusive).

        Returns:
            List of (frame_idx, timestamp_s, frame_copy) tuples.
        """
        with self._lock:
            if not self._buffer or start_frame > end_frame:
                return []

            matched: List[Tuple[int, float, np.ndarray]] = []
            for f_idx, ts, fr in self._buffer:
                if start_frame <= f_idx <= end_frame:
                    matched.append((f_idx, ts, fr.copy()))

            return matched

    def get_range(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """Slice retrieval alias to safely retrieve frames spanning a frame range.

        Args:
            start_frame_idx: Beginning frame index (inclusive).
            end_frame_idx: Terminating frame index (inclusive).

        Returns:
            List of (frame_idx, timestamp_s, frame_copy) tuples.
        """
        return self.get_frames_slice(start_frame_idx, end_frame_idx)

    def get_time_slice(
        self,
        start_time_s: float,
        end_time_s: float,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """Extract frames within a specified time range [start_time_s, end_time_s].

        Args:
            start_time_s: Beginning timestamp in seconds (inclusive).
            end_time_s: Ending timestamp in seconds (inclusive).

        Returns:
            List of (frame_idx, timestamp_s, frame_copy) tuples.
        """
        with self._lock:
            if not self._buffer or start_time_s > end_time_s:
                return []

            matched: List[Tuple[int, float, np.ndarray]] = []
            for f_idx, ts, fr in self._buffer:
                if start_time_s <= ts <= end_time_s:
                    matched.append((f_idx, ts, fr.copy()))

            return matched

    def clear(self) -> None:
        """Purge all buffered frames."""
        with self._lock:
            self._buffer.clear()

    @property
    def count(self) -> int:
        """Return the current number of frames in the circular buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def oldest_frame_idx(self) -> Optional[int]:
        """Return frame index of oldest buffered observation, or None if empty."""
        with self._lock:
            return self._buffer[0][0] if self._buffer else None

    @property
    def newest_frame_idx(self) -> Optional[int]:
        """Return frame index of newest buffered observation, or None if empty."""
        with self._lock:
            return self._buffer[-1][0] if self._buffer else None

    @property
    def oldest_timestamp_s(self) -> Optional[float]:
        """Return timestamp of oldest buffered observation, or None if empty."""
        with self._lock:
            return self._buffer[0][1] if self._buffer else None

    @property
    def newest_timestamp_s(self) -> Optional[float]:
        """Return timestamp of newest buffered observation, or None if empty."""
        with self._lock:
            return self._buffer[-1][1] if self._buffer else None


class EvidenceWriter:
    """Generates and archives visual evidence artifacts for emitted visits.

    Produces:
    1. Entry snapshot: Full-frame image with highlighted bounding box, track ID,
       foot-point marker, and timestamp badge.
    2. Peak snapshot: Tight high-resolution crop of the target entity at maximum
       detection confidence.
    3. Event clip: Short, self-contained MP4 video with embedded telemetry HUD.
    """

    def __init__(self, output_dir: Union[str, Path] = "outputs") -> None:
        """Initialize EvidenceWriter and ensure target directories exist.

        Args:
            output_dir: Base directory for generated artifacts.
        """
        self.output_dir = Path(output_dir)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.clips_dir = self.output_dir / "clips"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save_entry_snapshot(
        self,
        frame: np.ndarray,
        visit: Visit,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate and save an entry evidence snapshot with forensic annotations.

        Args:
            frame: Full-resolution video frame at visit entry.
            visit: Emitted Visit event data model.
            output_dir: Optional override directory for snapshots.

        Returns:
            Relative filepath string to the saved snapshot.
        """
        dest_dir = Path(output_dir) if output_dir is not None else self.snapshots_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw entry bounding box if available
        if visit.entry_box is not None:
            x1, y1, x2, y2 = [int(round(coord)) for coord in visit.entry_box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            # Emerald green bounding box
            box_color = (40, 200, 80)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Track ID badge
            id_badge = f"Track #{visit.track_id}"
            (tw, th), _ = cv2.getTextSize(id_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(0, y1)), box_color, -1)
            cv2.putText(
                annotated,
                id_badge,
                (x1 + 3, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw ground-contact foot point
        if visit.entry_foot_point is not None:
            fx, fy = int(round(visit.entry_foot_point[0])), int(round(visit.entry_foot_point[1]))
            # Bright yellow target circle
            cv2.circle(annotated, (fx, fy), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, (fx, fy), 8, (0, 165, 255), 1, cv2.LINE_AA)

        # Semi-transparent HUD header
        hud_h = 32
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.line(annotated, (0, hud_h), (w, hud_h), (80, 80, 80), 1)

        hud_text = (
            f"ENTRY SNAPSHOT | Visit: {visit.visit_id[:8]} | Track: {visit.track_id} | "
            f"Enter: {visit.video_t_enter_s:.2f}s | ROI: {visit.roi_id}"
        )
        cv2.putText(
            annotated,
            hud_text,
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        filename = f"entry_{visit.visit_id[:8]}_t{visit.track_id}.jpg"
        save_path = dest_dir / filename
        cv2.imwrite(str(save_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        alias_path = dest_dir / f"visit_{visit.visit_id[:8]}_entry.jpg"
        try:
            cv2.imwrite(str(alias_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception:
            pass

        # Return standardized relative path
        rel_path = f"outputs/snapshots/{filename}"
        return rel_path

    def save_peak_snapshot(
        self,
        frame: np.ndarray,
        visit: Visit,
        output_dir: Optional[Union[str, Path]] = None,
        margin_pct: float = 0.15,
    ) -> str:
        """Generate a tight, high-confidence crop of the target subject.

        Args:
            frame: Full video frame containing the peak confidence observation.
            visit: Emitted Visit event data model.
            output_dir: Optional override directory for snapshots.
            margin_pct: Bounding box expansion margin percentage (default: 15%).

        Returns:
            Relative filepath string to the saved peak crop snapshot.
        """
        dest_dir = Path(output_dir) if output_dir is not None else self.snapshots_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        box = visit.peak_box or visit.entry_box or visit.last_box

        if box is not None:
            bx1, by1, bx2, by2 = box
            bw = bx2 - bx1
            bh = by2 - by1

            pad_w = bw * margin_pct
            pad_h = bh * margin_pct

            x1 = max(0, int(round(bx1 - pad_w)))
            y1 = max(0, int(round(by1 - pad_h)))
            x2 = min(w, int(round(bx2 + pad_w)))
            y2 = min(h, int(round(by2 + pad_h)))

            crop = frame[y1:y2, x1:x2].copy()
        else:
            # Fallback to center region if no box recorded
            cx, cy = w // 2, h // 2
            crop = frame[max(0, cy - 100) : min(h, cy + 100), max(0, cx - 100) : min(w, cx + 100)].copy()

        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            crop = frame.copy()

        # Add modest confidence annotation header
        crop_h, crop_w = crop.shape[:2]
        if crop_h > 40 and crop_w > 60:
            conf_str = f"Conf: {visit.peak_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(crop, (0, 0), (tw + 6, th + 6), (20, 20, 20), -1)
            cv2.putText(
                crop,
                conf_str,
                (3, th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        filename = f"peak_{visit.visit_id[:8]}_t{visit.track_id}.jpg"
        save_path = dest_dir / filename
        cv2.imwrite(str(save_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        alias_path = dest_dir / f"visit_{visit.visit_id[:8]}_peak.jpg"
        try:
            cv2.imwrite(str(alias_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception:
            pass

        rel_path = f"outputs/snapshots/{filename}"
        return rel_path

    def save_event_clip(
        self,
        frames: List[Tuple[int, float, np.ndarray]],
        visit: Visit,
        fps: float = 24.0,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """Encode a short standalone MP4 clip of the visit event with telemetry overlay.

        Args:
            frames: Chronologically ordered sequence of (frame_idx, timestamp_s, frame) tuples.
            visit: Emitted Visit event data model.
            fps: Playback frame rate for the encoded MP4 container.
            output_dir: Optional override directory for video clips.

        Returns:
            Relative filepath string to the saved video clip.
        """
        dest_dir = Path(output_dir) if output_dir is not None else self.clips_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = f"clip_{visit.visit_id[:8]}_t{visit.track_id}.mp4"
        save_path = dest_dir / filename

        if not frames:
            logger.warning("No frames provided for visit %s clip generation", visit.visit_id[:8])
            return ""

        sample_frame = frames[0][2]
        h, w = sample_frame.shape[:2]
        effective_fps = max(1.0, float(fps))

        # Use mp4v fourcc for maximum cross-platform compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, effective_fps, (w, h))

        if not writer.isOpened():
            # Fallback to avc1 if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(save_path), fourcc, effective_fps, (w, h))

        hud_h = 30
        for f_idx, ts, fr in frames:
            canvas = fr.copy()
            overlay = canvas.copy()

            # Bottom telemetry HUD bar
            cv2.rectangle(overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
            cv2.line(canvas, (0, h - hud_h), (w, h - hud_h), (60, 60, 60), 1)

            clip_text = (
                f"VISIT: {visit.visit_id[:8]} | TRACK: {visit.track_id} | "
                f"TIME: {ts:5.2f}s | DWELL: {visit.duration_s:4.2f}s | ZONE: {visit.roi_id}"
            )
            cv2.putText(
                canvas,
                clip_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)

        writer.release()
        logger.debug("Encoded %d-frame clip for visit %s to %s", len(frames), visit.visit_id[:8], save_path)

        alias_path = dest_dir / f"visit_{visit.visit_id[:8]}.mp4"
        try:
            shutil.copyfile(str(save_path), str(alias_path))
        except Exception:
            pass

        rel_path = f"outputs/clips/{filename}"
        return rel_path
