"""Multi-object tracker implementation wrapping ByteTrack and BoT-SORT.

Conforms to the abstract Tracker protocol. Decoupled from detection sources:
accepts any List[Detection] and maintains ground plane trajectory memory.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from navsecura.detect.base import Detection
from navsecura.track.base import Tracker, Tracklet

logger = logging.getLogger("navsecura.track.ultralytics")


class _DetectionContainer:
    """NumPy-backed container adapter required by Ultralytics ByteTrack.

    Implements boolean indexing and exposes array views for bounding boxes,
    confidences, and class identifiers.
    """

    def __init__(
        self,
        xywh: np.ndarray,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xywh = xywh
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: Any) -> "_DetectionContainer":
        return _DetectionContainer(
            xywh=self.xywh[mask],
            xyxy=self.xyxy[mask],
            conf=self.conf[mask],
            cls=self.cls[mask],
        )

    @classmethod
    def from_detections(cls, detections: List[Detection]) -> "_DetectionContainer":
        """Construct container from a list of Detection dataclasses."""
        if not detections:
            return cls(
                xywh=np.zeros((0, 4), dtype=np.float32),
                xyxy=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros(0, dtype=np.float32),
                cls=np.zeros(0, dtype=np.float32),
            )

        xywh_list: List[List[float]] = []
        xyxy_list: List[List[float]] = []
        conf_list: List[float] = []
        cls_list: List[float] = []

        for d in detections:
            x1, y1, x2, y2 = d.box
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            xywh_list.append([cx, cy, w, h])
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(float(d.confidence))
            cls_list.append(float(d.class_id))

        return cls(
            xywh=np.array(xywh_list, dtype=np.float32),
            xyxy=np.array(xyxy_list, dtype=np.float32),
            conf=np.array(conf_list, dtype=np.float32),
            cls=np.array(cls_list, dtype=np.float32),
        )


class UltralyticsTracker(Tracker):
    """Multi-object tracker adapter wrapping ByteTrack with trajectory memory."""

    def __init__(
        self,
        tracker_type: Literal["bytetrack", "botsort"] = "bytetrack",
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.60,
        match_thresh: float = 0.80,
        track_buffer: int = 30,
        fuse_score: bool = True,
        max_trajectory_length: int = 60,
        gmc_method: str = "sparseOptFlow",
        proximity_thresh: float = 0.50,
        appearance_thresh: float = 0.25,
        with_reid: bool = False,
    ) -> None:
        """Initialize the tracker.

        Args:
            tracker_type: Tracker algorithm selection ('bytetrack' or 'botsort').
            track_high_thresh: High detection threshold for primary association.
            track_low_thresh: Low detection threshold for secondary recovery.
            new_track_thresh: Confidence threshold required to initialize a new tracklet.
            match_thresh: Hungarian IoU matching threshold.
            track_buffer: Number of lost frames before terminating a tracklet.
            fuse_score: Whether to fuse detection confidence with tracking similarity.
            max_trajectory_length: Maximum history size of foot-point coordinates.
            gmc_method: Global motion compensation method for BoT-SORT ('sparseOptFlow', 'none', 'orb').
            proximity_thresh: Spatial proximity distance threshold for BoT-SORT.
            appearance_thresh: Appearance ReID embedding threshold for BoT-SORT.
            with_reid: Whether to enable appearance ReID feature matching in BoT-SORT.
        """
        self.tracker_type = tracker_type.lower()
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.fuse_score = fuse_score
        self.max_trajectory_length = max_trajectory_length
        self.gmc_method = gmc_method
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid

        self._args = SimpleNamespace(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            match_thresh=self.match_thresh,
            track_buffer=self.track_buffer,
            fuse_score=self.fuse_score,
            gmc_method=self.gmc_method,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            with_reid=self.with_reid,
        )

        if self.tracker_type == "botsort":
            self._tracker = BOTSORT(self._args)
        else:
            self._tracker = BYTETracker(self._args)

        self._trajectories: Dict[int, List[Tuple[float, float]]] = {}
        self._track_hits: Dict[int, int] = {}
        self._unique_track_ids: Set[int] = set()

        self._last_update_ms: float = 0.0
        self._total_updates: int = 0
        self._cumulative_latency_ms: float = 0.0
        self._current_active_tracks: int = 0

        logger.info(
            "Initialized %s tracker: buffer=%d frames, high_th=%.2f, low_th=%.2f, match_th=%.2f",
            self.name,
            self.track_buffer,
            self.track_high_thresh,
            self.track_low_thresh,
            self.match_thresh,
        )

    @property
    def name(self) -> str:
        """Tracker algorithm identifier."""
        if self.tracker_type == "botsort":
            reid_tag = "+ReID" if self.with_reid else ""
            return f"BoT-SORT{reid_tag}(gmc={self.gmc_method}, buf={self.track_buffer})"
        return f"ByteTrack(buf={self.track_buffer})"

    @property
    def last_update_ms(self) -> float:
        """Latency of the most recent association update in milliseconds."""
        return self._last_update_ms

    @property
    def average_latency_ms(self) -> float:
        """Mean tracker update latency in milliseconds."""
        if self._total_updates == 0:
            return 0.0
        return self._cumulative_latency_ms / self._total_updates

    @property
    def active_tracks_count(self) -> int:
        """Current count of active tracked objects in the latest frame."""
        return self._current_active_tracks

    @property
    def total_unique_tracks(self) -> int:
        """Total unique track IDs instantiated across the session."""
        return len(self._unique_track_ids)

    @property
    def unique_track_ids(self) -> Set[int]:
        """Set of all unique track IDs generated."""
        return set(self._unique_track_ids)

    def reset(self) -> None:
        """Reset internal tracker state and trajectory history."""
        if self.tracker_type == "botsort":
            self._tracker = BOTSORT(self._args)
        else:
            self._tracker = BYTETracker(self._args)
        self._trajectories.clear()
        self._track_hits.clear()
        self._unique_track_ids.clear()
        self._last_update_ms = 0.0
        self._total_updates = 0
        self._cumulative_latency_ms = 0.0
        self._current_active_tracks = 0
        logger.info("Tracker state reset.")

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
    ) -> List[Tracklet]:
        """Update tracker state with new frame detections.

        Args:
            detections: Frame detections emitted by a Detector.
            frame: Image array for frame dimensions and optional appearance cues.
            frame_idx: Zero-indexed integer frame index.
            timestamp_s: Elapsed video timestamp in seconds.

        Returns:
            List of confirmed active Tracklet objects for the current frame.
        """
        frame_h, frame_w = frame.shape[:2] if frame is not None else (1080, 1920)

        start_time = time.perf_counter()
        container = _DetectionContainer.from_detections(detections)

        # Execute ByteTrack bipartite association
        output = self._tracker.update(container, img=frame)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._last_update_ms = elapsed_ms
        self._total_updates += 1
        self._cumulative_latency_ms += elapsed_ms

        tracklets: List[Tracklet] = []

        if output is None or len(output) == 0:
            self._current_active_tracks = 0
            return tracklets

        # output shape: (N, 8) with [x1, y1, x2, y2, track_id, score, cls, idx]
        for row in output:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            score = float(row[5])
            cls_id = int(row[6]) if len(row) > 6 else 0

            # Safe coordinate clipping to image bounds
            x1 = max(0.0, min(float(frame_w - 1), x1))
            y1 = max(0.0, min(float(frame_h - 1), y1))
            x2 = max(0.0, min(float(frame_w - 1), x2))
            y2 = max(0.0, min(float(frame_h - 1), y2))

            foot_x = (x1 + x2) / 2.0
            foot_y = y2

            # Trajectory history update
            if track_id not in self._trajectories:
                self._trajectories[track_id] = []
                self._track_hits[track_id] = 0
            self._trajectories[track_id].append((round(foot_x, 1), round(foot_y, 1)))
            if len(self._trajectories[track_id]) > self.max_trajectory_length:
                self._trajectories[track_id].pop(0)

            self._track_hits[track_id] += 1
            self._unique_track_ids.add(track_id)

            tracklets.append(
                Tracklet(
                    track_id=track_id,
                    box=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                    confidence=round(score, 4),
                    frame_idx=frame_idx,
                    timestamp_s=round(timestamp_s, 4),
                    trajectory=list(self._trajectories[track_id]),
                    state="active",
                    class_id=cls_id,
                    class_name="person",
                    hits=self._track_hits[track_id],
                )
            )

        self._current_active_tracks = len(tracklets)
        return tracklets

