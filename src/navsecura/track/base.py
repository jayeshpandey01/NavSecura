"""Tracking abstract protocol and tracklet data models.

Maintains multi-frame spatial continuity and trajectory state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np

from navsecura.detect.base import Detection


@dataclass
class Tracklet:
    """Active multi-frame tracked object instance.

    Tracks identity across frames, maintains ground-contact trajectory history,
    and exposes spatial projection helpers.
    """

    track_id: int
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    frame_idx: int
    timestamp_s: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)  # Historical foot points
    state: str = "active"  # "active", "lost", "closed"
    class_id: int = 0
    class_name: str = "person"
    hits: int = 1
    time_since_update: int = 0

    @property
    def x1(self) -> float:
        """Left coordinate."""
        return self.box[0]

    @property
    def y1(self) -> float:
        """Top coordinate."""
        return self.box[1]

    @property
    def x2(self) -> float:
        """Right coordinate."""
        return self.box[2]

    @property
    def y2(self) -> float:
        """Bottom coordinate (ground plane contact)."""
        return self.box[3]

    @property
    def width(self) -> float:
        """Bounding box pixel width."""
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        """Bounding box pixel height."""
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        """Bounding box pixel area."""
        return self.width * self.height

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Ground plane contact point: midpoint of the lower horizontal edge.

        P_foot = ((x1 + x2) / 2.0, y2)
        """
        return ((self.box[0] + self.box[2]) / 2.0, self.box[3])

    @property
    def center(self) -> Tuple[float, float]:
        """Geometric centroid of the bounding box."""
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        """Return box as (x1, y1, x2, y2)."""
        return self.box

    def as_xywh(self) -> Tuple[float, float, float, float]:
        """Return box as (center_x, center_y, width, height)."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)


@runtime_checkable
class Tracker(Protocol):
    """Abstract protocol for multi-object trackers."""

    @property
    def name(self) -> str:
        """Tracker algorithm identifier."""
        ...

    @property
    def last_update_ms(self) -> float:
        """Association and Kalman update latency in milliseconds."""
        ...

    @property
    def active_tracks_count(self) -> int:
        """Current count of active tracked objects."""
        ...

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
            frame: Raw image array for appearance or motion compensation.
            frame_idx: Zero-indexed integer frame number.
            timestamp_s: Stream elapsed time in seconds.

        Returns:
            List of confirmed active Tracklet objects for the current frame.
        """
        ...

    def reset(self) -> None:
        """Reset internal tracker state and trajectory memory."""
        ...

