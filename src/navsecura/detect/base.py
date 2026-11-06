"""Detection abstract protocol and data structures.

Enforces strict typed decoupling between object perception models
and downstream tracking/event aggregation modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Bounding box detection with class semantics and ground-plane projection.

    Coordinate Convention:
        box: (x1, y1, x2, y2) in absolute pixel coordinates.
        (0, 0) is the top-left corner of the image canvas.
    """

    box: Tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "person"

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
        """Bottom coordinate (ground contact plane)."""
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
class Detector(Protocol):
    """Abstract protocol for 2D visual object detectors."""

    @property
    def name(self) -> str:
        """Human-readable detector architecture identifier."""
        ...

    @property
    def last_inference_ms(self) -> float:
        """Execution duration of the most recent inference pass in milliseconds."""
        ...

    @property
    def device(self) -> str:
        """Target execution accelerator (cpu, mps, cuda)."""
        ...

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on a single image frame.

        Args:
            frame: Input image array as numpy ndarray (H, W, C) in BGR or RGB format.

        Returns:
            List of Detection instances for confirmed targets.
        """
        ...

