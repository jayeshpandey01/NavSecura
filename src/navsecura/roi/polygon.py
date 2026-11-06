"""Polygon Region of Interest (ROI) and spatial membership engine.

Re-exports core geometric primitives and ray-casting utilities from
``src.navsecura.roi.geometry`` for protocol compliance.
"""

from __future__ import annotations

from navsecura.roi.geometry import (
    FootPointMode,
    Point2D,
    PolygonROI,
    ROIConfig,
)

__all__ = ["PolygonROI", "ROIConfig", "Point2D", "FootPointMode"]
