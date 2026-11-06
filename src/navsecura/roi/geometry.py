"""Dynamic Region of Interest (ROI) geometry and spatial containment engine.

Fully resolution-independent and dynamic: supports normalized coordinates,
automatic resolution scaling, foot-point ground plane projection, and
high-performance polygon ray-casting.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

logger = logging.getLogger("navsecura.roi")

FootPointMode = Literal["bottom_center", "center", "bottom_left", "bottom_right"]


class Point2D(BaseModel):
    """2D spatial point coordinate."""

    x: float
    y: float


class ROIConfig(BaseModel):
    """Dynamic Region of Interest configuration specification.

    Supports both resolution-independent normalized coordinates [0.0, 1.0]
    and legacy/reference pixel coordinates with automatic rescaling.
    """

    roi_id: str = Field(..., description="Unique identifier for the ROI polygon")
    name: str = Field(..., description="Human-readable description of the zone")
    description: Optional[str] = Field(None, description="Detailed operational context")
    camera_id: Optional[str] = Field(None, description="Associated camera or sensor stream")
    coordinates_norm: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Normalized polygon vertices [x_norm, y_norm] where 0.0 <= x,y <= 1.0",
    )
    coordinates: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Pixel coordinates [[x, y], ...]",
    )
    resolution: Optional[Dict[str, int]] = Field(
        None,
        description="Reference resolution {width: int, height: int} for pixel coordinates",
    )
    foot_point_mode: FootPointMode = Field(
        default="bottom_center",
        description="Ground-plane projection mode: bottom_center, center, bottom_left, or bottom_right",
    )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ROIConfig":
        """Load ROI configuration dynamically from JSON or YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"ROI configuration file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix.lower() in {".yaml", ".yml"}:
                import yaml

                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)


class PolygonROI:
    """Runtime dynamic polygon ROI with resolution-adaptive geometry."""

    def __init__(self, config: ROIConfig, frame_width: int, frame_height: int) -> None:
        """Initialize the polygon ROI dynamically adapted to stream dimensions.

        Args:
            config: ROI configuration specification.
            frame_width: Active video frame width in pixels.
            frame_height: Active video frame height in pixels.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Invalid frame dimensions: {frame_width}x{frame_height}")

        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.diagonal = math.sqrt(frame_width**2 + frame_height**2)
        self.foot_point_mode: FootPointMode = config.foot_point_mode

        # Dynamically compute pixel polygon vertices
        self.pixel_vertices = self._compute_pixel_vertices()
        self.polygon = Polygon(self.pixel_vertices)
        if not self.polygon.is_valid:
            self.polygon = self.polygon.buffer(0)
        self._prepared_poly = prep(self.polygon)

        logger.debug(
            "Initialized dynamic PolygonROI '%s' for resolution %dx%d: %s",
            self.config.roi_id,
            frame_width,
            frame_height,
            self.pixel_vertices,
        )

    def _compute_pixel_vertices(self) -> List[Tuple[float, float]]:
        """Compute pixel coordinates from normalized or reference coordinates."""
        # Case 1: Normalized coordinates provided (ideal: fully resolution-independent)
        if self.config.coordinates_norm is not None and len(self.config.coordinates_norm) >= 3:
            return [
                (
                    round(x_norm * self.frame_width, 1),
                    round(y_norm * self.frame_height, 1),
                )
                for x_norm, y_norm in self.config.coordinates_norm
            ]

        # Case 2: Pixel coordinates with reference resolution
        if self.config.coordinates is not None and len(self.config.coordinates) >= 3:
            coords = self.config.coordinates
            ref_w = (
                self.config.resolution.get("width", self.frame_width)
                if self.config.resolution
                else self.frame_width
            )
            ref_h = (
                self.config.resolution.get("height", self.frame_height)
                if self.config.resolution
                else self.frame_height
            )

            scale_x = self.frame_width / float(ref_w)
            scale_y = self.frame_height / float(ref_h)

            return [
                (round(x * scale_x, 1), round(y * scale_y, 1)) for x, y in coords
            ]

        raise ValueError(
            f"ROI '{self.config.roi_id}' must provide either 'coordinates_norm' "
            f"or 'coordinates' with at least 3 polygon vertices."
        )

    def compute_foot_point(
        self,
        box: Union[List[float], Tuple[float, float, float, float]],
    ) -> Tuple[float, float]:
        """Compute ground plane contact point from bounding box [x1, y1, x2, y2].

        Args:
            box: Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Tuple of (x, y) contact coordinates on the ground plane.
        """
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        if self.foot_point_mode == "bottom_center":
            return ((x1 + x2) / 2.0, float(y2))
        elif self.foot_point_mode == "center":
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        elif self.foot_point_mode == "bottom_left":
            return (float(x1), float(y2))
        elif self.foot_point_mode == "bottom_right":
            return (float(x2), float(y2))
        return ((x1 + x2) / 2.0, float(y2))

    def contains_point(self, x: float, y: float) -> bool:
        """Test if 2D coordinate point lies within the polygon.

        Args:
            x: X-coordinate in pixels.
            y: Y-coordinate in pixels.

        Returns:
            True if point is inside or on polygon boundary.
        """
        pt = Point(x, y)
        return bool(self._prepared_poly.contains(pt) or self.polygon.touches(pt))

    def contains_box(
        self,
        box: Union[List[float], Tuple[float, float, float, float]],
    ) -> Tuple[bool, Tuple[float, float]]:
        """Test if tracked target is inside the ROI via its ground contact point.

        Args:
            box: Target bounding box [x1, y1, x2, y2].

        Returns:
            Tuple of (is_inside: bool, foot_point: (x, y)).
        """
        foot_pt = self.compute_foot_point(box)
        is_inside = self.contains_point(foot_pt[0], foot_pt[1])
        return is_inside, foot_pt

    def distance_to_perimeter_px(self, x: float, y: float) -> float:
        """Compute the signed Euclidean distance to the polygon boundary in pixels.

        Negative values indicate interior depth; positive values indicate exterior distance.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            Signed Euclidean distance in pixels.
        """
        pt = Point(x, y)
        boundary_dist = self.polygon.exterior.distance(pt)
        if self._prepared_poly.contains(pt):
            return -boundary_dist
        return boundary_dist

    def distance_to_perimeter_normalized(self, x: float, y: float) -> float:
        """Compute normalized distance invariant to frame resolution.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            Signed distance normalized by frame diagonal (diagonal = 1.0).
        """
        return self.distance_to_perimeter_px(x, y) / self.diagonal

