"""Unit tests for PolygonROI and spatial containment logic.

Verifies foot-point projection, ray-casting containment, resolution
scaling, and distance calculations under nominal and edge cases.
"""

from __future__ import annotations

import pytest
from navsecura.roi.geometry import PolygonROI, ROIConfig


@pytest.fixture
def sample_roi_config() -> ROIConfig:
    """Fixture providing a normalized 4-vertex quadrilateral ROI."""
    return ROIConfig(
        roi_id="roi_test_zone",
        name="Test Entrance Corridor",
        coordinates_norm=[
            (0.1, 0.5),
            (0.9, 0.5),
            (0.95, 0.95),
            (0.05, 0.95),
        ],
        foot_point_mode="bottom_center",
    )


def test_polygon_roi_initialization_normalized(sample_roi_config: ROIConfig) -> None:
    """Test resolution adaptation from normalized coordinates."""
    roi = PolygonROI(sample_roi_config, frame_width=1000, frame_height=1000)
    assert len(roi.pixel_vertices) == 4
    assert roi.pixel_vertices[0] == (100.0, 500.0)
    assert roi.pixel_vertices[1] == (900.0, 500.0)
    assert roi.pixel_vertices[2] == (950.0, 950.0)
    assert roi.pixel_vertices[3] == (50.0, 950.0)


def test_polygon_roi_scaling_different_resolutions(sample_roi_config: ROIConfig) -> None:
    """Test scaling across multiple aspect ratios and resolutions."""
    roi_720p = PolygonROI(sample_roi_config, frame_width=1280, frame_height=720)
    assert roi_720p.pixel_vertices[0] == (128.0, 360.0)

    roi_1080p = PolygonROI(sample_roi_config, frame_width=1920, frame_height=1080)
    assert roi_1080p.pixel_vertices[0] == (192.0, 540.0)


def test_foot_point_modes(sample_roi_config: ROIConfig) -> None:
    """Test calculation of ground plane contact points across modes."""
    roi = PolygonROI(sample_roi_config, frame_width=1000, frame_height=1000)
    box = [100.0, 200.0, 300.0, 600.0]  # [x1, y1, x2, y2]

    # Mode: bottom_center
    roi.foot_point_mode = "bottom_center"
    assert roi.compute_foot_point(box) == (200.0, 600.0)

    # Mode: center
    roi.foot_point_mode = "center"
    assert roi.compute_foot_point(box) == (200.0, 400.0)

    # Mode: bottom_left
    roi.foot_point_mode = "bottom_left"
    assert roi.compute_foot_point(box) == (100.0, 600.0)

    # Mode: bottom_right
    roi.foot_point_mode = "bottom_right"
    assert roi.compute_foot_point(box) == (300.0, 600.0)


def test_box_containment_ground_plane_vs_torso(sample_roi_config: ROIConfig) -> None:
    """Demonstrate why foot-point is superior to bounding box centroid."""
    roi = PolygonROI(sample_roi_config, frame_width=1000, frame_height=1000)

    # Person walking outside/above the zone:
    # Torso (centroid) is at (500, 400) [outside], feet at (500, 600) [inside]
    box_entering = [450.0, 200.0, 550.0, 600.0]
    is_inside, foot_pt = roi.contains_box(box_entering)

    assert foot_pt == (500.0, 600.0)
    assert is_inside is True
    # If we tested centroid (500, 400), it would falsely claim outside
    assert roi.contains_point(500.0, 400.0) is False


def test_perimeter_distance_signed(sample_roi_config: ROIConfig) -> None:
    """Verify signed perimeter distance: negative inside, positive outside."""
    roi = PolygonROI(sample_roi_config, frame_width=1000, frame_height=1000)

    # Point deeply inside polygon center (500, 700)
    dist_in = roi.distance_to_perimeter_px(500.0, 700.0)
    assert dist_in < 0.0

    # Point clearly outside (500, 100)
    dist_out = roi.distance_to_perimeter_px(500.0, 100.0)
    assert dist_out > 0.0

