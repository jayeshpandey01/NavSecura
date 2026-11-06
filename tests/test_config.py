"""Unit tests for configuration engine and dynamic ROI registry."""

from __future__ import annotations

from pathlib import Path
from navsecura.io.config import AppConfig
from navsecura.roi.registry import ROIRegistry


def test_app_config_load_defaults() -> None:
    """Test loading default configuration from configs/default.yaml."""
    cfg = AppConfig.load("configs/default.yaml")
    assert cfg.pipeline.name == "navsecura-video-analytics"
    assert cfg.detector.model_type == "yolo"
    assert cfg.detector.classes == [0]
    assert cfg.tracker.type == "bytetrack"
    assert cfg.events.grace_period_s == 2.0
    assert cfg.events.min_dwell_s == 1.0


def test_app_config_overrides() -> None:
    """Test dot-notated CLI and dictionary overrides."""
    overrides = {
        "events.grace_period_s": 3.5,
        "detector.confidence_threshold": 0.45,
        "video.camera_id": "cam_custom_99",
    }
    cfg = AppConfig.load("configs/default.yaml", overrides=overrides)
    assert cfg.events.grace_period_s == 3.5
    assert cfg.detector.confidence_threshold == 0.45
    assert cfg.video.camera_id == "cam_custom_99"


def test_stream_dynamic_adaptation() -> None:
    """Test adaptive buffer and stitching parameters based on stream properties."""
    cfg = AppConfig.load("configs/default.yaml")

    # Native 12 FPS stream (sample.mp4)
    cfg.adapt_to_stream(native_fps=12.0, width=768, height=432)
    assert cfg.tracker.track_buffer == int(round(12.0 * 2.5))  # 30 frames
    assert cfg.events.stitching.max_distance_px > 0

    # High FPS 60 FPS stream (shop_aisle.mp4)
    cfg.adapt_to_stream(native_fps=59.94, width=720, height=404)
    assert cfg.tracker.track_buffer == int(round(59.94 * 2.5))  # 150 frames


def test_roi_registry_lookup_and_resolution() -> None:
    """Test dynamic resolution of ROIs across registered scenarios."""
    registry = ROIRegistry("configs/roi_registry.json")

    # Lookup by direct roi_id
    cfg_entrance = registry.get_config("roi_entrance")
    assert cfg_entrance is not None
    assert cfg_entrance.camera_id == "cam_entrance_01"

    # Resolve PolygonROI for specific resolution
    roi = registry.resolve_roi(
        roi_target="roi_entrance",
        frame_width=768,
        frame_height=432,
    )
    assert len(roi.pixel_vertices) == 4
    assert roi.frame_width == 768
    assert roi.frame_height == 432

