"""Unit tests for CCTVVideoProcessor and diagnostic video rendering."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from navsecura.io.config import AppConfig
from navsecura.io.processor import CCTVVideoProcessor, get_color_for_id
from navsecura.roi.geometry import PolygonROI, ROIConfig
from navsecura.track.base import Tracklet


def test_get_color_for_id_determinism() -> None:
    """Verify color generation is deterministic and wraps across palette."""
    c1_a = get_color_for_id(1)
    c1_b = get_color_for_id(1)
    assert c1_a == c1_b
    assert len(c1_a) == 3

    c11 = get_color_for_id(11)
    assert len(c11) == 3


def test_cctv_processor_frame_annotation() -> None:
    """Verify annotate_frame properly renders bounding box, foot-point, and HUD."""
    config = AppConfig.load() if Path("configs/default.yaml").exists() else AppConfig()
    processor = CCTVVideoProcessor(config=config)

    canvas = np.zeros((432, 768, 3), dtype=np.uint8)
    tracklet = Tracklet(
        track_id=1,
        box=(100.0, 150.0, 160.0, 320.0),
        confidence=0.88,
        class_id=0,
        frame_idx=10,
        timestamp_s=0.833,
        trajectory=[(130, 310), (130, 315), (130, 320)],
    )

    annotated = processor.annotate_frame(
        frame_bgr=canvas,
        tracklets=[tracklet],
        frame_idx=10,
        total_frames=100,
        timestamp_s=0.833,
        det_ms=25.0,
        trk_ms=0.5,
        fps=24.0,
    )

    assert annotated.shape == (432, 768, 3)
    # Ensure drawing occurred (canvas is not all zeros)
    assert np.count_nonzero(annotated) > 0


def test_cctv_processor_dry_run(tmp_path: Path) -> None:
    """Verify processor runs cleanly on synthetic video frames."""
    import cv2

    test_video = tmp_path / "synthetic_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(test_video), fourcc, 10.0, (320, 240))
    for i in range(15):
        frame = np.full((240, 320, 3), fill_value=i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()

    test_out = tmp_path / "annotated_out.mp4"
    config = AppConfig.load() if Path("configs/default.yaml").exists() else AppConfig()
    processor = CCTVVideoProcessor(config=config)

    roi_cfg = ROIConfig(
        roi_id="test_zone",
        name="Test Corridor",
        coordinates=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        resolution={"width": 320, "height": 240},
    )
    roi_file = tmp_path / "test_roi.json"
    roi_file.write_text(roi_cfg.model_dump_json())

    summary = processor.process_video(
        video_path=test_video,
        output_video_path=test_out,
        roi_config_path=roi_file,
    )

    assert summary["total_frames"] == 15
    assert test_out.exists()
    assert test_out.stat().st_size > 0
