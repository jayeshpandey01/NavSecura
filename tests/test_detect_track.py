"""Unit test suite for VideoStreamReader, Detector, and Tracker protocols."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from navsecura.detect import Detection, Detector, YOLODetector
from navsecura.io.video import VideoStreamReader
from navsecura.track import Tracker, Tracklet, UltralyticsTracker


class MockDetector(Detector):
    """Mock implementation conforming to the Detector protocol."""

    def __init__(self) -> None:
        self._last_ms: float = 1.5

    @property
    def name(self) -> str:
        return "MockDetector"

    @property
    def last_inference_ms(self) -> float:
        return self._last_ms

    @property
    def device(self) -> str:
        return "cpu"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return [
            Detection(box=(50.0, 50.0, 100.0, 150.0), confidence=0.92, class_id=0, class_name="person")
        ]


class MockTracker(Tracker):
    """Mock implementation conforming to the Tracker protocol."""

    def __init__(self) -> None:
        self._last_ms: float = 0.5
        self._tracks: List[Tracklet] = []

    @property
    def name(self) -> str:
        return "MockTracker"

    @property
    def last_update_ms(self) -> float:
        return self._last_ms

    @property
    def active_tracks_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
    ) -> List[Tracklet]:
        self._tracks = [
            Tracklet(
                track_id=1,
                box=d.box,
                confidence=d.confidence,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                trajectory=[d.foot_point],
            )
            for d in detections
        ]
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()


def test_detector_protocol_conformance() -> None:
    """Verify that MockDetector and YOLODetector satisfy the runtime Detector protocol."""
    mock = MockDetector()
    assert isinstance(mock, Detector)
    assert mock.name == "MockDetector"
    assert mock.last_inference_ms == 1.5

    yolo = YOLODetector(weights="yolo11n.pt")
    assert isinstance(yolo, Detector)
    assert "YOLO" in yolo.name


def test_tracker_protocol_conformance() -> None:
    """Verify that MockTracker and UltralyticsTracker satisfy the runtime Tracker protocol."""
    mock = MockTracker()
    assert isinstance(mock, Tracker)
    assert mock.name == "MockTracker"
    assert mock.active_tracks_count == 0

    trk = UltralyticsTracker()
    assert isinstance(trk, Tracker)
    assert "ByteTrack" in trk.name


def test_detection_dataclass_geometry() -> None:
    """Validate spatial projections and geometric properties of Detection."""
    det = Detection(box=(100.0, 200.0, 300.0, 600.0), confidence=0.85, class_id=0)
    assert det.x1 == 100.0
    assert det.y1 == 200.0
    assert det.x2 == 300.0
    assert det.y2 == 600.0
    assert det.width == 200.0
    assert det.height == 400.0
    assert det.area == 80000.0

    # Foot-point must strictly evaluate ground contact: ((x1+x2)/2, y2)
    assert det.foot_point == (200.0, 600.0)
    assert det.center == (200.0, 400.0)
    assert det.as_xyxy() == (100.0, 200.0, 300.0, 600.0)
    assert det.as_xywh() == (200.0, 400.0, 200.0, 400.0)


def test_tracklet_dataclass_geometry() -> None:
    """Validate spatial projections of Tracklet."""
    t = Tracklet(
        track_id=42,
        box=(50.0, 100.0, 150.0, 300.0),
        confidence=0.91,
        frame_idx=10,
        timestamp_s=0.333,
        trajectory=[(100.0, 300.0)],
    )
    assert t.track_id == 42
    assert t.foot_point == (100.0, 300.0)
    assert t.center == (100.0, 200.0)
    assert t.width == 100.0
    assert t.height == 200.0


def test_yolo_detector_inference() -> None:
    """Test YOLO detector on blank frame and real video frame."""
    detector = YOLODetector(weights="yolo11n.pt", confidence_threshold=0.35)

    # 1. Blank frame smoke test
    blank = np.zeros((432, 768, 3), dtype=np.uint8)
    blank_dets = detector.detect(blank)
    assert isinstance(blank_dets, list)
    assert len(blank_dets) == 0
    assert detector.last_inference_ms > 0.0

    # 2. Real frame from sample video (frame 40 has person P1)
    cap = cv2.VideoCapture("data/sample.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read frame 40 from data/sample.mp4"

    real_dets = detector.detect(frame)
    assert len(real_dets) >= 1
    for d in real_dets:
        assert d.confidence >= 0.35
        assert d.class_name == "person"
        assert d.x1 >= 0 and d.y1 >= 0
        assert d.x2 <= 768 and d.y2 <= 432


def test_tracker_coordinate_clipping_and_trajectory() -> None:
    """Validate that bounding boxes are clipped to canvas and trajectories are bounded."""
    tracker = UltralyticsTracker(max_trajectory_length=5)
    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Provide detection that extends slightly out of bounds
    out_of_bounds_det = [
        Detection(box=(-10.0, -5.0, 620.0, 410.0), confidence=0.88)
    ]

    tracklets = tracker.update(out_of_bounds_det, frame, frame_idx=0, timestamp_s=0.0)
    assert len(tracklets) == 1
    t = tracklets[0]

    # Verify coordinates are safely clipped
    assert t.x1 >= 0.0
    assert t.y1 >= 0.0
    assert t.x2 <= 599.0
    assert t.y2 <= 399.0

    # Feed 10 consecutive updates and verify trajectory sliding window cap
    for idx in range(1, 10):
        dets = [Detection(box=(100.0 + idx, 100.0, 200.0 + idx, 300.0), confidence=0.90)]
        out = tracker.update(dets, frame, frame_idx=idx, timestamp_s=idx * 0.1)

    assert tracker.active_tracks_count == 1
    t_final = out[0]
    assert len(t_final.trajectory) == 5, f"Expected 5 historical points, got {len(t_final.trajectory)}"


def test_video_stream_reader() -> None:
    """Test VideoStreamReader container validation, context manager, and timebase."""
    with pytest.raises(FileNotFoundError):
        VideoStreamReader("data/non_existent_stream.mp4")

    with VideoStreamReader("data/sample.mp4") as reader:
        assert reader.is_opened
        assert reader.width == 768
        assert reader.height == 432
        assert reader.native_fps == 12.0
        assert reader.total_frames == 596

        frames_read = 0
        last_ts = -1.0
        for idx, ts, frame in reader:
            assert frame.shape == (432, 768, 3)
            assert ts > last_ts  # Monotonically increasing
            assert abs(ts - (idx / 12.0)) < 1e-4  # Deterministic timebase
            last_ts = ts
            frames_read += 1
            if frames_read >= 10:
                break

        assert frames_read == 10

    # Verify closed upon exit
    assert not reader.is_opened

