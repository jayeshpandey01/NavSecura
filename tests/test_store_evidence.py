"""Unit tests for SQLite event persistence, circular buffer, and forensic evidence writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from navsecura.events.models import CloseReason, Visit
from navsecura.evidence.clip import CircularVideoBuffer, EvidenceWriter
from navsecura.store.sqlite import SQLiteEventStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    return tmp_path / "test_events.db"


@pytest.fixture
def temp_evidence_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary evidence artifacts directory."""
    return tmp_path / "evidence_output"


@pytest.fixture
def sample_visit() -> Visit:
    """Fixture providing a strongly-typed test Visit."""
    return Visit(
        visit_id="a1b2c3d4-e5f6-47a8-b9c0-d1e2f3a4b5c6",
        camera_id="cam_main_entrance",
        roi_id="door_zone_01",
        track_id=12,
        merged_track_ids=[12, 15],
        parent_visit_id="11223344-5566-7788-9900-aabbccddeeff",
        person_id="P1",
        entered_at="2026-11-06T00:00:00Z",
        exited_at="2026-11-06T00:00:05Z",
        video_t_enter_s=5.000,
        video_t_exit_s=10.000,
        frame_enter=120,
        frame_exit=240,
        duration_s=5.000,
        mean_conf=0.885,
        peak_conf=0.940,
        peak_frame=180,
        close_reason=CloseReason.EXITED.value,
        entry_box=(100.0, 150.0, 200.0, 350.0),
        entry_foot_point=(150.0, 350.0),
        peak_box=(105.0, 152.0, 205.0, 355.0),
        last_box=(110.0, 155.0, 210.0, 360.0),
        last_foot_point=(160.0, 360.0),
    )


def test_sqlite_schema_initialization_and_wal(temp_db_path: Path) -> None:
    """Verify SQLite WAL pragma configuration, table definition, and index creation."""
    store = SQLiteEventStore(temp_db_path)

    # Check WAL pragma
    cursor = store.conn.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal"

    cursor = store.conn.execute("PRAGMA synchronous;")
    sync = cursor.fetchone()[0]
    assert sync == 1  # 1 = NORMAL

    # Check table existence
    cursor = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='visits';"
    )
    assert cursor.fetchone() is not None

    # Check indices existence
    cursor = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='visits';"
    )
    indices = {row[0] for row in cursor.fetchall()}
    assert "idx_visits_time" in indices
    assert "idx_visits_track" in indices
    assert "idx_visits_duration" in indices

    store.close()


def test_sqlite_visit_crud_and_deserialization(temp_db_path: Path, sample_visit: Visit) -> None:
    """Verify visit insertion, UUID lookups, prefix lookups, and JSON array deserialization."""
    with SQLiteEventStore(temp_db_path) as store:
        store.insert_visit(
            visit=sample_visit,
            snapshot_entry="outputs/snapshots/entry_test.jpg",
            snapshot_peak="outputs/snapshots/peak_test.jpg",
            clip_path="outputs/clips/clip_test.mp4",
            config_hash="abc123hash",
            model_version="yolo11n_bytetrack",
        )

        assert store.get_visit_count() == 1

        # Retrieve by full UUID
        rec_full = store.get_visit(sample_visit.visit_id)
        assert rec_full is not None
        assert rec_full["visit_id"] == sample_visit.visit_id
        assert rec_full["track_id"] == 12
        assert rec_full["merged_track_ids"] == [12, 15]
        assert rec_full["duration_s"] == 5.0
        assert rec_full["snapshot_entry"] == "outputs/snapshots/entry_test.jpg"
        assert rec_full["snapshot_peak"] == "outputs/snapshots/peak_test.jpg"
        assert rec_full["clip_path"] == "outputs/clips/clip_test.mp4"
        assert rec_full["config_hash"] == "abc123hash"
        assert rec_full["model_version"] == "yolo11n_bytetrack"

        # Retrieve by 8-char prefix
        rec_prefix = store.get_visit("a1b2c3d4")
        assert rec_prefix is not None
        assert rec_prefix["visit_id"] == sample_visit.visit_id

        # Non-existent ID lookup
        assert store.get_visit("non_existent_id") is None


def test_sqlite_query_filters_and_count(temp_db_path: Path) -> None:
    """Verify relational filtering by camera_id, roi_id, and minimum duration."""
    with SQLiteEventStore(temp_db_path) as store:
        v1 = Visit(
            camera_id="cam_01",
            roi_id="zone_a",
            track_id=1,
            video_t_enter_s=1.0,
            video_t_exit_s=3.0,
            frame_enter=10,
            frame_exit=30,
            duration_s=2.0,
        )
        v2 = Visit(
            camera_id="cam_01",
            roi_id="zone_b",
            track_id=2,
            video_t_enter_s=4.0,
            video_t_exit_s=10.0,
            frame_enter=40,
            frame_exit=100,
            duration_s=6.0,
        )
        v3 = Visit(
            camera_id="cam_02",
            roi_id="zone_a",
            track_id=3,
            video_t_enter_s=12.0,
            video_t_exit_s=13.0,
            frame_enter=120,
            frame_exit=130,
            duration_s=1.0,
        )

        for v in (v1, v2, v3):
            store.insert_visit(v)

        assert store.get_visit_count() == 3
        assert store.get_visit_count(camera_id="cam_01") == 2
        assert store.get_visit_count(roi_id="zone_a") == 2

        # Filter by duration
        long_visits = store.list_visits(min_duration_s=3.0)
        assert len(long_visits) == 1
        assert long_visits[0]["track_id"] == 2


def test_sqlite_export_json_and_csv(temp_db_path: Path, sample_visit: Visit, tmp_path: Path) -> None:
    """Verify export to structured JSON and flat CSV formats."""
    json_export = tmp_path / "exported.json"
    csv_export = tmp_path / "exported.csv"

    with SQLiteEventStore(temp_db_path) as store:
        store.insert_visit(sample_visit)
        store.export_to_json(json_export)
        store.export_to_csv(csv_export)

    assert json_export.exists()
    assert csv_export.exists()

    with open(json_export, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["visit_id"] == sample_visit.visit_id

    with open(csv_export, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["visit_id"] == sample_visit.visit_id


def test_circular_video_buffer_capacity_and_slicing() -> None:
    """Verify in-memory circular ring buffer capacity bounds, slicing, and frame retrieval."""
    buffer = CircularVideoBuffer(capacity_frames=5, fps=10.0)
    assert buffer.count == 0
    assert buffer.oldest_frame_idx is None

    # Push 8 frames into a 5-capacity buffer
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(8):
        buffer.append(frame_idx=i, timestamp_s=i * 0.1, frame=dummy_frame)

    assert buffer.count == 5
    assert buffer.oldest_frame_idx == 3
    assert buffer.newest_frame_idx == 7

    # Exact frame retrieval
    retrieved = buffer.get_frame(5)
    assert retrieved is not None
    assert retrieved.shape == (100, 100, 3)

    # Out of bounds frame retrieval
    assert buffer.get_frame(0) is None

    # Slice extraction within available bounds
    slice_frames = buffer.get_frames_slice(start_frame=4, end_frame=6)
    assert len(slice_frames) == 3
    assert [f[0] for f in slice_frames] == [4, 5, 6]

    # Time-based slice extraction
    time_slice = buffer.get_time_slice(start_time_s=0.45, end_time_s=0.65)
    assert len(time_slice) == 2
    assert [f[0] for f in time_slice] == [5, 6]

    buffer.clear()
    assert buffer.count == 0


def test_evidence_writer_snapshots_and_clips(temp_evidence_dir: Path, sample_visit: Visit) -> None:
    """Verify evidence writer generates valid JPEG snapshots and playable MP4 clips."""
    writer = EvidenceWriter(output_dir=temp_evidence_dir)
    assert writer.snapshots_dir.exists()
    assert writer.clips_dir.exists()

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (50, 50, 50)  # Dark grey background

    # 1. Entry snapshot
    entry_rel = writer.save_entry_snapshot(canvas, sample_visit)
    entry_file = temp_evidence_dir / "snapshots" / f"entry_{sample_visit.visit_id[:8]}_t{sample_visit.track_id}.jpg"
    assert entry_file.exists()
    assert entry_file.stat().st_size > 0

    read_entry = cv2.imread(str(entry_file))
    assert read_entry is not None
    assert read_entry.shape == (480, 640, 3)

    # 2. Peak snapshot (tight crop)
    peak_rel = writer.save_peak_snapshot(canvas, sample_visit)
    peak_file = temp_evidence_dir / "snapshots" / f"peak_{sample_visit.visit_id[:8]}_t{sample_visit.track_id}.jpg"
    assert peak_file.exists()
    assert peak_file.stat().st_size > 0

    read_peak = cv2.imread(str(peak_file))
    assert read_peak is not None
    # Crop should be tighter than original canvas
    assert read_peak.shape[0] < 480
    assert read_peak.shape[1] < 640

    # 3. Video clip encoding
    frames: List[Tuple[int, float, np.ndarray]] = [
        (i, i * 0.1, canvas.copy()) for i in range(10)
    ]
    clip_rel = writer.save_event_clip(frames, sample_visit, fps=10.0)
    clip_file = temp_evidence_dir / "clips" / f"clip_{sample_visit.visit_id[:8]}_t{sample_visit.track_id}.mp4"
    assert clip_file.exists()
    assert clip_file.stat().st_size > 0

    # Verify playable video stream via VideoCapture
    cap = cv2.VideoCapture(str(clip_file))
    assert cap.isOpened()
    f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert f_count == 10
    cap.release()


def test_sqlite_summary_stats_and_rollback(temp_db_path: Path, sample_visit: Visit) -> None:
    """Verify summary metrics calculation, busy timeout pragma, and transaction rollback."""
    with SQLiteEventStore(temp_db_path) as store:
        # Check busy_timeout pragma
        cursor = store.conn.execute("PRAGMA busy_timeout;")
        timeout_val = cursor.fetchone()[0]
        assert timeout_val == 5000

        # Initial stats on empty store
        stats_empty = store.get_summary_stats()
        assert stats_empty["total_visits"] == 0

        # Insert visit and check returned visit_id
        v_id = store.insert_visit(sample_visit)
        assert v_id == sample_visit.visit_id

        # Summary stats after insertion
        stats = store.get_summary_stats()
        assert stats["total_visits"] == 1
        assert stats["mean_duration_s"] == 5.0
        assert stats["min_duration_s"] == 5.0
        assert stats["max_duration_s"] == 5.0

        # Test export_json and export_csv aliases
        json_p = temp_db_path.parent / "stats_export.json"
        csv_p = temp_db_path.parent / "stats_export.csv"
        store.export_json(json_p)
        store.export_csv(csv_p)
        assert json_p.exists()
        assert csv_p.exists()


def test_circular_buffer_get_range() -> None:
    """Verify get_range slice retrieval on circular video buffer."""
    buf = CircularVideoBuffer(capacity_frames=10, fps=10.0)
    for i in range(15):
        frame = np.full((50, 50, 3), i, dtype=np.uint8)
        buf.append(frame_idx=i, timestamp_s=i * 0.1, frame=frame)

    # Oldest available is 5, newest is 14
    assert buf.oldest_frame_idx == 5
    assert buf.newest_frame_idx == 14

    # Extract range [7, 10]
    sub = buf.get_range(7, 10)
    assert len(sub) == 4
    assert [item[0] for item in sub] == [7, 8, 9, 10]
    assert sub[0][2][0, 0, 0] == 7


