"""Unit tests for multi-object tracking, visit evaluation, and parameter ablation.

Verifies:
1. Mathematical precision of temporal IoU (t-IoU) interval matching.
2. Greedy bipartite matching with competing visit candidate resolution.
3. Tracker adapter interface consistency and dataclass polymorphism (ByteTrack & BoT-SORT).
4. Spatio-temporal stitching efficacy in absorbing dropout-induced fragmentation.
5. Automated ablation sweep execution and schema validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from eval.evaluate import VisitEvaluator, compute_temporal_iou
from navsecura.detect import Detection
from navsecura.events.models import CloseReason, Visit
from navsecura.events.state_machine import VisitStateMachine
from navsecura.events.stitch import TrackletStitcher
from navsecura.roi.geometry import PolygonROI, ROIConfig
from navsecura.track import Tracklet, UltralyticsTracker


def test_temporal_iou_calculation() -> None:
    """Test mathematical accuracy of temporal IoU across interval configurations."""
    # 1. Identical intervals -> IoU = 1.0
    assert pytest.approx(compute_temporal_iou(10.0, 20.0, 10.0, 20.0)) == 1.0

    # 2. Completely disjoint intervals -> IoU = 0.0
    assert pytest.approx(compute_temporal_iou(10.0, 20.0, 25.0, 35.0)) == 0.0
    assert pytest.approx(compute_temporal_iou(30.0, 40.0, 10.0, 20.0)) == 0.0

    # 3. Touching at boundary point -> IoU = 0.0
    assert pytest.approx(compute_temporal_iou(10.0, 20.0, 20.0, 30.0)) == 0.0

    # 4. Partial overlap: [10, 20] and [15, 25]
    # Intersection = 20 - 15 = 5. Union = 25 - 10 = 15. IoU = 5 / 15 = 0.3333...
    assert pytest.approx(compute_temporal_iou(10.0, 20.0, 15.0, 25.0), rel=1e-3) == 0.3333

    # 5. Complete containment: [10, 30] and [15, 25]
    # Intersection = 10. Union = 20. IoU = 10 / 20 = 0.50
    assert pytest.approx(compute_temporal_iou(10.0, 30.0, 15.0, 25.0)) == 0.50

    # 6. Degenerate zero-duration interval -> IoU = 0.0
    assert pytest.approx(compute_temporal_iou(15.0, 15.0, 10.0, 20.0)) == 0.0


def test_bipartite_matching_hungarian() -> None:
    """Test bipartite matching resolution when multiple predicted visits compete for one ground truth."""
    evaluator = VisitEvaluator(iou_threshold=0.50)

    # 1 Ground truth visit: [10.0s, 20.0s] (10.0s duration)
    gt = [
        {"visit_id": "GT1", "person_id": "P1", "t_enter_s": 10.0, "t_exit_s": 20.0, "duration_s": 10.0},
    ]

    # Two predicted visits overlapping with GT1:
    # Pred 1: [10.2s, 19.8s] -> IoU = 9.6 / 10.0 = 0.96
    # Pred 2: [11.0s, 22.0s] -> IoU = 9.0 / 12.0 = 0.75
    # Both satisfy threshold 0.50, but Pred 1 has higher t-IoU and must win the match.
    # Pred 2 must remain unmatched (counted as False Positive).
    pred = [
        {"visit_id": "P1", "person_id": "P1", "video_t_enter_s": 10.2, "video_t_exit_s": 19.8, "duration_s": 9.6},
        {"visit_id": "P2", "person_id": "P1", "video_t_enter_s": 11.0, "video_t_exit_s": 22.0, "duration_s": 11.0},
    ]

    res = evaluator.evaluate(predicted_visits=pred, ground_truth_visits=gt)

    assert res["total_ground_truth_visits"] == 1
    assert res["total_predicted_visits"] == 2
    assert res["matched_visits"] == 1
    # Precision: 1 matched / 2 predicted = 0.50
    assert pytest.approx(res["visit_precision"]) == 0.50
    # Recall: 1 matched / 1 GT = 1.00
    assert pytest.approx(res["visit_recall"]) == 1.00
    # Fragmentation: (2 - 1) / 1 = 1.00
    assert pytest.approx(res["fragmentation_rate"]) == 1.00

    # Ensure match record links GT1 to P1 (not P2)
    assert len(res["match_records"]) == 1
    assert res["match_records"][0]["ground_truth_id"] == "GT1"
    assert res["match_records"][0]["predicted_visit_id"] == "P1"


def test_tracker_adapter_interfaces() -> None:
    """Test tracker polymorphism and interface consistency across ByteTrack and BoT-SORT."""
    frame = np.zeros((432, 768, 3), dtype=np.uint8)
    dets = [
        Detection(box=(100.0, 100.0, 200.0, 300.0), confidence=0.85, class_id=0, class_name="person"),
        Detection(box=(300.0, 150.0, 400.0, 350.0), confidence=0.75, class_id=0, class_name="person"),
    ]

    # Test ByteTrack adapter
    bytetrack = UltralyticsTracker(tracker_type="bytetrack", track_buffer=25)
    assert "ByteTrack" in bytetrack.name
    byte_tracklets = bytetrack.update(dets, frame, frame_idx=0, timestamp_s=0.0)
    assert len(byte_tracklets) == 2
    for t in byte_tracklets:
        assert isinstance(t, Tracklet)
        assert isinstance(t.track_id, int)
        assert len(t.box) == 4
        assert t.box[0] < t.box[2]
        assert t.box[1] < t.box[3]
        assert 0.0 <= t.confidence <= 1.0
        assert t.frame_idx == 0
        assert t.timestamp_s == 0.0

    # Test BoT-SORT adapter
    botsort = UltralyticsTracker(tracker_type="botsort", track_buffer=25, gmc_method="none")
    assert "BoT-SORT" in botsort.name
    bot_tracklets = botsort.update(dets, frame, frame_idx=0, timestamp_s=0.0)
    assert len(bot_tracklets) == 2
    for t in bot_tracklets:
        assert isinstance(t, Tracklet)
        assert isinstance(t.track_id, int)
        assert len(t.box) == 4
        assert t.box[0] < t.box[2]
        assert t.box[1] < t.box[3]
        assert 0.0 <= t.confidence <= 1.0

    # Reset behavior
    bytetrack.reset()
    assert bytetrack.active_tracks_count == 0
    assert bytetrack.total_unique_tracks == 0

    botsort.reset()
    assert botsort.active_tracks_count == 0
    assert botsort.total_unique_tracks == 0


def test_stitching_reduces_fragmentation() -> None:
    """Test that spatio-temporal stitching merges dropout-fragmented tracklets into 1 visit."""
    roi_cfg = ROIConfig(
        roi_id="test_roi",
        name="Test Concourse",
        camera_id="cam_01",
        coordinates=[[50, 50], [400, 50], [400, 350], [50, 350]],
    )
    roi = PolygonROI(roi_cfg, frame_width=500, frame_height=500)

    # Synthetic track:
    # Segment 1 (Track ID 1): Frames 0 to 20 (t=0.0s to 2.0s) inside ROI
    # Dropout gap: Frames 21 to 30 (t=2.1s to 3.0s, duration 1.0s)
    # Segment 2 (Track ID 2): Frames 31 to 50 (t=3.1s to 5.0s) at the exact same location
    fps = 10.0
    box_pos = (150.0, 150.0, 220.0, 300.0)

    # 1. State machine without stitching -> emits 2 fragmented visits
    sm_unstitched = VisitStateMachine(roi=roi, grace_period_s=0.5, min_dwell_s=0.5)
    unstitched_visits: List[Visit] = []

    # Segment 1
    for f in range(21):
        trk = Tracklet(track_id=1, box=box_pos, confidence=0.9, frame_idx=f, timestamp_s=f / fps)
        for v in sm_unstitched.update([trk], frame_idx=f, timestamp_s=f / fps):
            unstitched_visits.append(v)

    # Gap (empty frames) -> triggers grace expiry on track 1
    for f in range(21, 31):
        for v in sm_unstitched.update([], frame_idx=f, timestamp_s=f / fps):
            unstitched_visits.append(v)

    # Segment 2 (new tracker ID 2)
    for f in range(31, 51):
        trk = Tracklet(track_id=2, box=box_pos, confidence=0.9, frame_idx=f, timestamp_s=f / fps)
        for v in sm_unstitched.update([trk], frame_idx=f, timestamp_s=f / fps):
            unstitched_visits.append(v)

    # Stream end flush
    for v in sm_unstitched.force_close_all(frame_idx=50, timestamp_s=5.0, reason=CloseReason.STREAM_END):
        unstitched_visits.append(v)

    # Without stitching: 2 separate visits emitted for the same visitor
    assert len(unstitched_visits) == 2
    assert unstitched_visits[0].track_id == 1
    assert unstitched_visits[1].track_id == 2

    # 2. State machine with spatio-temporal stitching -> merges into 1 coherent visit
    stitcher = TrackletStitcher(stitch_window_s=2.0, max_stitch_dist_px=50.0)
    for v in unstitched_visits:
        stitcher.process_visit(v)

    committed = stitcher.committed_visits
    assert len(committed) == 1
    merged_visit = committed[0]
    assert merged_visit.track_id == 1
    assert 2 in merged_visit.merged_track_ids
    assert merged_visit.frame_enter == 0
    assert merged_visit.frame_exit >= 50
    assert merged_visit.duration_s >= 4.5


def test_ablation_sweep_runner() -> None:
    """Test that ablation study results JSON exists and satisfies schema constraints."""
    ablation_file = Path("eval/ablation_study.json")
    if not ablation_file.exists():
        pytest.skip("eval/ablation_study.json not yet generated; run tools/run_ablation_study.py first")

    with open(ablation_file, "r", encoding="utf-8") as f:
        study = json.load(f)

    # Verify top-level structure
    required_keys = [
        "metadata",
        "stitching_ablation",
        "grace_period_sweep",
        "min_dwell_sweep",
        "entry_hysteresis_sweep",
        "tracker_comparison",
    ]
    for key in required_keys:
        assert key in study, f"Missing key in ablation study: {key}"

    # Verify stitching ablation gain
    st = study["stitching_ablation"]
    assert "raw_tracks_without_stitching" in st
    assert "stitched_tracks_navsecura" in st
    assert "gain_summary" in st
    assert st["stitched_tracks_navsecura"]["fragmentation_rate"] <= st["raw_tracks_without_stitching"]["fragmentation_rate"]

    # Verify grace period sweep has 5 entries
    assert len(study["grace_period_sweep"]) == 5
    for row in study["grace_period_sweep"]:
        assert "grace_period_s" in row
        assert "visit_precision" in row
        assert "visit_recall" in row
        assert "mean_duration_mae_s" in row

    # Verify minimum dwell sweep has 5 entries
    assert len(study["min_dwell_sweep"]) == 5
    for row in study["min_dwell_sweep"]:
        assert "min_dwell_s" in row
        assert "short_transit_v07_recalled" in row

    # Verify tracker comparison contains bytetrack and botsort
    tc = study["tracker_comparison"]
    assert "bytetrack" in tc
    assert "botsort" in tc
    assert tc["bytetrack"]["visit_precision"] >= 0.90
    assert tc["bytetrack"]["visit_recall"] >= 0.90
