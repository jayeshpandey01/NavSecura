"""Comprehensive unit test suite for visit state machine and spatio-temporal stitching.

Tests the state machine lifecycle across failure modes F1 through F8 using
deterministic synthetic tracklet generators without requiring live model inference.
"""

from __future__ import annotations

import pytest

from navsecura.events.models import CloseReason, Visit, VisitState
from navsecura.events.state_machine import (
    TrackVisitSession,
    VisitStateMachine,
)
from navsecura.events.stitch import TrackletStitcher
from navsecura.roi.geometry import PolygonROI, ROIConfig
from navsecura.track.base import Tracklet


@pytest.fixture
def test_roi() -> PolygonROI:
    """Fixture providing a deterministic 200x200 pixel ROI polygon on a 1000x1000 canvas."""
    config = ROIConfig(
        roi_id="test_zone_01",
        name="Test Zone",
        coordinates=[(200.0, 200.0), (400.0, 200.0), (400.0, 400.0), (200.0, 400.0)],
        resolution={"width": 1000, "height": 1000},
        foot_point_mode="bottom_center",
    )
    return PolygonROI(config=config, frame_width=1000, frame_height=1000)


def create_tracklet(
    track_id: int,
    foot_x: float,
    foot_y: float,
    frame_idx: int,
    fps: float = 10.0,
    width: float = 40.0,
    height: float = 80.0,
    conf: float = 0.90,
) -> Tracklet:
    """Helper to synthesize a tracklet at a given foot-point coordinate."""
    box = (
        foot_x - width / 2.0,
        foot_y - height,
        foot_x + width / 2.0,
        foot_y,
    )
    return Tracklet(
        track_id=track_id,
        box=box,
        confidence=conf,
        frame_idx=frame_idx,
        timestamp_s=frame_idx / fps,
        trajectory=[(foot_x, foot_y)],
    )


def test_nominal_transit_lifecycle(test_roi: PolygonROI) -> None:
    """Test standard transit through ROI: OUTSIDE -> INSIDE -> CLOSED."""
    sm = VisitStateMachine(
        roi=test_roi,
        enter_consecutive_frames=3,
        grace_period_s=1.0,
        min_dwell_s=1.0,
    )

    fps = 10.0
    emitted = []

    # Frame 0 to 4: Approach outside ROI (x=100, y=300)
    for f in range(5):
        trk = create_tracklet(1, 100.0, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        emitted.extend(res)
    assert len(emitted) == 0
    assert sm.sessions[1].state == VisitState.OUTSIDE

    # Frame 5 to 7: Enter ROI (x=300, y=300) -> 3 frames hysteresis
    for f in range(5, 8):
        trk = create_tracklet(1, 300.0, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        emitted.extend(res)
    assert len(emitted) == 0
    assert sm.sessions[1].state == VisitState.INSIDE
    assert sm.sessions[1].frame_enter == 5
    assert pytest.approx(sm.sessions[1].t_enter_s) == 0.5

    # Frame 8 to 25: Remain inside (dwell = 25 - 5 = 20 frames = 2.0s)
    for f in range(8, 26):
        trk = create_tracklet(1, 300.0, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        emitted.extend(res)
    assert sm.sessions[1].state == VisitState.INSIDE

    # Frame 26: Step outside (x=500, y=300) -> GRACE starts
    trk = create_tracklet(1, 500.0, 300.0, 26, fps)
    res = sm.update([trk], 26, 26 / fps)
    emitted.extend(res)
    assert sm.sessions[1].state == VisitState.GRACE

    # Frame 27 to 37: Remain outside until grace (1.0s = 10 frames) expires
    for f in range(27, 38):
        trk = create_tracklet(1, 500.0, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        emitted.extend(res)

    assert len(emitted) == 1
    v = emitted[0]
    assert v.track_id == 1
    assert v.frame_enter == 5
    assert v.frame_exit == 25
    assert pytest.approx(v.video_t_enter_s) == 0.5
    assert pytest.approx(v.video_t_exit_s) == 2.5
    assert pytest.approx(v.duration_s) == 2.0
    assert v.close_reason == CloseReason.EXITED.value


def test_f1_detector_dropout_absorbed_by_grace(test_roi: PolygonROI) -> None:
    """Failure Mode F1: Brief detector dropouts (e.g. 3 frames) inside ROI are absorbed by grace."""
    sm = VisitStateMachine(
        roi=test_roi,
        enter_consecutive_frames=2,
        grace_period_s=2.0,
        min_dwell_s=1.0,
    )
    fps = 10.0

    # Frames 0 to 5: Inside ROI
    for f in range(6):
        trk = create_tracklet(2, 300.0, 300.0, f, fps)
        sm.update([trk], f, f / fps)
    assert sm.sessions[2].state == VisitState.INSIDE

    # Frames 6 to 8 (3 frames = 0.3s): Detector dropout (empty tracklets)
    for f in range(6, 9):
        res = sm.update([], f, f / fps)
        assert len(res) == 0
    assert sm.sessions[2].state == VisitState.GRACE

    # Frame 9: Detector recovers tracklet inside ROI -> returns to INSIDE
    trk = create_tracklet(2, 300.0, 300.0, 9, fps)
    res = sm.update([trk], 9, 9 / fps)
    assert len(res) == 0
    assert sm.sessions[2].state == VisitState.INSIDE

    # Dwell for 15 more frames then exit
    for f in range(10, 25):
        trk = create_tracklet(2, 300.0, 300.0, f, fps)
        sm.update([trk], f, f / fps)

    # Let grace expire
    emitted = []
    for f in range(25, 50):
        trk = create_tracklet(2, 600.0, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        emitted.extend(res)

    # Exactly 1 coherent visit emitted despite dropout!
    assert len(emitted) == 1
    assert emitted[0].frame_enter == 0
    assert emitted[0].frame_exit == 24


def test_f4_boundary_pacing_hysteresis(test_roi: PolygonROI) -> None:
    """Failure Mode F4: Rapid alternation across perimeter boundary is suppressed by hysteresis."""
    sm = VisitStateMachine(
        roi=test_roi,
        enter_consecutive_frames=3,
        grace_period_s=1.0,
        min_dwell_s=1.0,
    )
    fps = 10.0

    # Jitter back and forth across boundary (boundary is at x=200):
    # inside: 205, outside: 195
    for f in range(20):
        x = 205.0 if f % 2 == 0 else 195.0
        trk = create_tracklet(3, x, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        assert len(res) == 0
        # Hysteresis counter resets on every outside frame, so OUTSIDE state is preserved
        assert sm.sessions[3].state == VisitState.OUTSIDE


def test_f6_transient_false_positive_filtered(test_roi: PolygonROI) -> None:
    """Failure Mode F6: Visits shorter than min_dwell_s are filtered out."""
    sm = VisitStateMachine(
        roi=test_roi,
        enter_consecutive_frames=2,
        grace_period_s=0.5,
        min_dwell_s=1.5,  # Requires at least 1.5s
    )
    fps = 10.0

    # Entity inside ROI for only 4 frames = 0.4s (less than 1.5s)
    for f in range(4):
        trk = create_tracklet(4, 300.0, 300.0, f, fps)
        sm.update([trk], f, f / fps)
    assert sm.sessions[4].state == VisitState.INSIDE

    # Exit and let grace expire
    emitted = []
    for f in range(4, 15):
        trk = create_tracklet(4, 600.0, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        emitted.extend(res)

    # Filtered: 0 visits emitted!
    assert len(emitted) == 0


def test_f7_end_of_stream_force_close(test_roi: PolygonROI) -> None:
    """Failure Mode F7: Stream termination while person is inside ROI force-closes with 'stream_end'."""
    sm = VisitStateMachine(
        roi=test_roi,
        enter_consecutive_frames=2,
        grace_period_s=2.0,
        min_dwell_s=1.0,
    )
    fps = 10.0

    # Person remains inside ROI for 30 frames (3.0s)
    for f in range(30):
        trk = create_tracklet(5, 300.0, 300.0, f, fps)
        sm.update([trk], f, f / fps)

    assert sm.sessions[5].state == VisitState.INSIDE

    # Video stream ends at frame 29!
    emitted = sm.force_close_all(frame_idx=29, timestamp_s=2.9)
    assert len(emitted) == 1
    v = emitted[0]
    assert v.track_id == 5
    assert v.close_reason == CloseReason.STREAM_END.value
    assert v.frame_enter == 0
    assert v.frame_exit == 29
    assert pytest.approx(v.duration_s) == 2.9


def test_f8_duration_invariant_excludes_grace(test_roi: PolygonROI) -> None:
    """Failure Mode F8: Dwell duration strictly equals t_last_seen_inside - t_enter (no grace inflation)."""
    sm = VisitStateMachine(
        roi=test_roi,
        enter_consecutive_frames=1,
        grace_period_s=5.0,  # Generous 5-second grace period
        min_dwell_s=1.0,
    )
    fps = 10.0

    # Enters at frame 10 (t = 1.0s), exits at frame 30 (t = 3.0s). Real dwell = 2.0s.
    for f in range(10, 31):
        trk = create_tracklet(6, 300.0, 300.0, f, fps)
        sm.update([trk], f, f / fps)

    # Spends 5.0s (50 frames) in grace before closing
    emitted = []
    for f in range(31, 90):
        trk = create_tracklet(6, 800.0, 300.0, f, fps)
        res = sm.update([trk], f, f / fps)
        emitted.extend(res)

    assert len(emitted) == 1
    v = emitted[0]
    # Invariant check: duration must be exactly 2.0s, NOT 7.0s!
    assert pytest.approx(v.video_t_enter_s) == 1.0
    assert pytest.approx(v.video_t_exit_s) == 3.0
    assert pytest.approx(v.duration_s) == 2.0


def test_stitcher_short_occlusion_merge() -> None:
    """Test Regime 1: Fragmented tracklets from brief occlusion are stitched into 1 visit."""
    stitcher = TrackletStitcher(stitch_window_s=3.0, max_stitch_dist_px=50.0)

    # Visit 1 from Track 1: frames 10-30 (t=1.0-3.0s)
    v1 = Visit(
        roi_id="test_zone",
        track_id=1,
        video_t_enter_s=1.0,
        video_t_exit_s=3.0,
        frame_enter=10,
        frame_exit=30,
        duration_s=2.0,
        last_foot_point=(300.0, 300.0),
    )
    res1 = stitcher.process_visit(v1)
    assert len(stitcher.committed_visits) == 1

    # Visit 2 from Track 2 (ID switch after 1.5s gap, nearby position): frames 45-65 (t=4.5-6.5s)
    v2 = Visit(
        roi_id="test_zone",
        track_id=2,
        video_t_enter_s=4.5,
        video_t_exit_s=6.5,
        frame_enter=45,
        frame_exit=65,
        duration_s=2.0,
        entry_foot_point=(310.0, 305.0),  # Distance ~11px < 50px
        last_foot_point=(350.0, 320.0),
    )
    res2 = stitcher.process_visit(v2)

    # Must be merged into Visit 1
    assert len(stitcher.committed_visits) == 1
    merged = stitcher.committed_visits[0]
    assert merged.merged_track_ids == [1, 2]
    assert merged.frame_enter == 10
    assert merged.frame_exit == 65
    assert pytest.approx(merged.video_t_exit_s) == 6.5
    assert pytest.approx(merged.duration_s) == 5.5


def test_stitcher_reentry_linking() -> None:
    """Test Regime 2: Long absence (>5s) emits separate visit linked via parent_visit_id."""
    stitcher = TrackletStitcher(
        stitch_window_s=3.0,
        reentry_threshold_s=5.0,
        min_appearance_sim=0.70,
    )

    # Synthetic normalized appearance histogram
    hist_p = [0.2] * 24

    # Visit 1: Track 6 enters at 30.4s, exits at 32.75s
    v1 = Visit(
        roi_id="roi_entrance",
        track_id=6,
        video_t_enter_s=30.417,
        video_t_exit_s=32.750,
        frame_enter=365,
        frame_exit=393,
        duration_s=2.333,
        color_histogram=hist_p,
    )
    res1 = stitcher.process_visit(v1)
    assert len(stitcher.committed_visits) == 1
    assert res1.person_id == "P1"

    # Re-entry: Track 8 enters at 44.333s (absence = 11.583s > 5s)
    v2 = Visit(
        roi_id="roi_entrance",
        track_id=8,
        video_t_enter_s=44.333,
        video_t_exit_s=46.000,
        frame_enter=532,
        frame_exit=552,
        duration_s=1.667,
        color_histogram=hist_p,
    )
    res2 = stitcher.process_visit(v2)

    # Must be 2 distinct visits, but linked via parent_visit_id!
    assert len(stitcher.committed_visits) == 2
    assert res2.parent_visit_id == res1.visit_id
    assert res2.person_id == "P1"
    assert "re-entry of P1" in (res2.notes or "")
