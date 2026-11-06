---
name: visit-event-guard
description: Implements, verifies, and audits the visit lifecycle state machine, foot-point polygon ROI containment, hysteresis, grace timeouts, and spatio-temporal ID stitching. Use whenever editing event logic, state transitions, ROI geometries, or writing synthetic track tests.
---

# Visit Event Guard

## Goal
Ensure frame-by-frame bounding boxes are deterministically aggregated into exactly one coherent, non-repetitive event per visit, immune to tracker flicker, boundary jitter, and brief occlusions.

## Instructions
1. **Foot-Point Determination:** Calculate the bottom-center coordinate of the bounding box `( (x1 + x2)/2, y2 )`. Ground plane intersection avoids perspective distortion.
2. **Polygon Containment & Hysteresis:**
   - Test point-in-polygon containment using ray-casting.
   - Require `enter_frames` (default: 3) consecutive positive detections inside the ROI before transitioning from `OUTSIDE` to `INSIDE`.
   - Require `exit_frames` or `grace_s` absence before transitioning to `CLOSED`.
3. **Visit State Machine Lifecycle:**
   - `OUTSIDE`: Track exists outside ROI.
   - `INSIDE`: Track confirmed inside ROI. Record `t_enter = video_t_enter_s`. Record `t_last_seen_inside`.
   - `GRACE`: Track missing or outside ROI. Start grace countdown (`grace_s`, default: 2.0s). If re-detected inside before timeout, return to `INSIDE`.
   - `CLOSED`: Grace timer expired. Calculate `duration = t_last_seen_inside - t_enter`. If `duration >= min_dwell` (default: 1.0s), commit visit event; else discard as transient false positive.
4. **Stream-End Invariant:** On EOF or stream disruption, force-close all open `INSIDE` or `GRACE` visits with `close_reason = "stream_end"`.
5. **ID Stitching:** When a track terminates and a new track originates within `stitch_window_s` and spatial distance threshold $D_{\max}$, evaluate bounding box proximity and optional ReID cosine similarity to merge into the parent visit.
6. **Synthetic Unit Testing:** Maintain comprehensive unit tests in `tests/test_state_machine.py` validating behavior under:
   - Detector flicker (1-frame dropouts).
   - Boundary oscillating trajectories (walking along ROI perimeter).
   - Occlusion gaps shorter and longer than `grace_s`.
   - Immediate stream termination while subject is inside ROI.

## Examples
- Run synthetic test harness: `pytest tests/test_state_machine.py`
- Validate boundary jitter handling: simulate track alternating across ROI polygon boundary and verify only 1 visit is emitted without flicker.

## Constraints
- Never calculate dwell duration using the grace timeout timestamp; always anchor duration to `t_last_seen_inside`.
- Ensure all coordinate operations handle frame boundary clipping.
- Every state transition must be logged with timestamp and trigger reason.

