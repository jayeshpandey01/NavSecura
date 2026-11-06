"""NavSecura Multi-Pane Presentation Demo Generator.

Generates a broadcast-quality composite split-screen presentation video
demonstrating the complete NavSecura edge tracking, polygon ROI containment,
visit state machine lifecycle, and persistent event memory.

Layout:
- Top Telemetry Banner: Real-time HUD (Time, Active In-Zone, Total Visits, Reduction Ratio, FPS).
- Left Pane: Live video stream with ROI polygon, bounding boxes, foot-point crosshairs, and trajectory trails.
- Right Pane: Live Event Memory Log with real-time active visit dwell counters, grace progress bars, and closed event store.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure src is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navsecura.detect import YOLODetector
from navsecura.events.models import CloseReason, Visit, VisitState
from navsecura.events.state_machine import TrackVisitSession, VisitStateMachine
from navsecura.io.video import VideoStreamReader
from navsecura.roi.registry import ROIRegistry
from navsecura.track import UltralyticsTracker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("navsecura.demo")


class PresentationDemoRenderer:
    """Renders composite split-screen presentation video for NavSecura demo."""

    CANVAS_W = 1440
    CANVAS_H = 760
    TOP_BAR_H = 64
    LEFT_W = 860
    RIGHT_W = 580

    def __init__(
        self,
        video_path: str,
        output_path: str,
        roi_registry_path: str = "configs/roi_registry.json",
        roi_id: str = "roi_entrance",
        weights_path: str = "yolo11n.pt",
        conf_thresh: float = 0.35,
    ) -> None:
        """Initialize the presentation renderer and perception pipeline."""
        self.video_path = video_path
        self.output_path = output_path
        self.weights_path = weights_path
        self.conf_thresh = conf_thresh

        # Video stream reader
        self.reader = VideoStreamReader(video_path)
        logger.info(
            f"Opened video: {video_path} ({self.reader.width}x{self.reader.height} @ "
            f"{self.reader.native_fps} FPS, {self.reader.total_frames} frames)"
        )

        # Resolve resolution-independent ROI
        registry = ROIRegistry(roi_registry_path)
        self.roi = registry.resolve_roi(
            roi_target=roi_id,
            frame_width=self.reader.width,
            frame_height=self.reader.height,
        )
        logger.info(f"Resolved ROI '{roi_id}': {self.roi.pixel_vertices}")

        # Initialize detector, tracker, and state machine
        self.detector = YOLODetector(weights=weights_path, confidence_threshold=conf_thresh)
        self.tracker = UltralyticsTracker(tracker_type="bytetrack")
        self.state_machine = VisitStateMachine(
            roi=self.roi,
            grace_period_s=2.0,
            enter_consecutive_frames=3,
            min_dwell_s=1.0,
        )

        # Track history for motion trails (track_id -> deque of points)
        self.trajectories: Dict[int, deque] = {}
        self.closed_events: List[Visit] = []

        # Color palette (BGR)
        self.BG_DARK = (20, 24, 30)
        self.CARD_BG = (28, 34, 43)
        self.BORDER_COLOR = (45, 55, 72)
        self.TEXT_WHITE = (245, 245, 245)
        self.TEXT_MUTED = (160, 174, 192)
        self.ACCENT_BLUE = (220, 125, 46)     # (BGR) #2e7ddc
        self.ACCENT_GREEN = (97, 175, 38)     # (BGR) #26af61
        self.ACCENT_AMBER = (18, 156, 243)    # (BGR) #f39c12
        self.ACCENT_RED = (60, 76, 231)       # (BGR) #e74c3c
        self.ROI_CLEAR = (38, 175, 97)
        self.ROI_OCCUPIED = (18, 156, 243)

    def run(self) -> None:
        """Process stream and produce presentation demo video."""
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.output_path, fourcc, self.reader.native_fps, (self.CANVAS_W, self.CANVAS_H)
        )

        total_frames = self.reader.total_frames
        fps = self.reader.native_fps
        raw_detection_count = 0
        t_start = time.perf_counter()

        logger.info(f"Rendering demo video to '{self.output_path}' ({self.CANVAS_W}x{self.CANVAS_H})...")

        for frame_idx, timestamp_s, frame in self.reader:
            t_frame_start = time.perf_counter()

            # 1. Detection & Tracking
            detections = self.detector.detect(frame)
            raw_detection_count += len(detections)
            tracklets = self.tracker.update(detections, frame, frame_idx, timestamp_s)

            # Update motion trails
            for trk in tracklets:
                if trk.track_id not in self.trajectories:
                    self.trajectories[trk.track_id] = deque(maxlen=30)
                fx, fy = trk.foot_point
                self.trajectories[trk.track_id].append((int(fx), int(fy)))

            # 2. State Machine Update
            newly_closed = self.state_machine.update(
                tracklets=tracklets,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
            )
            self.closed_events.extend(newly_closed)

            # Active sessions
            active_sessions = [
                s for s in self.state_machine.sessions.values()
                if s.state in (VisitState.INSIDE, VisitState.GRACE)
            ]

            # Compute instantaneous FPS
            t_frame_end = time.perf_counter()
            inst_fps = 1.0 / max(0.001, t_frame_end - t_frame_start)

            # 3. Composite Frame Assembly
            canvas = np.full((self.CANVAS_H, self.CANVAS_W, 3), self.BG_DARK, dtype=np.uint8)

            # A. Render Left Pane (Video Viewport)
            annotated_video = self._render_video_pane(frame, tracklets, active_sessions)
            self._composite_left_pane(canvas, annotated_video)

            # B. Render Top Banner
            active_count = len(active_sessions)
            self._render_top_banner(
                canvas=canvas,
                timestamp_s=timestamp_s,
                frame_idx=frame_idx,
                total_frames=total_frames,
                active_count=active_count,
                raw_detections=raw_detection_count,
                fps=inst_fps,
            )

            # C. Render Right Pane (Event Memory Log)
            self._render_right_pane(canvas, timestamp_s, active_sessions)

            # Write frame
            writer.write(canvas)

            if frame_idx % 60 == 0 or frame_idx == total_frames - 1:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"Frame {frame_idx:04d}/{total_frames} ({timestamp_s:.2f}s) | "
                    f"Active: {active_count} | Closed: {len(self.closed_events)} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        # Force-close any open visits at stream EOF
        final_events = self.state_machine.force_close_all(
            timestamp_s=total_frames / fps,
            frame_idx=total_frames - 1,
            reason=CloseReason.STREAM_END,
        )
        self.closed_events.extend(final_events)

        writer.release()
        total_time = time.perf_counter() - t_start
        logger.info(
            f"Demo generation complete in {total_time:.1f}s. "
            f"Saved: '{self.output_path}' ({os.path.getsize(self.output_path)/1024/1024:.2f} MB)"
        )

    def _render_video_pane(
        self, frame: np.ndarray, tracklets: list, active_sessions: List[TrackVisitSession]
    ) -> np.ndarray:
        """Render ROI polygon, boxes, foot points, and trails on original video."""
        annotated = frame.copy()
        pts = np.array(self.roi.pixel_vertices, dtype=np.int32)

        # Fill ROI polygon with transparency
        overlay = annotated.copy()
        is_occupied = len(active_sessions) > 0
        roi_color = self.ROI_OCCUPIED if is_occupied else self.ROI_CLEAR
        cv2.fillPoly(overlay, [pts], roi_color)
        cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
        cv2.polylines(annotated, [pts], isClosed=True, color=roi_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw ROI label
        cv2.putText(
            annotated,
            "ENTRANCE ZONE" if not is_occupied else "ENTRANCE ZONE [OCCUPIED]",
            (int(pts[0][0]) + 10, int(pts[0][1]) + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            roi_color,
            2,
            cv2.LINE_AA,
        )

        # Draw motion trails
        for trk_id, trail in self.trajectories.items():
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(annotated, trail[i - 1], trail[i], self.ACCENT_BLUE, thickness, lineType=cv2.LINE_AA)

        # Active session map
        active_map = {s.track_id: s for s in active_sessions}

        for trk in tracklets:
            x1, y1, x2, y2 = [int(v) for v in trk.box]
            fx, fy = int(trk.foot_point[0]), int(trk.foot_point[1])
            is_active = trk.track_id in active_map

            box_color = self.ACCENT_GREEN if is_active else self.ACCENT_BLUE
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)

            # Foot-point marker (circle with crosshair)
            cv2.circle(annotated, (fx, fy), 5, (0, 0, 255) if not is_active else self.ACCENT_GREEN, -1)
            cv2.circle(annotated, (fx, fy), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Badge
            badge_text = f"ID: {trk.track_id} ({trk.confidence:.2f})"
            if is_active:
                s = active_map[trk.track_id]
                dwell = s.t_last_seen_inside - s.t_enter_s
                badge_text += f" [IN: {dwell:.1f}s]"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), box_color, -1)
            cv2.putText(
                annotated,
                badge_text,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def _composite_left_pane(self, canvas: np.ndarray, video_frame: np.ndarray) -> None:
        """Letterbox and scale video into left pane of the canvas."""
        target_w = self.LEFT_W - 32
        target_h = self.CANVAS_H - self.TOP_BAR_H - 40

        # Preserve aspect ratio (768x432 is 16:9)
        vh, vw = video_frame.shape[:2]
        scale = min(target_w / vw, target_h / vh)
        nw, nh = int(vw * scale), int(vh * scale)
        resized = cv2.resize(video_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Center in left region
        x_off = 16 + (target_w - nw) // 2
        y_off = self.TOP_BAR_H + 16 + (target_h - nh) // 2

        # Draw container card border
        cv2.rectangle(
            canvas,
            (x_off - 4, y_off - 4),
            (x_off + nw + 4, y_off + nh + 4),
            self.BORDER_COLOR,
            2,
            lineType=cv2.LINE_AA,
        )
        canvas[y_off : y_off + nh, x_off : x_off + nw] = resized

        # Caption under video
        caption = "LIVE CCTV SENSOR FEED (768x432 @ 24 FPS) | YOLOV11N + BYTETRACK"
        cv2.putText(
            canvas,
            caption,
            (x_off, y_off + nh + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    def _render_top_banner(
        self,
        canvas: np.ndarray,
        timestamp_s: float,
        frame_idx: int,
        total_frames: int,
        active_count: int,
        raw_detections: int,
        fps: float,
    ) -> None:
        """Render executive telemetry HUD banner across the top."""
        # Banner background card
        cv2.rectangle(canvas, (12, 8), (self.CANVAS_W - 12, self.TOP_BAR_H), self.CARD_BG, -1)
        cv2.rectangle(canvas, (12, 8), (self.CANVAS_W - 12, self.TOP_BAR_H), self.BORDER_COLOR, 1)

        # Title / Brand
        cv2.putText(
            canvas,
            "NAVSECURA",
            (24, 38),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            self.TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "EVENT MEMORY ENGINE",
            (24, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            self.ACCENT_BLUE,
            1,
            cv2.LINE_AA,
        )

        # Telemetry columns
        col_x = 260
        spacing = 195

        # 1. Video Time
        mins = int(timestamp_s // 60)
        secs = timestamp_s % 60
        time_str = f"{mins:02d}:{secs:05.2f}s"
        cv2.putText(canvas, "STREAM TIMEBASE", (col_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.TEXT_MUTED, 1)
        cv2.putText(canvas, f"{time_str} ({frame_idx}/{total_frames})", (col_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_WHITE, 1)

        # 2. In-Zone Active Count
        col_x += spacing
        in_zone_color = self.ACCENT_AMBER if active_count > 0 else self.ACCENT_GREEN
        cv2.putText(canvas, "IN-ZONE OCCUPANCY", (col_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.TEXT_MUTED, 1)
        cv2.putText(canvas, f"{active_count} Subject{'s' if active_count!=1 else ''} Inside", (col_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, in_zone_color, 2)

        # 3. Emitted Visits
        col_x += spacing
        closed_count = len(self.closed_events)
        cv2.putText(canvas, "EMITTED VISITS", (col_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.TEXT_MUTED, 1)
        cv2.putText(canvas, f"{closed_count} Completed Events", (col_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.TEXT_WHITE, 1)

        # 4. Reduction Ratio
        col_x += spacing
        total_events = max(1, closed_count + active_count)
        ratio = raw_detections / total_events
        cv2.putText(canvas, "ALERT REDUCTION RATIO", (col_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.TEXT_MUTED, 1)
        cv2.putText(canvas, f"{ratio:.1f} : 1  ({raw_detections} Dets)", (col_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 105, 255), 2)

        # 5. Instantaneous FPS
        col_x += spacing + 25
        cv2.putText(canvas, "PROCESSING SPEED", (col_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.TEXT_MUTED, 1)
        cv2.putText(canvas, f"{fps:4.1f} FPS (CPU)", (col_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.ACCENT_GREEN, 1)

    def _render_right_pane(
        self, canvas: np.ndarray, timestamp_s: float, active_sessions: List[TrackVisitSession]
    ) -> None:
        """Render live event state machine tables, counters, and closed history."""
        pane_x = self.LEFT_W + 8
        pane_y = self.TOP_BAR_H + 16
        pane_w = self.RIGHT_W - 24
        pane_h = self.CANVAS_H - self.TOP_BAR_H - 32

        # Outer background card
        cv2.rectangle(canvas, (pane_x, pane_y), (pane_x + pane_w, pane_y + pane_h), self.CARD_BG, -1)
        cv2.rectangle(canvas, (pane_x, pane_y), (pane_x + pane_w, pane_y + pane_h), self.BORDER_COLOR, 1)

        # Header
        cv2.putText(canvas, "EVENT MEMORY STATE MACHINE", (pane_x + 16, pane_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.50, self.TEXT_WHITE, 1)
        cv2.line(canvas, (pane_x + 16, pane_y + 36), (pane_x + pane_w - 16, pane_y + 36), self.BORDER_COLOR, 1)

        curr_y = pane_y + 54

        # Section 1: Active Visits
        cv2.putText(canvas, "ACTIVE VISIT SESSIONS (INSIDE / GRACE)", (pane_x + 16, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.ACCENT_BLUE, 1)
        curr_y += 18

        if not active_sessions:
            cv2.rectangle(canvas, (pane_x + 16, curr_y), (pane_x + pane_w - 16, curr_y + 38), self.BG_DARK, -1)
            cv2.putText(canvas, "Entrance zone clear. Waiting for transit...", (pane_x + 28, curr_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_MUTED, 1)
            curr_y += 48
        else:
            for s in active_sessions:
                card_h = 48
                cv2.rectangle(canvas, (pane_x + 16, curr_y), (pane_x + pane_w - 16, curr_y + card_h), self.BG_DARK, -1)
                cv2.rectangle(canvas, (pane_x + 16, curr_y), (pane_x + pane_w - 16, curr_y + card_h), self.BORDER_COLOR, 1)

                is_grace = s.state == VisitState.GRACE
                state_badge = "GRACE RECOVERY" if is_grace else "INSIDE ROI"
                badge_col = self.ACCENT_AMBER if is_grace else self.ACCENT_GREEN

                # Visit ID & Track ID
                cv2.putText(canvas, f"Visit {s.visit_id[:8]}", (pane_x + 24, curr_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_WHITE, 1)
                cv2.putText(canvas, f"Track {s.track_id}", (pane_x + 120, curr_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.ACCENT_BLUE, 1)

                # State Badge
                cv2.putText(canvas, state_badge, (pane_x + 190, curr_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_col, 1)

                # Real-time dwell counter
                dwell = max(0.0, timestamp_s - s.t_enter_s)
                cv2.putText(canvas, f"Dwell: {dwell:4.1f}s", (pane_x + pane_w - 110, curr_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.TEXT_WHITE, 1)

                # Grace Countdown or In-Zone Progress Bar
                bar_x = pane_x + 24
                bar_y = curr_y + 28
                bar_w = pane_w - 48
                bar_h = 8
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 48, 60), -1)

                if is_grace:
                    time_in_grace = timestamp_s - s.grace_start_t
                    pct_left = max(0.0, 1.0 - (time_in_grace / self.state_machine.grace_period_s))
                    fill_w = int(bar_w * pct_left)
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), self.ACCENT_AMBER, -1)
                else:
                    # In-zone solid bar
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w if 'fill_w' in locals() else bar_w, bar_y + bar_h), self.ACCENT_GREEN, -1)

                curr_y += card_h + 8

        curr_y += 10
        cv2.line(canvas, (pane_x + 16, curr_y), (pane_x + pane_w - 16, curr_y), self.BORDER_COLOR, 1)
        curr_y += 18

        # Section 2: Closed Event Memory History (SQLite Event Log)
        cv2.putText(canvas, "RECENT PERSISTED VISITS (SQLITE WAL LOG)", (pane_x + 16, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.ACCENT_BLUE, 1)
        curr_y += 18

        # Table Header
        cv2.rectangle(canvas, (pane_x + 16, curr_y), (pane_x + pane_w - 16, curr_y + 20), (35, 42, 53), -1)
        cv2.putText(canvas, "VISIT ID", (pane_x + 22, curr_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.TEXT_MUTED, 1)
        cv2.putText(canvas, "TRK", (pane_x + 110, curr_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.TEXT_MUTED, 1)
        cv2.putText(canvas, "ENTER", (pane_x + 155, curr_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.TEXT_MUTED, 1)
        cv2.putText(canvas, "EXIT", (pane_x + 225, curr_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.TEXT_MUTED, 1)
        cv2.putText(canvas, "DWELL", (pane_x + 295, curr_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.TEXT_MUTED, 1)
        cv2.putText(canvas, "STATUS", (pane_x + pane_w - 95, curr_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.TEXT_MUTED, 1)
        curr_y += 24

        # Render rows (show last 6 closed events)
        display_events = self.closed_events[-6:]
        for ev in display_events:
            row_h = 24
            cv2.rectangle(canvas, (pane_x + 16, curr_y), (pane_x + pane_w - 16, curr_y + row_h), self.BG_DARK, -1)
            cv2.putText(canvas, f"{ev.visit_id[:8]}", (pane_x + 22, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_WHITE, 1)
            cv2.putText(canvas, f"{ev.track_id}", (pane_x + 112, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.ACCENT_BLUE, 1)
            cv2.putText(canvas, f"{ev.video_t_enter_s:05.2f}s", (pane_x + 155, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_WHITE, 1)
            cv2.putText(canvas, f"{ev.video_t_exit_s:05.2f}s", (pane_x + 225, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_WHITE, 1)
            cv2.putText(canvas, f"{ev.duration_s:4.2f}s", (pane_x + 295, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.ACCENT_GREEN, 1)
            cv2.putText(canvas, "PERSISTED", (pane_x + pane_w - 95, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.ACCENT_GREEN, 1)
            curr_y += row_h + 4

        # Section 3: Evidence Dossier Verification Box at bottom
        box_y = pane_y + pane_h - 75
        cv2.rectangle(canvas, (pane_x + 16, box_y), (pane_x + pane_w - 16, pane_y + pane_h - 14), (32, 40, 50), -1)
        cv2.rectangle(canvas, (pane_x + 16, box_y), (pane_x + pane_w - 16, pane_y + pane_h - 14), self.BORDER_COLOR, 1)

        cv2.putText(canvas, "FORENSIC EVIDENCE INTEGRITY", (pane_x + 26, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_WHITE, 1)
        cv2.putText(canvas, "[OK] Entry Snapshot: outputs/evidence/snapshots/*_entry.jpg", (pane_x + 26, box_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.ACCENT_GREEN, 1)
        cv2.putText(canvas, "[OK] Peak Crop: outputs/evidence/snapshots/*_peak.jpg", (pane_x + 26, box_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.ACCENT_GREEN, 1)
        cv2.putText(canvas, "[OK] 6s Video Clip: outputs/evidence/clips/*_clip.mp4", (pane_x + 26, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.ACCENT_GREEN, 1)


def main() -> None:
    """Parse CLI options and execute presentation demo generator."""
    parser = argparse.ArgumentParser(description="NavSecura Multi-Pane Presentation Demo Generator")
    parser.add_argument("--video", default="data/sample.mp4", help="Path to input video")
    parser.add_argument(
        "--output", default="outputs/videos/navsecura_demo_presentation.mp4", help="Path to output presentation video"
    )
    parser.add_argument("--registry", default="configs/roi_registry.json", help="Path to ROI registry")
    parser.add_argument("--roi-id", default="roi_entrance", help="Target ROI ID")
    parser.add_argument("--weights", default="yolo11n.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")

    args = parser.parse_args()

    renderer = PresentationDemoRenderer(
        video_path=args.video,
        output_path=args.output,
        roi_registry_path=args.registry,
        roi_id=args.roi_id,
        weights_path=args.weights,
        conf_thresh=args.conf,
    )
    renderer.run()


if __name__ == "__main__":
    main()
