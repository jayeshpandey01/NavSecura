"""High-level CCTV video processing and diagnostic annotation engine.

Provides end-to-end video decoding, detection, tracking, ground-plane
foot-point projection, state machine lifecycle management, and
broadcast-quality diagnostic overlay rendering for surveillance streams.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from navsecura.detect.base import Detection, Detector
from navsecura.detect.yolo import YOLODetector
from navsecura.events.models import CloseReason, VisitState
from navsecura.events.state_machine import VisitStateMachine
from navsecura.events.stitch import TrackletStitcher
from navsecura.io.config import AppConfig, VideoConfig
from navsecura.io.video import VideoStreamReader
from navsecura.roi.geometry import PolygonROI, ROIConfig
from navsecura.track.base import Tracker, Tracklet
from navsecura.track.ultralytics_trk import UltralyticsTracker

logger = logging.getLogger("navsecura.io.processor")


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, deterministic BGR color for a given track ID."""
    palette = [
        (238, 104, 76),   # Cyan/Teal
        (76, 178, 255),   # Amber/Orange
        (102, 205, 170),  # Medium Aquamarine
        (211, 85, 186),   # Violet
        (60, 20, 220),    # Crimson
        (255, 144, 30),   # Dodger Blue
        (50, 205, 50),    # Lime Green
        (0, 215, 255),    # Gold
        (180, 105, 255),  # Hot Pink
        (240, 50, 230),   # Magenta
    ]
    return palette[int(track_id) % len(palette)]


class CCTVVideoProcessor:
    """Production CCTV video processor with rich diagnostic labeling overlays.

    Handles frame ingestion, model inference, tracker updates, spatial ROI
    containment testing via ground plane foot-points, and renders explanatory
    overlays for CCTV analytics.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        detector: Optional[Detector] = None,
        tracker: Optional[Tracker] = None,
        state_machine: Optional[VisitStateMachine] = None,
    ) -> None:
        """Initialize CCTV Video Processor.

        Args:
            config: Application configuration. Defaults to AppConfig.load_default().
            detector: Object detector instance adhering to Detector protocol.
            tracker: Multi-object tracker adhering to Tracker protocol.
            state_machine: Visit state machine managing zone lifecycle.
        """
        self.config = config or (AppConfig.load() if Path("configs/default.yaml").exists() else AppConfig())
        detector_weights = getattr(self.config.detector, "weights", "yolo11n.pt")
        self.detector = detector or YOLODetector(
            weights=detector_weights,
            confidence_threshold=self.config.detector.confidence_threshold,
            iou_threshold=self.config.detector.iou_threshold,
            device=self.config.detector.device,
        )
        match_thresh = getattr(self.config.tracker, "match_thresh", 0.80)
        self.tracker = tracker or UltralyticsTracker(
            tracker_type=self.config.tracker.type,
            track_buffer=self.config.tracker.track_buffer,
            match_thresh=match_thresh,
        )
        self.state_machine = state_machine

        # Handle events config vs state_machine config
        stitching_cfg = getattr(self.config, "events", None)
        reentry_s = 5.0
        max_dist = 150.0
        if stitching_cfg and hasattr(stitching_cfg, "stitching"):
            reentry_s = getattr(stitching_cfg.stitching, "max_time_gap_s", 5.0)
            max_dist = getattr(stitching_cfg.stitching, "max_distance_px", 150.0)

        self.stitcher = TrackletStitcher(
            stitch_window_s=reentry_s if reentry_s < 5.0 else 3.0,
            max_stitch_dist_px=max_dist if max_dist < 200.0 else 120.0,
            reentry_threshold_s=5.0,
        )

    def draw_roi_polygon(
        self,
        canvas: np.ndarray,
        roi: PolygonROI,
        is_occupied: bool,
    ) -> None:
        """Render translucent ground-plane polygon zone and boundary border."""
        pts = np.array(roi.pixel_vertices, dtype=np.int32).reshape((-1, 1, 2))
        overlay = canvas.copy()

        # Amber if occupied, emerald green if clear
        fill_color = (30, 140, 255) if is_occupied else (40, 180, 60)
        border_color = (0, 165, 255) if is_occupied else (0, 230, 100)

        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)
        cv2.polylines(canvas, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

        # Zone Label
        status_text = "OCCUPIED" if is_occupied else "CLEAR"
        zone_name = getattr(roi.config, "name", None) or getattr(roi.config, "roi_id", "Entrance")
        zone_label = f"ZONE: {zone_name} [{status_text}]"
        top_pt = min(roi.pixel_vertices, key=lambda p: p[1])
        cv2.putText(
            canvas,
            zone_label,
            (int(top_pt[0]) + 8, max(20, int(top_pt[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            border_color,
            1,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        total_frames: int,
        timestamp_s: float,
        active_in_zone: int,
        total_closed_visits: int,
        det_ms: float,
        trk_ms: float,
        fps: float,
    ) -> None:
        """Render upper HUD banner with real-time stream and tracking telemetry."""
        h, w = canvas.shape[:2]
        hud_h = 36
        hud_bg = canvas[:hud_h, :].copy()
        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (20, 20, 24), -1)
        cv2.addWeighted(hud_bg, 0.85, canvas[:hud_h, :], 0.15, 0, canvas[:hud_h, :])
        cv2.line(canvas, (0, hud_h), (w, hud_h), (60, 60, 70), 1)

        pct = (frame_idx + 1) / total_frames * 100.0 if total_frames > 0 else 0.0
        time_str = f"TIME: {timestamp_s:05.2f}s ({frame_idx+1:04d}/{total_frames} | {pct:3.0f}%)"
        occ_str = f"IN-ZONE: {active_in_zone} Subjects"
        visits_str = f"EVENTS: {total_closed_visits} Completed"
        perf_str = f"PERF: {fps:4.1f} FPS (Det:{det_ms:4.1f}ms Trk:{trk_ms:4.2f}ms)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        cv2.putText(canvas, time_str, (12, 23), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        cv2.putText(canvas, occ_str, (int(w * 0.33), 23), font, font_scale, (50, 215, 255) if active_in_zone > 0 else (180, 180, 180), thickness, cv2.LINE_AA)
        cv2.putText(canvas, visits_str, (int(w * 0.58), 23), font, font_scale, (100, 230, 120), thickness, cv2.LINE_AA)
        cv2.putText(canvas, perf_str, (int(w * 0.77), 23), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        tracklets: List[Tracklet],
        frame_idx: int,
        total_frames: int,
        timestamp_s: float,
        det_ms: float,
        trk_ms: float,
        fps: float,
    ) -> np.ndarray:
        """Render complete CCTV diagnostic labels on a single frame."""
        canvas = frame_bgr.copy()
        active_sessions = self.state_machine.get_active_sessions() if self.state_machine else []
        session_map = {s.track_id: s for s in active_sessions}
        is_occupied = len(active_sessions) > 0

        # 1. Render ROI Polygon
        if self.state_machine and hasattr(self.state_machine, "roi"):
            self.draw_roi_polygon(canvas, self.state_machine.roi, is_occupied)

        # 2. Render Tracklets with Ground-Plane Foot-Points and State Tags
        for trk in tracklets:
            x1, y1, x2, y2 = [int(v) for v in trk.box]
            color = get_color_for_id(trk.track_id)
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            # Motion trajectory history tail (last 30 frames)
            if len(trk.trajectory) > 1:
                pts = np.array(trk.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Bounding Box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Ground-Plane Foot-Point Crosshair (P_foot)
            cross_r = 7
            cv2.line(canvas, (foot_x - cross_r, foot_y), (foot_x + cross_r, foot_y), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (foot_x, foot_y - cross_r), (foot_x, foot_y + cross_r), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (foot_x, foot_y), 3, (255, 255, 255), -1)

            # State Tag & Real-Time Dwell Stopwatch
            session = session_map.get(trk.track_id)
            if session is not None:
                if session.state == VisitState.INSIDE:
                    tag_text = f"INSIDE ({session.duration_s:.1f}s)"
                    tag_bg = (30, 160, 40)
                elif session.state == VisitState.GRACE:
                    rem_grace = max(0.0, session.grace_period_s - (timestamp_s - session.grace_start_t))
                    tag_text = f"GRACE ({rem_grace:.1f}s)"
                    tag_bg = (0, 140, 255)
                else:
                    tag_text = session.state.value
                    tag_bg = (60, 60, 60)
                label = f"ID:{trk.track_id} | {tag_text} | {trk.confidence:.2f}"
            else:
                label = f"ID:{trk.track_id} | OUTSIDE | {trk.confidence:.2f}"
                tag_bg = (40, 40, 45)

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            tag_y1 = max(0, y1 - lh - 7)
            cv2.rectangle(canvas, (x1, tag_y1), (x1 + lw + 8, tag_y1 + lh + 7), tag_bg, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, tag_y1 + lh + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Upper Telemetry HUD
        self.draw_hud(
            canvas=canvas,
            frame_idx=frame_idx,
            total_frames=total_frames,
            timestamp_s=timestamp_s,
            active_in_zone=len(active_sessions),
            total_closed_visits=len(self.stitcher.committed_visits),
            det_ms=det_ms,
            trk_ms=trk_ms,
            fps=fps,
        )
        return canvas

    def process_video(
        self,
        video_path: Union[str, Path],
        output_video_path: Optional[Union[str, Path]] = None,
        roi_config_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Process video end-to-end, render labeled CCTV diagnostics, and return metrics.

        Args:
            video_path: Path to input video file.
            output_video_path: Optional destination path for annotated MP4 video.
            roi_config_path: Optional path to JSON ROI configuration.

        Returns:
            Dictionary containing processing telemetry, frame count, FPS, and visits.
        """
        reader = VideoStreamReader(video_path)
        roi_config = roi_config_path or Path("configs/roi_example.json")

        # Initialize ROI and StateMachine if not pre-configured
        if self.state_machine is None:
            roi_cfg = ROIConfig.from_file(roi_config)
            roi = PolygonROI(
                roi_cfg,
                frame_width=reader.width,
                frame_height=reader.height,
            )
            events_cfg = getattr(self.config, "events", None)
            grace_s = getattr(events_cfg, "grace_period_s", 2.0) if events_cfg else 2.0
            min_dwell = getattr(events_cfg, "min_dwell_s", 1.0) if events_cfg else 1.0
            enter_frames = getattr(events_cfg, "enter_consecutive_frames", 3) if events_cfg else 3
            exit_frames = getattr(events_cfg, "exit_consecutive_frames", 5) if events_cfg else 5

            self.state_machine = VisitStateMachine(
                roi=roi,
                camera_id="cam_cctv_01",
                grace_period_s=grace_s,
                min_dwell_s=min_dwell,
                enter_consecutive_frames=enter_frames,
            )

        writer: Optional[cv2.VideoWriter] = None
        if output_video_path:
            out_p = Path(output_video_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_p),
                fourcc,
                reader.fps,
                (reader.width, reader.height),
            )

        det_latencies: List[float] = []
        trk_latencies: List[float] = []
        total_detections = 0
        t_start = time.perf_counter()

        logger.info(
            "Processing CCTV video '%s' (%dx%d @ %.1f FPS, %d frames)...",
            video_path,
            reader.width,
            reader.height,
            reader.fps,
            reader.total_frames,
        )

        try:
            for frame_idx, timestamp_s, frame_bgr in reader:
                # 1. Detection
                t0_det = time.perf_counter()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = self.detector.detect(frame_rgb)
                det_latencies.append((time.perf_counter() - t0_det) * 1000.0)
                total_detections += len(detections)

                # 2. Tracking
                t0_trk = time.perf_counter()
                tracklets = self.tracker.update(detections, frame_rgb, frame_idx, timestamp_s)
                trk_latencies.append((time.perf_counter() - t0_trk) * 1000.0)

                # 3. State Machine Update
                closed = self.state_machine.update(
                    tracklets=tracklets,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    frame_rgb=frame_rgb,
                )
                for v in closed:
                    self.stitcher.process_visit(v)

                # 4. Render Annotations
                if writer is not None:
                    elapsed = time.perf_counter() - t_start
                    fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                    annotated = self.annotate_frame(
                        frame_bgr=frame_bgr,
                        tracklets=tracklets,
                        frame_idx=frame_idx,
                        total_frames=reader.total_frames,
                        timestamp_s=timestamp_s,
                        det_ms=det_latencies[-1],
                        trk_ms=trk_latencies[-1],
                        fps=fps,
                    )
                    writer.write(annotated)

            # 5. Flush remaining active sessions at stream conclusion (Failure Mode F7)
            stream_end_visits = self.state_machine.force_close_all(
                frame_idx=reader.total_frames - 1,
                timestamp_s=reader.duration_s,
                reason=CloseReason.STREAM_END,
            )
            for v in stream_end_visits:
                self.stitcher.process_visit(v)

        finally:
            if writer is not None:
                writer.release()

        total_wall_time = time.perf_counter() - t_start
        overall_fps = reader.total_frames / total_wall_time if total_wall_time > 0 else 0.0

        committed = self.stitcher.committed_visits
        alert_reduction = f"{total_detections / len(committed):.1f}:1" if committed else "N/A"

        summary = {
            "video_path": str(video_path),
            "output_video_path": str(output_video_path) if output_video_path else None,
            "total_frames": reader.total_frames,
            "native_fps": reader.fps,
            "resolution": [reader.width, reader.height],
            "wall_clock_time_s": round(total_wall_time, 3),
            "processing_fps": round(overall_fps, 2),
            "mean_det_latency_ms": round(float(np.mean(det_latencies)), 2) if det_latencies else 0.0,
            "mean_trk_latency_ms": round(float(np.mean(trk_latencies)), 2) if trk_latencies else 0.0,
            "total_raw_detections": total_detections,
            "total_emitted_visits": len(committed),
            "alert_reduction_ratio": alert_reduction,
            "visits": [v.model_dump() if hasattr(v, "model_dump") else dict(v) for v in committed],
        }

        logger.info(
            "CCTV video processing complete: %d frames processed in %.2fs (%.1f FPS). Collapsed %d raw detections into %d visits (%s reduction).",
            reader.total_frames,
            total_wall_time,
            overall_fps,
            total_detections,
            len(committed),
            alert_reduction,
        )
        return summary
