"""Pipeline execution tool for detection and tracking baseline.

Processes video frames through Detector and Tracker protocols, records
granular latency telemetry, and produces annotated diagnostic video output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from navsecura.detect import Detection, Detector, YOLODetector
from navsecura.io.config import AppConfig
from navsecura.io.video import VideoStreamReader
from navsecura.track import Tracker, Tracklet, UltralyticsTracker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_tracker_baseline")

# Distinct color palette for consistent track ID visualization (BGR format)
PALETTE: List[Tuple[int, int, int]] = [
    (230, 92, 43),    # Blue-teal
    (2, 95, 217),     # Orange
    (179, 112, 117),  # Slate
    (138, 41, 231),   # Magenta
    (119, 158, 27),   # Green
    (31, 199, 237),   # Gold
    (180, 48, 153),   # Violet
    (54, 67, 244),    # Coral
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministically assign a distinct BGR color to a track ID."""
    return PALETTE[track_id % len(PALETTE)]


def draw_hud(
    canvas: np.ndarray,
    frame_idx: int,
    total_frames: int,
    timestamp_s: float,
    active_tracks: int,
    unique_tracks: int,
    det_ms: float,
    trk_ms: float,
    fps: float,
) -> None:
    """Draw a semi-transparent diagnostic telemetry HUD at the top of the frame."""
    hud_h = 36
    canvas_w = canvas.shape[1]

    # Semi-transparent overlay banner
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_w, hud_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.line(canvas, (0, hud_h), (canvas_w, hud_h), (60, 60, 60), 1)

    hud_text = (
        f"Frame: {frame_idx:04d}/{total_frames:04d} | Time: {timestamp_s:6.2f}s | "
        f"Active Tracks: {active_tracks:2d} | Unique IDs: {unique_tracks:2d} | "
        f"Det: {det_ms:5.1f}ms | Trk: {trk_ms:4.1f}ms | Pipeline: {fps:5.1f} FPS"
    )

    cv2.putText(
        canvas,
        hud_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame: np.ndarray,
    tracklets: List[Tracklet],
    frame_idx: int,
    total_frames: int,
    timestamp_s: float,
    unique_tracks_count: int,
    det_ms: float,
    trk_ms: float,
    inst_fps: float,
) -> np.ndarray:
    """Render bounding boxes, foot points, trajectories, and HUD telemetry onto the frame."""
    canvas = frame.copy()

    # Draw trajectories and track bounding boxes
    for t in tracklets:
        color = get_color_for_id(t.track_id)
        x1, y1, x2, y2 = [int(v) for v in t.box]

        # Draw trajectory tail connecting historical ground contact points
        if len(t.trajectory) >= 2:
            pts = np.array(t.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw foot-point ground contact marker
        fx, fy = int(t.foot_point[0]), int(t.foot_point[1])
        cv2.circle(canvas, (fx, fy), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (fx, fy), 6, color, 1, cv2.LINE_AA)

        # Draw label header
        label = f"ID: {t.track_id} ({t.confidence:.2f})"
        font_scale = 0.45
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        label_y1 = max(0, y1 - label_h - baseline - 4)
        label_y2 = y1
        cv2.rectangle(canvas, (x1, label_y1), (x1 + label_w + 6, label_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, label_y2 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Render HUD
    draw_hud(
        canvas=canvas,
        frame_idx=frame_idx,
        total_frames=total_frames,
        timestamp_s=timestamp_s,
        active_tracks=len(tracklets),
        unique_tracks=unique_tracks_count,
        det_ms=det_ms,
        trk_ms=trk_ms,
        fps=inst_fps,
    )

    return canvas


def run_pipeline(
    video_source: str,
    output_video_path: Optional[Path] = None,
    output_meta_path: Optional[Path] = None,
    config_path: str = "configs/default.yaml",
    max_frames: Optional[int] = None,
    device_override: Optional[str] = None,
    weights_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the detection and tracking baseline pipeline end-to-end.

    Args:
        video_source: Path to target video file or scenario alias.
        output_video_path: Optional path for saving annotated diagnostic MP4 video.
        output_meta_path: Optional path for saving structured JSON telemetry.
        config_path: Path to configuration YAML.
        max_frames: Optional maximum number of frames to process.
        device_override: Optional device override ('cpu', 'mps', 'cuda').
        weights_override: Optional model weights override.

    Returns:
        Structured performance and telemetry report dictionary.
    """
    overrides: Dict[str, Any] = {"video.source": video_source}
    if device_override:
        overrides["detector.device"] = device_override
    if weights_override:
        overrides["detector.weights"] = weights_override

    cfg = AppConfig.load(config_path, overrides=overrides)
    resolved_source = cfg.video.source

    logger.info("Initializing VideoStreamReader for: %s", resolved_source)
    reader = VideoStreamReader(resolved_source, target_fps=cfg.video.target_fps)

    # Adapt parameters dynamically to stream resolution and FPS
    cfg.adapt_to_stream(reader.native_fps, reader.width, reader.height)

    # Initialize Detector and Tracker
    logger.info("Initializing Detector (%s)...", cfg.detector.model_type)
    detector = YOLODetector(
        weights=cfg.detector.weights,
        confidence_threshold=cfg.detector.confidence_threshold,
        iou_threshold=cfg.detector.iou_threshold,
        classes=cfg.detector.classes,
        device=cfg.detector.device,
    )

    logger.info("Initializing Tracker (%s)...", cfg.tracker.type)
    tracker = UltralyticsTracker(
        tracker_type=cfg.tracker.type,
        track_high_thresh=cfg.tracker.track_high_thresh,
        track_low_thresh=cfg.tracker.track_low_thresh,
        new_track_thresh=cfg.tracker.new_track_thresh,
        match_thresh=cfg.tracker.match_thresh,
        track_buffer=cfg.tracker.track_buffer,
    )

    writer: Optional[cv2.VideoWriter] = None
    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            reader.effective_fps,
            (reader.width, reader.height),
        )
        logger.info("Writing annotated output to: %s", output_video_path)

    total_frames_processed = 0
    total_detections_count = 0
    confidence_sum = 0.0

    inference_latencies: List[float] = []
    tracking_latencies: List[float] = []
    frame_latencies: List[float] = []

    wall_start = time.perf_counter()

    try:
        for frame_idx, timestamp_s, frame in reader:
            if max_frames is not None and total_frames_processed >= max_frames:
                logger.info("Reached max_frames limit (%d). Terminating loop.", max_frames)
                break

            t_frame_start = time.perf_counter()

            # 1. Detection Stage
            detections = detector.detect(frame)
            det_ms = detector.last_inference_ms
            inference_latencies.append(det_ms)
            total_detections_count += len(detections)
            for d in detections:
                confidence_sum += d.confidence

            # 2. Tracking Stage
            tracklets = tracker.update(detections, frame, frame_idx, timestamp_s)
            trk_ms = tracker.last_update_ms
            tracking_latencies.append(trk_ms)

            # Compute timing & FPS
            t_frame_end = time.perf_counter()
            frame_elapsed = t_frame_end - t_frame_start
            frame_latencies.append(frame_elapsed * 1000.0)
            inst_fps = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0

            # 3. Visualization & Output Writing
            if writer is not None:
                annotated = annotate_frame(
                    frame=frame,
                    tracklets=tracklets,
                    frame_idx=frame_idx,
                    total_frames=reader.total_frames,
                    timestamp_s=timestamp_s,
                    unique_tracks_count=tracker.total_unique_tracks,
                    det_ms=det_ms,
                    trk_ms=trk_ms,
                    inst_fps=inst_fps,
                )
                writer.write(annotated)

            total_frames_processed += 1
            if total_frames_processed % 50 == 0 or total_frames_processed == reader.total_frames:
                logger.info(
                    "Processed frame %d/%d (%.1f%%) | Active: %d | Det: %.1fms | Trk: %.1fms | Pipeline: %.1f FPS",
                    frame_idx,
                    reader.total_frames,
                    (frame_idx / reader.total_frames) * 100.0 if reader.total_frames else 0.0,
                    len(tracklets),
                    det_ms,
                    trk_ms,
                    inst_fps,
                )

    finally:
        reader.close()
        if writer is not None:
            writer.release()
            logger.info("Released video writer.")

    wall_duration = time.perf_counter() - wall_start
    overall_fps = total_frames_processed / wall_duration if wall_duration > 0 else 0.0
    mean_inf_ms = float(np.mean(inference_latencies)) if inference_latencies else 0.0
    mean_trk_ms = float(np.mean(tracking_latencies)) if tracking_latencies else 0.0
    mean_conf = (confidence_sum / total_detections_count) if total_detections_count > 0 else 0.0

    telemetry: Dict[str, Any] = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_source": str(resolved_source),
            "camera_id": cfg.video.camera_id,
            "resolution": [reader.width, reader.height],
            "native_fps": reader.native_fps,
            "effective_fps": reader.effective_fps,
            "device": cfg.detector.device,
            "model_architecture": detector.name,
            "tracker_architecture": tracker.name,
        },
        "performance_metrics": {
            "total_frames_processed": total_frames_processed,
            "total_video_duration_s": round(total_frames_processed / reader.native_fps, 3),
            "wall_clock_time_s": round(wall_duration, 3),
            "overall_processing_fps": round(overall_fps, 2),
            "average_detector_latency_ms": round(mean_inf_ms, 2),
            "average_tracker_latency_ms": round(mean_trk_ms, 2),
            "average_frame_latency_ms": round(float(np.mean(frame_latencies)), 2) if frame_latencies else 0.0,
        },
        "perception_metrics": {
            "total_raw_person_detections": total_detections_count,
            "mean_detection_confidence": round(mean_conf, 4),
            "total_unique_track_ids": tracker.total_unique_tracks,
            "unique_track_ids": sorted(list(tracker.unique_track_ids)),
        },
    }

    logger.info(
        "=== Baseline Execution Complete ===\n"
        "  Processed: %d frames in %.2fs (%.1f FPS)\n"
        "  Avg Detector Latency: %.2f ms\n"
        "  Avg Tracker Latency:  %.2f ms\n"
        "  Unique Tracks Found:  %d\n"
        "  Total Detections:     %d (Mean Conf: %.3f)",
        total_frames_processed,
        wall_duration,
        overall_fps,
        mean_inf_ms,
        mean_trk_ms,
        tracker.total_unique_tracks,
        total_detections_count,
        mean_conf,
    )

    if output_meta_path is not None:
        output_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_meta_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        logger.info("Saved telemetry to: %s", output_meta_path)

    return telemetry


def main() -> int:
    """CLI entrypoint for baseline tracking runner."""
    parser = argparse.ArgumentParser(
        description="Run perception detection and multi-object tracking baseline on a video stream."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/sample.mp4",
        help="Path to input video file or scenario alias (concourse, shop, office, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/videos/sample_tracking_baseline.mp4"),
        help="Path for saving annotated diagnostic MP4 video",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("outputs/telemetry/tracking_baseline_meta.json"),
        help="Path for saving telemetry benchmark JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Accelerator device override ('cpu', 'mps', 'cuda')",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights file",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            video_source=args.video,
            output_video_path=args.output,
            output_meta_path=args.meta,
            config_path=args.config,
            max_frames=args.max_frames,
            device_override=args.device,
            weights_override=args.weights,
        )
        return 0
    except Exception as exc:
        logger.error("Pipeline baseline run failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

