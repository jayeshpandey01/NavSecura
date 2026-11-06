"""Comparative Tracker Benchmarking Engine (ByteTrack vs BoT-SORT).

Executes multi-object tracking architectures sequentially across identical video streams,
evaluating association latency, processing FPS, ID stability/churn during crossing,
and occlusion survival.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from navsecura.detect import Detection, YOLODetector
from navsecura.io.config import AppConfig
from navsecura.io.video import VideoStreamReader
from navsecura.track import Tracklet, UltralyticsTracker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark_trackers")


def run_tracker_on_stream(
    tracker: UltralyticsTracker,
    video_path: Path,
    detector: YOLODetector,
    cached_detections: Optional[List[List[Detection]]] = None,
) -> Dict[str, Any]:
    """Run an individual tracker across the video stream and record granular telemetry."""
    reader = VideoStreamReader(video_path)
    tracker.reset()

    tracking_latencies: List[float] = []
    detection_latencies: List[float] = []
    active_tracks_per_frame: List[int] = []
    unique_ids_per_window: Dict[str, set] = {
        "pedestrian_crossing_f196_f235": set(),
        "converging_group_f352_f401": set(),
        "reentry_window_f532_f552": set(),
    }

    all_detections: List[List[Detection]] = []
    tracklets_by_frame: List[List[Tracklet]] = []

    t_wall_start = time.perf_counter()
    total_detections_count = 0

    with reader:
        for frame_idx, timestamp_s, frame_bgr in reader:
            # 1. Detection
            if cached_detections is not None and frame_idx < len(cached_detections):
                dets = cached_detections[frame_idx]
                det_ms = 0.0
            else:
                t_det0 = time.perf_counter()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                dets = detector.detect(frame_rgb)
                det_ms = (time.perf_counter() - t_det0) * 1000.0
                detection_latencies.append(det_ms)
                all_detections.append(dets)

            total_detections_count += len(dets)

            # 2. Tracking
            t_trk0 = time.perf_counter()
            tracklets = tracker.update(dets, frame_bgr, frame_idx, timestamp_s)
            trk_ms = (time.perf_counter() - t_trk0) * 1000.0
            tracking_latencies.append(trk_ms)
            active_tracks_per_frame.append(len(tracklets))
            tracklets_by_frame.append(tracklets)

            # Critical scenario window tracking
            for trk in tracklets:
                if 196 <= frame_idx <= 235:
                    unique_ids_per_window["pedestrian_crossing_f196_f235"].add(trk.track_id)
                elif 352 <= frame_idx <= 401:
                    unique_ids_per_window["converging_group_f352_f401"].add(trk.track_id)
                elif 532 <= frame_idx <= 552:
                    unique_ids_per_window["reentry_window_f532_f552"].add(trk.track_id)

    t_wall_elapsed = time.perf_counter() - t_wall_start
    overall_fps = reader.total_frames / t_wall_elapsed if t_wall_elapsed > 0 else 0.0

    return {
        "tracker_name": tracker.name,
        "tracker_type": tracker.tracker_type,
        "total_frames": reader.total_frames,
        "wall_time_s": round(t_wall_elapsed, 3),
        "overall_fps": round(overall_fps, 1),
        "avg_detector_latency_ms": round(float(np.mean(detection_latencies)), 2) if detection_latencies else 0.0,
        "avg_tracker_latency_ms": round(float(np.mean(tracking_latencies)), 2),
        "median_tracker_latency_ms": round(float(np.median(tracking_latencies)), 2),
        "p95_tracker_latency_ms": round(float(np.percentile(tracking_latencies, 95)), 2),
        "total_unique_tracks": tracker.total_unique_tracks,
        "unique_track_ids": sorted(list(tracker.unique_track_ids)),
        "crossing_unique_ids": sorted(list(unique_ids_per_window["pedestrian_crossing_f196_f235"])),
        "crossing_id_count": len(unique_ids_per_window["pedestrian_crossing_f196_f235"]),
        "converging_unique_ids": sorted(list(unique_ids_per_window["converging_group_f352_f401"])),
        "converging_id_count": len(unique_ids_per_window["converging_group_f352_f401"]),
        "reentry_unique_ids": sorted(list(unique_ids_per_window["reentry_window_f532_f552"])),
        "cached_detections": all_detections if cached_detections is None else None,
        "tracklets_by_frame": tracklets_by_frame,
    }


def render_comparison_snapshots(
    video_path: Path,
    bytetrack_runs: List[List[Tracklet]],
    botsort_runs: List[List[Tracklet]],
    output_dir: Path,
    target_frames: List[int] = [215, 380, 540],
) -> List[str]:
    """Render side-by-side diagnostic snapshots showing ByteTrack vs BoT-SORT tracks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: List[str] = []

    reader = VideoStreamReader(video_path)
    with reader:
        for frame_idx, timestamp_s, frame_bgr in reader:
            if frame_idx in target_frames:
                # Render ByteTrack view
                byte_canvas = frame_bgr.copy()
                if frame_idx < len(bytetrack_runs):
                    for t in bytetrack_runs[frame_idx]:
                        x1, y1, x2, y2 = [int(v) for v in t.box]
                        cv2.rectangle(byte_canvas, (x1, y1), (x2, y2), (230, 92, 43), 2)
                        cv2.putText(
                            byte_canvas,
                            f"ByteTrack ID:{t.track_id}",
                            (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (230, 92, 43),
                            1,
                            cv2.LINE_AA,
                        )

                # Render BoT-SORT view
                bot_canvas = frame_bgr.copy()
                if frame_idx < len(botsort_runs):
                    for t in botsort_runs[frame_idx]:
                        x1, y1, x2, y2 = [int(v) for v in t.box]
                        cv2.rectangle(bot_canvas, (x1, y1), (x2, y2), (2, 95, 217), 2)
                        cv2.putText(
                            bot_canvas,
                            f"BoT-SORT ID:{t.track_id}",
                            (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (2, 95, 217),
                            1,
                            cv2.LINE_AA,
                        )

                # Combine side by side
                combined = np.hstack([byte_canvas, bot_canvas])
                # Overlay banner
                h_comb, w_comb = combined.shape[:2]
                banner = np.zeros((32, w_comb, 3), dtype=np.uint8)
                cv2.putText(
                    banner,
                    f"Frame {frame_idx:04d} ({timestamp_s:.2f}s) | LEFT: ByteTrack (2-Stage Hungarian) | RIGHT: BoT-SORT (GMC Motion Compensated)",
                    (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                final_img = np.vstack([banner, combined])

                out_path = output_dir / f"compare_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(out_path), final_img)
                snapshot_paths.append(str(out_path))
                logger.info("Saved comparison snapshot: %s", out_path)

    return snapshot_paths


def main() -> None:
    """Execute side-by-side benchmark comparing ByteTrack and BoT-SORT."""
    parser = argparse.ArgumentParser(description="NavSecura Side-by-Side Tracker Benchmark")
    parser.add_argument("--video", type=str, default="data/sample.mp4", help="Video file path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--output", type=str, default="outputs/telemetry/tracker_comparison_meta.json", help="Telemetry output path")
    parser.add_argument("--snapshot-dir", type=str, default="outputs/figures/tracker_comparison", help="Snapshot output dir")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    app_config = AppConfig.load(args.config)
    detector = YOLODetector(
        weights=app_config.detector.weights,
        confidence_threshold=app_config.detector.confidence_threshold,
        iou_threshold=app_config.detector.iou_threshold,
        device=app_config.detector.device,
        classes=app_config.detector.classes,
    )

    logger.info("=" * 80)
    logger.info("STARTING COMPARATIVE TRACKER BENCHMARK (ByteTrack vs BoT-SORT)")
    logger.info("=" * 80)

    # 1. Run ByteTrack
    logger.info(">> Running Candidate 1: ByteTrack (track_buffer=%d)", app_config.tracker.track_buffer)
    bytetrack = UltralyticsTracker(
        tracker_type="bytetrack",
        track_high_thresh=app_config.tracker.track_high_thresh,
        track_low_thresh=app_config.tracker.track_low_thresh,
        new_track_thresh=app_config.tracker.new_track_thresh,
        track_buffer=app_config.tracker.track_buffer,
        match_thresh=app_config.tracker.match_thresh,
    )
    byte_results = run_tracker_on_stream(bytetrack, video_path, detector)
    cached_dets = byte_results.pop("cached_detections")
    byte_tracklets = byte_results.pop("tracklets_by_frame")

    # 2. Run BoT-SORT with GMC (sparseOptFlow)
    logger.info(">> Running Candidate 2: BoT-SORT (GMC=sparseOptFlow, track_buffer=%d)", app_config.tracker.track_buffer)
    botsort_gmc = UltralyticsTracker(
        tracker_type="botsort",
        track_high_thresh=app_config.tracker.track_high_thresh,
        track_low_thresh=app_config.tracker.track_low_thresh,
        new_track_thresh=app_config.tracker.new_track_thresh,
        track_buffer=app_config.tracker.track_buffer,
        match_thresh=app_config.tracker.match_thresh,
        gmc_method="sparseOptFlow",
        with_reid=False,
    )
    bot_results = run_tracker_on_stream(botsort_gmc, video_path, detector, cached_detections=cached_dets)
    bot_tracklets = bot_results.pop("tracklets_by_frame")
    if "cached_detections" in bot_results:
        bot_results.pop("cached_detections")

    # 3. Render side-by-side comparison snapshots for crossing and occlusion
    snapshot_dir = Path(args.snapshot_dir)
    snapshots = render_comparison_snapshots(
        video_path=video_path,
        bytetrack_runs=byte_tracklets,
        botsort_runs=bot_tracklets,
        output_dir=snapshot_dir,
        target_frames=[215, 380, 540],
    )

    # 4. Synthesize comparative telemetry
    comparison_meta = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_source": str(video_path),
            "detector_architecture": detector.name,
            "total_frames_evaluated": byte_results["total_frames"],
        },
        "bytetrack_benchmark": byte_results,
        "botsort_benchmark": bot_results,
        "side_by_side_comparison": {
            "latency_ratio_botsort_to_bytetrack": round(
                bot_results["avg_tracker_latency_ms"] / byte_results["avg_tracker_latency_ms"], 2
            )
            if byte_results["avg_tracker_latency_ms"] > 0
            else 1.0,
            "fps_ratio_bytetrack_to_botsort": round(
                byte_results["overall_fps"] / bot_results["overall_fps"], 2
            )
            if bot_results["overall_fps"] > 0
            else 1.0,
            "id_switch_stability_crossing": {
                "ground_truth_target_count": 2,
                "bytetrack_unique_ids": byte_results["crossing_unique_ids"],
                "botsort_unique_ids": bot_results["crossing_unique_ids"],
                "bytetrack_passed": byte_results["crossing_id_count"] == 2,
                "botsort_passed": bot_results["crossing_id_count"] == 2,
            },
            "snapshots_generated": snapshots,
        },
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_meta, f, indent=2)
    logger.info("Saved comparative benchmark telemetry to %s", out_file)

    # Print comparative Markdown table
    print("\n" + "=" * 80)
    print("COMPARATIVE MULTI-OBJECT TRACKING BENCHMARK: ByteTrack vs BoT-SORT")
    print("=" * 80)
    print(f"{'Metric':<36} | {'ByteTrack':<18} | {'BoT-SORT':<18}")
    print("-" * 80)
    print(f"{'Tracker Algorithm':<36} | {byte_results['tracker_name']:<18} | {bot_results['tracker_name']:<18}")
    print(f"{'Average Tracker Latency':<36} | {byte_results['avg_tracker_latency_ms']:>14.2f} ms | {bot_results['avg_tracker_latency_ms']:>14.2f} ms")
    print(f"{'Median Tracker Latency':<36} | {byte_results['median_tracker_latency_ms']:>14.2f} ms | {bot_results['median_tracker_latency_ms']:>14.2f} ms")
    print(f"{'95th Percentile Latency':<36} | {byte_results['p95_tracker_latency_ms']:>14.2f} ms | {bot_results['p95_tracker_latency_ms']:>14.2f} ms")
    print(f"{'Overall Processing Throughput':<36} | {byte_results['overall_fps']:>14.1f} FPS| {bot_results['overall_fps']:>14.1f} FPS")
    print(f"{'Total Unique Track IDs':<36} | {byte_results['total_unique_tracks']:>18d} | {bot_results['total_unique_tracks']:>18d}")
    print(f"{'Pedestrian Crossing IDs (GT=2)':<36} | {str(byte_results['crossing_unique_ids']):>18} | {str(bot_results['crossing_unique_ids']):>18}")
    print(f"{'Converging Group IDs (GT=3)':<36} | {str(byte_results['converging_unique_ids']):>18} | {str(bot_results['converging_unique_ids']):>18}")
    print(f"{'Re-entry Window IDs (GT=1)':<36} | {str(byte_results['reentry_unique_ids']):>18} | {str(bot_results['reentry_unique_ids']):>18}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

