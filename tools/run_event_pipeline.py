"""Comprehensive End-to-End Visit Event Pipeline and Verification Runner.

Integrates VideoStreamReader -> YOLODetector -> UltralyticsTracker (ByteTrack)
-> PolygonROI -> VisitStateMachine -> TrackletStitcher.

Renders high-fidelity annotated video with ROI occupancy, tracklet trajectories,
state machine transitions, and produces visit event records with ground-truth evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from navsecura.detect import YOLODetector
from navsecura.events import (
    CloseReason,
    TrackletStitcher,
    Visit,
    VisitState,
    VisitStateMachine,
)
from navsecura.evidence import CircularVideoBuffer, EvidenceWriter
from navsecura.io.config import AppConfig
from navsecura.io.video import VideoStreamReader
from navsecura.roi.geometry import PolygonROI, ROIConfig
from navsecura.store import SQLiteEventStore
from navsecura.track import Tracklet, UltralyticsTracker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_event_pipeline")

# Color palette for tracks (BGR)
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


def draw_roi_polygon(
    canvas: np.ndarray,
    roi: PolygonROI,
    is_occupied: bool,
) -> None:
    """Render semi-transparent filled polygon and contrasting boundary line."""
    pts = np.array(roi.pixel_vertices, dtype=np.int32).reshape((-1, 1, 2))
    overlay = canvas.copy()

    # Green when unoccupied, Amber/Red when occupied
    fill_color = (40, 160, 240) if is_occupied else (40, 180, 50)
    line_color = (0, 215, 255) if is_occupied else (0, 255, 100)

    cv2.fillPoly(overlay, [pts], fill_color)
    cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
    cv2.polylines(canvas, [pts], isClosed=True, color=line_color, thickness=2, lineType=cv2.LINE_AA)

    # Zone label
    top_pt = min(roi.pixel_vertices, key=lambda p: p[1])
    label_text = f"ROI: {roi.config.roi_id} {'[OCCUPIED]' if is_occupied else '[CLEAR]'}"
    cv2.putText(
        canvas,
        label_text,
        (int(top_pt[0]) + 8, int(top_pt[1]) + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        line_color,
        2,
        cv2.LINE_AA,
    )


def draw_hud(
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
    """Draw telemetry HUD at top of frame."""
    hud_h = 38
    canvas_w = canvas.shape[1]

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas_w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
    cv2.line(canvas, (0, hud_h), (canvas_w, hud_h), (60, 60, 60), 1)

    hud_text = (
        f"Frame: {frame_idx:04d}/{total_frames:04d} | Time: {timestamp_s:5.2f}s | "
        f"In ROI: {active_in_zone:2d} | Emitted Visits: {total_closed_visits:2d} | "
        f"Det: {det_ms:4.1f}ms | Trk: {trk_ms:3.1f}ms | FPS: {fps:4.1f}"
    )

    cv2.putText(
        canvas,
        hud_text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    tracklets: List[Tracklet],
    state_machine: VisitStateMachine,
    stitcher: TrackletStitcher,
    frame_idx: int,
    total_frames: int,
    timestamp_s: float,
    det_ms: float,
    trk_ms: float,
    fps: float,
) -> np.ndarray:
    """Render full annotations including detections, tracking, ROI, and visit state."""
    canvas = frame_bgr.copy()
    active_sessions = state_machine.get_active_sessions()
    is_occupied = len(active_sessions) > 0

    # Draw ROI polygon
    draw_roi_polygon(canvas, state_machine.roi, is_occupied)

    # Draw active tracks
    session_map = {s.track_id: s for s in active_sessions}

    for trk in tracklets:
        x1, y1, x2, y2 = [int(v) for v in trk.box]
        color = get_color_for_id(trk.track_id)
        foot_x, foot_y = int((x1 + x2) / 2), int(y2)

        # Draw trajectory tail
        if len(trk.trajectory) > 1:
            pts = np.array(trk.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Foot-point ground plane marker
        cv2.circle(canvas, (foot_x, foot_y), 5, (0, 0, 255), -1)
        cv2.circle(canvas, (foot_x, foot_y), 6, (255, 255, 255), 1)

        # Label tag with visit state
        session = session_map.get(trk.track_id)
        if session is not None:
            if session.state == VisitState.INSIDE:
                state_tag = f"INSIDE ({session.duration_s:.1f}s)"
                tag_bg = (30, 160, 40)
            elif session.state == VisitState.GRACE:
                rem_grace = max(0.0, session.grace_period_s - (timestamp_s - session.grace_start_t))
                state_tag = f"GRACE ({rem_grace:.1f}s)"
                tag_bg = (0, 140, 255)
            else:
                state_tag = session.state.value
                tag_bg = (60, 60, 60)
            label = f"ID:{trk.track_id} {state_tag} {trk.confidence:.2f}"
        else:
            label = f"ID:{trk.track_id} {trk.confidence:.2f}"
            tag_bg = color

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, y1 - lh - 6)
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + lw + 6, tag_y1 + lh + 6), tag_bg, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, tag_y1 + lh + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw HUD
    draw_hud(
        canvas=canvas,
        frame_idx=frame_idx,
        total_frames=total_frames,
        timestamp_s=timestamp_s,
        active_in_zone=len(active_sessions),
        total_closed_visits=len(stitcher.committed_visits),
        det_ms=det_ms,
        trk_ms=trk_ms,
        fps=fps,
    )
    return canvas


def evaluate_against_ground_truth(
    predicted_visits: List[Visit],
    ground_truth_path: Path,
    roi_id: str,
) -> Dict[str, Any]:
    """Compute visit-level evaluation metrics against ground truth CSV."""
    if not ground_truth_path.exists():
        logger.warning("Ground truth file %s not found; skipping evaluation", ground_truth_path)
        return {}

    gt_rows: List[Dict[str, Any]] = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("roi_id") == roi_id:
                gt_rows.append(
                    {
                        "visit_id": r["visit_id"],
                        "person_id": r["person_id"],
                        "frame_enter": int(r["frame_enter"]),
                        "frame_exit": int(r["frame_exit"]),
                        "t_enter_s": float(r["t_enter_s"]),
                        "t_exit_s": float(r["t_exit_s"]),
                        "duration_s": float(r["duration_s"]),
                    }
                )

    n_gt = len(gt_rows)
    n_pred = len(predicted_visits)

    if n_gt == 0:
        return {"n_gt": 0, "n_pred": n_pred}

    # Bipartite matching based on temporal overlap
    matched_gt = set()
    matched_pred = set()
    dwell_errors: List[float] = []

    for p_idx, pred in enumerate(predicted_visits):
        best_gt_idx = -1
        best_iou = 0.0
        for g_idx, gt in enumerate(gt_rows):
            if g_idx in matched_gt:
                continue
            # Compute temporal IoU
            start = max(pred.video_t_enter_s, gt["t_enter_s"])
            end = min(pred.video_t_exit_s, gt["t_exit_s"])
            inter = max(0.0, end - start)
            union = (pred.video_t_exit_s - pred.video_t_enter_s) + gt["duration_s"] - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou and iou >= 0.25:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            gt_dur = gt_rows[best_gt_idx]["duration_s"]
            dwell_errors.append(abs(pred.duration_s - gt_dur))

    n_matched = len(matched_gt)
    precision = n_matched / float(n_pred) if n_pred > 0 else 0.0
    recall = n_matched / float(n_gt) if n_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_dwell_mae = float(np.mean(dwell_errors)) if dwell_errors else 0.0

    return {
        "ground_truth_visits": n_gt,
        "predicted_visits": n_pred,
        "matched_visits": n_matched,
        "visit_precision": round(precision, 4),
        "visit_recall": round(recall, 4),
        "visit_f1_score": round(f1, 4),
        "fragmentation_rate": round(max(0, n_pred - n_gt) / float(n_gt), 4),
        "mean_dwell_error_mae_s": round(mean_dwell_mae, 3),
    }


def main() -> None:
    """Execute end-to-end event pipeline with full telemetry and evaluation."""
    parser = argparse.ArgumentParser(description="NavSecura End-to-End Visit Event Pipeline")
    parser.add_argument("--video", type=str, default="data/sample.mp4", help="Input video filepath")
    parser.add_argument("--roi-config", type=str, default="configs/roi_example.json", help="ROI configuration file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Application YAML configuration")
    parser.add_argument("--output-video", type=str, default="outputs/sample_event_pipeline.mp4", help="Annotated video path")
    parser.add_argument("--output-json", type=str, default="outputs/sample_visits.json", help="Emitted visits JSON path")
    parser.add_argument("--output-csv", type=str, default="outputs/sample_visits.csv", help="Emitted visits CSV path")
    parser.add_argument("--meta", type=str, default="outputs/sample_event_pipeline_meta.json", help="Pipeline telemetry JSON")
    parser.add_argument("--ground-truth", type=str, default="eval/ground_truth.csv", help="Ground truth CSV for evaluation")
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Multi-object tracker algorithm")
    parser.add_argument("--output-db", type=str, default="outputs/events.db", help="SQLite database path for visit events")
    parser.add_argument("--events-json", type=str, default="outputs/events.json", help="Export path for finalized events JSON")
    parser.add_argument("--events-csv", type=str, default="outputs/events.csv", help="Export path for finalized events CSV")
    parser.add_argument("--save-evidence", action="store_true", default=True, help="Generate forensic snapshots and video clips")
    parser.add_argument("--no-evidence", action="store_false", dest="save_evidence", help="Disable evidence generation")
    parser.add_argument("--pre-roll", type=float, default=3.0, help="Pre-roll seconds for clip extraction")
    parser.add_argument("--post-roll", type=float, default=3.0, help="Post-roll seconds for clip extraction")
    args = parser.parse_args()
    run_pipeline(args)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute end-to-end event pipeline with full telemetry, evidence generation, and persistence."""
    app_config = AppConfig.load(args.config)
    video_path = Path(args.video)
    roi_config_path = Path(args.roi_config)

    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    logger.info("Initializing VideoStreamReader: %s", video_path)
    reader = VideoStreamReader(video_path)

    # Dynamic adaptation to stream properties
    app_config.adapt_to_stream(native_fps=reader.fps, width=reader.width, height=reader.height)

    # Initialize dynamic ROI
    roi_config = ROIConfig.from_file(roi_config_path)
    polygon_roi = PolygonROI(roi_config, frame_width=reader.width, frame_height=reader.height)
    logger.info(
        "Loaded ROI '%s' with %d vertices on canvas %dx%d",
        polygon_roi.config.roi_id,
        len(polygon_roi.pixel_vertices),
        reader.width,
        reader.height,
    )

    # Initialize Detector & Tracker
    logger.info("Initializing YOLODetector (weights=%s, conf=%.2f)", app_config.detector.weights, app_config.detector.confidence_threshold)
    detector = YOLODetector(
        weights=app_config.detector.weights,
        confidence_threshold=app_config.detector.confidence_threshold,
        iou_threshold=app_config.detector.iou_threshold,
        device=app_config.detector.device,
        classes=app_config.detector.classes,
    )

    logger.info("Initializing UltralyticsTracker (%s, buf=%d)", args.tracker, app_config.tracker.track_buffer)
    tracker = UltralyticsTracker(
        tracker_type=args.tracker,
        track_high_thresh=app_config.tracker.track_high_thresh,
        track_low_thresh=app_config.tracker.track_low_thresh,
        new_track_thresh=app_config.tracker.new_track_thresh,
        track_buffer=app_config.tracker.track_buffer,
        match_thresh=app_config.tracker.match_thresh,
        gmc_method=app_config.tracker.gmc_method,
    )

    # Initialize Visit State Machine & Stitcher
    state_machine = VisitStateMachine(
        roi=polygon_roi,
        camera_id=polygon_roi.config.camera_id or "cam_entrance_01",
        enter_consecutive_frames=app_config.events.enter_consecutive_frames,
        grace_period_s=app_config.events.grace_period_s,
        min_dwell_s=app_config.events.min_dwell_s,
    )

    stitcher = TrackletStitcher(
        stitch_window_s=app_config.events.stitching.max_time_gap_s,
        max_stitch_dist_px=app_config.events.stitching.max_distance_px,
        reentry_threshold_s=5.0,
        max_reentry_window_s=120.0,
        min_appearance_sim=0.80,
    )

    # Video writer setup
    out_video_path = Path(args.output_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_video_path),
        fourcc,
        reader.fps,
        (reader.width, reader.height),
    )

    det_latencies: List[float] = []
    trk_latencies: List[float] = []
    total_detections = 0
    t_start = time.perf_counter()

    # Evidence & persistence infrastructure
    video_buffer = CircularVideoBuffer(
        capacity_frames=int(max(60.0, reader.duration_s + 10.0) * reader.fps),
        fps=reader.fps,
    )
    evidence_writer = EvidenceWriter(output_dir="outputs")
    event_store = SQLiteEventStore(args.output_db)

    last_frame_idx = 0
    last_timestamp_s = 0.0

    try:
        with reader:
            for frame_idx, timestamp_s, frame_bgr in reader:
                last_frame_idx = frame_idx
                last_timestamp_s = timestamp_s

                # Ingest into rolling circular video buffer
                video_buffer.append(frame_idx, timestamp_s, frame_bgr)

                # 1. Detection
                t_det0 = time.perf_counter()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = detector.detect(frame_rgb)
                det_latencies.append((time.perf_counter() - t_det0) * 1000.0)
                total_detections += len(detections)

                # 2. Tracking
                t_trk0 = time.perf_counter()
                tracklets = tracker.update(detections, frame_rgb, frame_idx, timestamp_s)
                trk_latencies.append((time.perf_counter() - t_trk0) * 1000.0)

                # 3. State Machine Update
                newly_closed = state_machine.update(
                    tracklets=tracklets,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    frame_rgb=frame_rgb,
                )

                # 4. Spatio-temporal stitching & re-entry linking
                for visit in newly_closed:
                    stitcher.process_visit(visit)

                # 5. Annotation & HUD
                elapsed = time.perf_counter() - t_start
                inst_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                annotated = annotate_frame(
                    frame_bgr=frame_bgr,
                    tracklets=tracklets,
                    state_machine=state_machine,
                    stitcher=stitcher,
                    frame_idx=frame_idx,
                    total_frames=reader.total_frames,
                    timestamp_s=timestamp_s,
                    det_ms=det_latencies[-1],
                    trk_ms=trk_latencies[-1],
                    fps=inst_fps,
                )
                writer.write(annotated)

        # 6. Stream End Force-Close (Failure Mode F7)
        stream_end_visits = state_machine.force_close_all(
            frame_idx=last_frame_idx,
            timestamp_s=last_timestamp_s,
            reason=CloseReason.STREAM_END,
        )
        for visit in stream_end_visits:
            stitcher.process_visit(visit)

    finally:
        writer.release()

    total_wall_time = time.perf_counter() - t_start
    overall_fps = reader.total_frames / total_wall_time if total_wall_time > 0 else 0.0

    committed_visits = stitcher.committed_visits
    logger.info(
        "Pipeline finished in %.2fs (%.1f FPS). Emitted %d coherent visits.",
        total_wall_time,
        overall_fps,
        len(committed_visits),
    )

    # 7. Forensic Evidence Generation (Snapshots & Clips) and SQLite Persistence
    evidence_status: List[Dict[str, Any]] = []
    if args.save_evidence:
        logger.info(
            "Generating forensic evidence (snapshots & clips) for %d visits...",
            len(committed_visits),
        )
        for visit in committed_visits:
            # 7a. Entry Snapshot
            entry_fr = video_buffer.get_frame(visit.frame_enter)
            if entry_fr is None and video_buffer.count > 0:
                entry_fr = video_buffer.get_frame(video_buffer.oldest_frame_idx or 0)
            if entry_fr is not None:
                visit.snapshot_entry = evidence_writer.save_entry_snapshot(entry_fr, visit)

            # 7b. Peak Confidence Crop
            peak_f = visit.peak_frame or visit.frame_enter
            peak_fr = video_buffer.get_frame(peak_f)
            if peak_fr is None and entry_fr is not None:
                peak_fr = entry_fr
            if peak_fr is not None:
                visit.snapshot_peak = evidence_writer.save_peak_snapshot(peak_fr, visit)

            # 7c. Localized Pre-roll / Post-roll Clip
            pre_f = int(args.pre_roll * reader.fps)
            post_f = int(args.post_roll * reader.fps)
            clip_start = max(0, visit.frame_enter - pre_f)
            clip_end = min(reader.total_frames - 1, visit.frame_exit + post_f)
            clip_frames = video_buffer.get_frames_slice(clip_start, clip_end)
            if clip_frames:
                visit.clip_path = evidence_writer.save_event_clip(clip_frames, visit, fps=reader.fps)

            # 7d. Persist Enriched Record to SQLite
            event_store.insert_visit(
                visit=visit,
                config_hash=app_config.compute_hash(),
                model_version=f"{app_config.detector.weights}_{args.tracker}",
            )

            # Verification of evidence file existence
            entry_exists = Path(visit.snapshot_entry).exists() if visit.snapshot_entry else False
            peak_exists = Path(visit.snapshot_peak).exists() if visit.snapshot_peak else False
            clip_exists = Path(visit.clip_path).exists() if visit.clip_path else False

            evidence_status.append({
                "visit_id": visit.visit_id[:8],
                "track_id": visit.track_id,
                "snapshot_entry": visit.snapshot_entry,
                "snapshot_peak": visit.snapshot_peak,
                "clip_path": visit.clip_path,
                "entry_exists": entry_exists,
                "peak_exists": peak_exists,
                "clip_exists": clip_exists,
                "all_valid": entry_exists and peak_exists and clip_exists,
            })

    # Export finalized SQLite database
    event_store.export_to_json(args.events_json)
    event_store.export_to_csv(args.events_csv)
    event_store.close()

    # Export visits to JSON
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump([v.model_dump() for v in committed_visits], f, indent=2)
    logger.info("Exported %d visits to %s", len(committed_visits), out_json_path)

    # Export visits to CSV
    out_csv_path = Path(args.output_csv)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        if committed_visits:
            fieldnames = list(committed_visits[0].to_csv_dict().keys())
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            for v in committed_visits:
                writer_csv.writerow(v.to_csv_dict())
    logger.info("Exported %d visits to %s", len(committed_visits), out_csv_path)

    # Evaluation against ground truth
    eval_metrics = evaluate_against_ground_truth(
        predicted_visits=committed_visits,
        ground_truth_path=Path(args.ground_truth),
        roi_id=polygon_roi.config.roi_id,
    )

    # Compute Event Reduction Ratio
    event_reduction_ratio = (
        round(total_detections / float(len(committed_visits)), 1)
        if len(committed_visits) > 0
        else 0.0
    )

    telemetry = {
        "pipeline_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_source": str(video_path),
            "camera_id": state_machine.camera_id,
            "roi_id": polygon_roi.config.roi_id,
            "resolution": [reader.width, reader.height],
            "native_fps": reader.fps,
            "total_frames": reader.total_frames,
            "total_duration_s": round(reader.duration_s, 3),
        },
        "performance_benchmarks": {
            "wall_clock_time_s": round(total_wall_time, 3),
            "overall_processing_fps": round(overall_fps, 1),
            "average_detector_latency_ms": round(float(np.mean(det_latencies)), 2),
            "average_tracker_latency_ms": round(float(np.mean(trk_latencies)), 2),
            "total_raw_person_detections": total_detections,
            "total_emitted_visits": len(committed_visits),
            "event_reduction_ratio": f"{event_reduction_ratio}:1",
        },
        "visit_evaluation": eval_metrics,
        "evidence_catalog": evidence_status,
        "emitted_visits_summary": [
            {
                "visit_id": v.visit_id[:8],
                "track_id": v.track_id,
                "merged_tracks": v.merged_track_ids,
                "parent_visit_id": v.parent_visit_id[:8] if v.parent_visit_id else None,
                "person_id": v.person_id,
                "frames": f"{v.frame_enter}-{v.frame_exit}",
                "t_enter_s": v.video_t_enter_s,
                "t_exit_s": v.video_t_exit_s,
                "duration_s": v.duration_s,
                "close_reason": v.close_reason,
            }
            for v in committed_visits
        ],
    }

    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    logger.info("Saved telemetry to %s", meta_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("NAVSECURA VISIT EVENT PIPELINE BENCHMARK & EVALUATION REPORT")
    print("=" * 80)
    print(f"Video Source:             {video_path} ({reader.width}x{reader.height} @ {reader.fps} FPS)")
    print(f"Frames Processed:         {reader.total_frames} frames in {total_wall_time:.2f}s ({overall_fps:.1f} FPS)")
    print(f"Latency Profile:          Detector: {np.mean(det_latencies):.1f}ms | Tracker: {np.mean(trk_latencies):.2f}ms")
    print(f"Perception Filtering:     {total_detections} raw detections -> {len(committed_visits)} visits ({event_reduction_ratio}:1 reduction)")
    print("-" * 80)
    print("VISIT EVALUATION METRICS (vs. eval/ground_truth.csv):")
    for k, v in eval_metrics.items():
        print(f"  {k:28s}: {v}")
    print("-" * 80)
    print("EMITTED VISIT EVENTS:")
    for v in committed_visits:
        parent_tag = f" (re-entry of {v.parent_visit_id[:8]})" if v.parent_visit_id else ""
        merged_tag = f" [merged tracks: {v.merged_track_ids}]" if len(v.merged_track_ids) > 1 else ""
        evidence_tag = " [Evidence: OK]" if (v.snapshot_entry and Path(v.snapshot_entry).exists() and v.clip_path and Path(v.clip_path).exists()) else " [Evidence: None]"
        print(
            f"  * Visit {v.visit_id[:8]} | Track {v.track_id:2d}{merged_tag} | "
            f"Person {v.person_id or 'N/A':2s}{parent_tag} | "
            f"Frames {v.frame_enter:3d}-{v.frame_exit:3d} | "
            f"Time {v.video_t_enter_s:5.2f}s - {v.video_t_exit_s:5.2f}s | "
            f"Dwell: {v.duration_s:5.2f}s | Reason: {v.close_reason}{evidence_tag}"
        )
    print("=" * 80 + "\n")
    return telemetry


if __name__ == "__main__":
    main()
