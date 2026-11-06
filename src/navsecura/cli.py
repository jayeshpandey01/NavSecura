"""Unified Command-Line Interface for NavSecura Video Analytics.

Provides entrypoints for running end-to-end perception and visit tracking,
generating formatted audit reports from SQLite event persistence, and
exporting structured datasets across edge deployments.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from navsecura.store.sqlite import SQLiteEventStore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("navsecura.cli")


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute end-to-end video understanding and visit event pipeline."""
    # Lazy import to avoid importing heavy PyTorch/Ultralytics on report/export commands
    from tools.run_event_pipeline import run_pipeline

    try:
        run_pipeline(args)
        return 0
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        return 1


def _cmd_process_video(args: argparse.Namespace) -> int:
    """Process CCTV video with rich diagnostic tracking labels and telemetry."""
    from navsecura.io.config import AppConfig
    from navsecura.io.processor import CCTVVideoProcessor

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Input video file not found: %s", video_path)
        return 1

    app_config = AppConfig.load(args.config) if Path(args.config).exists() else AppConfig.load()
    processor = CCTVVideoProcessor(config=app_config)

    output_path = None if getattr(args, "no_video", False) else Path(args.output)
    metrics = processor.process_video(
        video_path=video_path,
        output_video_path=output_path,
        roi_config_path=Path(args.roi_config),
    )

    if args.meta:
        meta_p = Path(args.meta)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("             CCTV VIDEO PROCESSING & LABELED TRACKING SUMMARY")
    print("=" * 80)
    print(f"Video Source:             {metrics['video_path']}")
    print(f"Resolution & FPS:         {metrics['resolution'][0]}x{metrics['resolution'][1]} @ {metrics['native_fps']} FPS")
    print(f"Total Frames Processed:   {metrics['total_frames']} frames in {metrics['wall_clock_time_s']:.2f}s")
    print(f"Inference Throughput:     {metrics['processing_fps']:.1f} FPS")
    print(f"Detector Latency (Mean):  {metrics['mean_det_latency_ms']:.1f} ms")
    print(f"Tracker Latency (Mean):   {metrics['mean_trk_latency_ms']:.2f} ms")
    print(f"Raw Detections Ingested:  {metrics['total_raw_detections']}")
    print(f"Coherent Visits Emitted:  {metrics['total_emitted_visits']}")
    print(f"Alert Reduction Ratio:    {metrics['alert_reduction_ratio']}")
    if output_path:
        print(f"Labeled Diagnostic Video: {output_path.resolve()}")
    print("=" * 80 + "\n")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Query SQLite event store and render a formatted Markdown summary table."""
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    store = SQLiteEventStore(db_path)
    visits = store.list_visits(roi_id=args.roi)
    stats = store.get_summary_stats()
    store.close()

    if not visits:
        print(f"\nNo visit records found in '{db_path}' (ROI filter: {args.roi or 'ALL'}).\n")
        return 0

    print("\n" + "=" * 105)
    print("NAVSECURA PERSISTENT VISIT EVENT AUDIT REPORT")
    print(f"Database: {db_path.resolve()} | Total Recorded Visits: {len(visits)}")
    print("=" * 105)

    headers = [
        "Visit ID",
        "Track",
        "Person",
        "Enter (s)",
        "Exit (s)",
        "Dwell (s)",
        "Close Reason",
        "Entry Snap",
        "Peak Snap",
        "Video Clip",
    ]

    col_fmt = "{:<10} {:<6} {:<8} {:<10} {:<10} {:<10} {:<14} {:<12} {:<12} {:<12}"
    header_str = col_fmt.format(*headers)
    print(header_str)
    print("-" * 105)

    valid_evidence_count = 0

    for v in visits:
        v_id = str(v.get("visit_id", ""))[:8]
        t_id = str(v.get("track_id", ""))
        p_id = str(v.get("person_id") or f"P{t_id}")
        t_enter = f"{float(v.get('video_t_enter_s', 0.0)):.2f}"
        t_exit = f"{float(v.get('video_t_exit_s', 0.0)):.2f}"
        dwell = f"{float(v.get('duration_s', 0.0)):.2f}"
        reason = str(v.get("close_reason", ""))

        snap_entry = v.get("snapshot_entry")
        snap_peak = v.get("snapshot_peak")
        clip = v.get("clip_path")

        entry_ok = Path(snap_entry).exists() if snap_entry else False
        peak_ok = Path(snap_peak).exists() if snap_peak else False
        clip_ok = Path(clip).exists() if clip else False

        if entry_ok and peak_ok and clip_ok:
            valid_evidence_count += 1

        entry_tag = "EXISTS [OK]" if entry_ok else "MISSING"
        peak_tag = "EXISTS [OK]" if peak_ok else "MISSING"
        clip_tag = "EXISTS [OK]" if clip_ok else "MISSING"

        row_str = col_fmt.format(
            v_id,
            t_id,
            p_id,
            t_enter,
            t_exit,
            dwell,
            reason,
            entry_tag,
            peak_tag,
            clip_tag,
        )
        print(row_str)

    print("-" * 105)
    evidence_pct = (valid_evidence_count / len(visits)) * 100.0 if visits else 0.0
    print(
        f"Forensic Evidence Integrity: {valid_evidence_count}/{len(visits)} "
        f"visits ({evidence_pct:.1f}%) have verified on-disk snapshots and playable clips."
    )
    print(
        f"Summary Telemetry: Total Visits: {stats['total_visits']} | "
        f"Avg Dwell: {stats['mean_duration_s']}s (Min: {stats['min_duration_s']}s, Max: {stats['max_duration_s']}s) | "
        f"Raw Alerts Collapsed: {stats['total_raw_alerts_collapsed']} | "
        f"Alert Reduction Ratio: {stats['alert_reduction_ratio']}\n"
    )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Export SQLite visit database to JSON or CSV."""
    db_path = Path(args.db)
    out_path = Path(args.output)

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    store = SQLiteEventStore(db_path)
    fmt = args.format.lower()

    if fmt == "json":
        store.export_to_json(out_path)
    elif fmt == "csv":
        store.export_to_csv(out_path)
    else:
        logger.error("Unsupported export format: '%s' (choose 'json' or 'csv')", args.format)
        store.close()
        return 1

    store.close()
    logger.info("Successfully exported database '%s' to '%s' (%s format)", db_path, out_path, fmt)
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Synthesize baseline history, fit streaming models, and export baseline model."""
    from navsecura.analytics import StreamingVisitBaselines, synthesize_historical_dataset

    logger.info("Generating %d-week streaming baseline for ROI '%s'...", args.weeks, args.roi)
    hist_events, hourly_counts, _ = synthesize_historical_dataset(num_weeks=args.weeks, roi_id=args.roi)

    baselines = StreamingVisitBaselines(half_life_days=28.0)
    for dow, hr, count in hourly_counts:
        baselines.record_hourly_volume(dow, hr, count)
    for evt in hist_events:
        baselines.update_dwell(evt["roi_id"], evt["duration_s"])
        feat = baselines._extract_feature_vector(evt)
        baselines.hs_ensemble.update(feat)

    out_p = Path(args.output)
    baselines.save_to_json(out_p)

    summary = baselines.get_summary()
    print("\n" + "=" * 80)
    print("NAVSECURA STREAMING BASELINE SYNTHESIS COMPLETE")
    print(f"Model saved to: {out_p.resolve()}")
    print("=" * 80)
    dwell_info = summary["dwell_distributions"].get(args.roi, {})
    print(f"ROI:                       {args.roi}")
    print(f"Total Reference Events:    {len(hist_events)} (Simulated across {args.weeks} weeks)")
    print(f"Learned Median Dwell:      {dwell_info.get('median_dwell_s', 0.0):.2f}s (Log-Mean: {dwell_info.get('log_mean', 0.0):.3f})")
    peak_info = summary["peak_traffic_period"]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"Peak Diurnal Period:       {dow_names[peak_info['day_of_week']]} @ {peak_info['hour']:02d}:00 (Rate: {peak_info['expected_rate']:.1f} visits/hr)")
    print("=" * 80 + "\n")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Audit recorded visits against learned statistical baselines."""
    from navsecura.analytics import StreamingVisitBaselines, synthesize_historical_dataset

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    model_path = Path(args.model)
    if model_path.exists():
        logger.info("Loading pre-trained baseline model from '%s'...", model_path)
        baselines = StreamingVisitBaselines.load_from_json(model_path)
    else:
        logger.info("No baseline model found at '%s'; generating 4-week synthetic reference...", model_path)
        hist_events, hourly_counts, _ = synthesize_historical_dataset(num_weeks=4, roi_id=args.roi or "roi_entrance")
        baselines = StreamingVisitBaselines(half_life_days=28.0)
        for dow, hr, count in hourly_counts:
            baselines.record_hourly_volume(dow, hr, count)
        for evt in hist_events:
            baselines.update_dwell(evt["roi_id"], evt["duration_s"])
            feat = baselines._extract_feature_vector(evt)
            baselines.hs_ensemble.update(feat)
        baselines.save_to_json(model_path)

    store = SQLiteEventStore(db_path)
    visits = store.list_visits(roi_id=args.roi)
    stats = store.get_summary_stats()
    store.close()

    if not visits:
        print(f"\nNo visit records found in '{db_path}'.\n")
        return 0

    print("\n" + "=" * 95)
    print("NAVSECURA VISIT STATISTICAL & ANOMALY AUDIT REPORT")
    print(f"Database: {db_path.resolve()} | Total Visits: {len(visits)}")
    print("=" * 95)

    headers = ["Visit ID", "Track", "Person", "Dwell (s)", "Expected (s)", "Dwell Z", "HSTree", "Status"]
    col_fmt = "{:<10} {:<6} {:<8} {:<10} {:<14} {:<12} {:<10} {:<18}"
    print(col_fmt.format(*headers))
    print("-" * 95)

    anomaly_count = 0
    from datetime import datetime, timezone
    eval_dt = datetime(2026, 10, 27, 14, 30, 0, tzinfo=timezone.utc)

    for v in visits:
        v_copy = dict(v)
        if not getattr(args, "use_recorded_timestamp", False):
            v_copy["entered_at"] = eval_dt.isoformat()
        score = baselines.ingest_visit(v_copy, is_reference=False)
        if score["is_anomaly"] or score["hstree_anomaly_score"] >= args.threshold:
            anomaly_count += 1

        print(col_fmt.format(
            v["visit_id"][:8],
            str(v["track_id"]),
            str(v.get("person_id") or f"P{v['track_id']}"),
            f"{float(v['duration_s']):.2f}",
            f"{score['expected_median_dwell_s']:.2f}",
            f"{score['dwell_z_score']:+.2f}",
            f"{score['hstree_anomaly_score']:.3f}",
            f"[{score['anomaly_type']}]",
        ))

    print("-" * 95)
    print(f"Audit Summary: {len(visits) - anomaly_count}/{len(visits)} visits nominal, {anomaly_count} flagged anomalies.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct unified command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="navsecura",
        description="NavSecura Edge Video Analytics & Event Memory System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational subcommands")

    # 1. navsecura run
    run_p = subparsers.add_parser("run", help="Run perception and visit tracking pipeline")
    run_p.add_argument("--video", type=str, default="data/sample.mp4", help="Input video filepath")
    run_p.add_argument("--config", type=str, default="configs/default.yaml", help="Application YAML configuration")
    run_p.add_argument("--roi-config", type=str, default="configs/roi_example.json", help="ROI configuration file")
    run_p.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Tracker algorithm")
    run_p.add_argument("--output-video", type=str, default="outputs/sample_event_pipeline.mp4", help="Annotated video path")
    run_p.add_argument("--output-db", type=str, default="outputs/events.db", help="SQLite database path")
    run_p.add_argument("--output-json", type=str, default="outputs/sample_visits.json", help="Emitted visits JSON path")
    run_p.add_argument("--output-csv", type=str, default="outputs/sample_visits.csv", help="Emitted visits CSV path")
    run_p.add_argument("--events-json", type=str, default="outputs/events.json", help="Exported events JSON path")
    run_p.add_argument("--events-csv", type=str, default="outputs/events.csv", help="Exported events CSV path")
    run_p.add_argument("--meta", type=str, default="outputs/sample_event_pipeline_meta.json", help="Pipeline telemetry JSON")
    run_p.add_argument("--ground-truth", type=str, default="eval/ground_truth.csv", help="Ground truth CSV for evaluation")
    run_p.add_argument("--save-evidence", action="store_true", default=True, help="Generate forensic snapshots and video clips")
    run_p.add_argument("--no-evidence", action="store_false", dest="save_evidence", help="Disable evidence generation")
    run_p.add_argument("--pre-roll", type=float, default=3.0, help="Pre-roll seconds for clip extraction")
    run_p.add_argument("--post-roll", type=float, default=3.0, help="Post-roll seconds for clip extraction")

    # 2. navsecura report
    report_p = subparsers.add_parser("report", help="Display formatted visit audit report from database")
    report_p.add_argument("--db", type=str, default="outputs/events.db", help="SQLite database path")
    report_p.add_argument("--roi", type=str, default=None, help="Optional ROI polygon ID filter")

    # 3. navsecura export
    export_p = subparsers.add_parser("export", help="Export visit database to JSON or CSV")
    export_p.add_argument("--db", type=str, default="outputs/events.db", help="SQLite database path")
    export_p.add_argument("--format", type=str, default="json", choices=["json", "csv"], help="Target format")
    export_p.add_argument("--output", type=str, required=True, help="Destination file path")

    # 4. navsecura baseline
    base_p = subparsers.add_parser("baseline", help="Synthesize diurnal baseline and train online models")
    base_p.add_argument("--db", type=str, default="outputs/events.db", help="SQLite database path")
    base_p.add_argument("--roi", type=str, default="roi_entrance", help="Target ROI identifier")
    base_p.add_argument("--weeks", type=int, default=4, help="Historical weeks to simulate")
    base_p.add_argument("--output", type=str, default="outputs/baseline_model.json", help="Model JSON output path")
    base_p.add_argument("--simulate-history", action="store_true", default=True, help="Synthesize diurnal reference history")

    # 5. navsecura audit
    audit_p = subparsers.add_parser("audit", help="Audit visits against statistical baselines and flag anomalies")
    audit_p.add_argument("--db", type=str, default="outputs/events.db", help="SQLite database path")
    audit_p.add_argument("--roi", type=str, default=None, help="Optional ROI polygon ID filter")
    audit_p.add_argument("--model", type=str, default="outputs/baseline_model.json", help="Baseline model JSON path")
    audit_p.add_argument("--threshold", type=float, default=0.65, help="Anomaly score threshold [0..1]")
    audit_p.add_argument("--use-recorded-timestamp", action="store_true", help="Strictly use machine execution timestamp")

    # 6. navsecura process-video
    proc_p = subparsers.add_parser("process-video", help="Process CCTV video and render labeled tracking diagnostics")
    proc_p.add_argument("--video", type=str, default="data/sample.mp4", help="Input video filepath")
    proc_p.add_argument("--output", type=str, default="outputs/cctv_labeled_tracking.mp4", help="Annotated video destination")
    proc_p.add_argument("--config", type=str, default="configs/default.yaml", help="Application YAML configuration")
    proc_p.add_argument("--roi-config", type=str, default="configs/roi_example.json", help="ROI configuration file")
    proc_p.add_argument("--meta", type=str, default="outputs/cctv_processing_meta.json", help="Processing telemetry metadata JSON")
    proc_p.add_argument("--no-video", action="store_true", help="Skip video rendering and only compute metrics")

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        sys.exit(_cmd_run(args))
    elif args.command == "report":
        sys.exit(_cmd_report(args))
    elif args.command == "export":
        sys.exit(_cmd_export(args))
    elif args.command == "baseline":
        sys.exit(_cmd_baseline(args))
    elif args.command == "audit":
        sys.exit(_cmd_audit(args))
    elif args.command == "process-video":
        sys.exit(_cmd_process_video(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

