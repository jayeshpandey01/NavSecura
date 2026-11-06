"""Stand-alone CLI utility for processing CCTV video streams with rich diagnostic labeling.

Visualizes detection bounding boxes, tracking ID badges, motion trajectory trails,
ground-plane foot-point crosshairs, polygon ROI zones, and real-time dwell stopwatches.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from navsecura.io.config import AppConfig
from navsecura.io.processor import CCTVVideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("process_cctv_video")


def main() -> int:
    """CLI entrypoint for CCTV video processor."""
    parser = argparse.ArgumentParser(
        description="Process CCTV video and render labeled tracking diagnostics",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/sample.mp4",
        help="Path to input video file or RTSP stream",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/cctv_labeled_tracking.mp4",
        help="Destination path for labeled video output",
    )
    parser.add_argument(
        "--roi-config",
        type=str,
        default="configs/roi_example.json",
        help="Path to JSON ROI configuration file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to application YAML configuration",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="outputs/telemetry/cctv_processing_meta.json",
        help="Destination path for output telemetry metadata JSON",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video rendering and only compute metrics",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Input video not found: %s", video_path)
        return 1

    app_config = AppConfig.load(args.config) if Path(args.config).exists() else AppConfig.load_default()
    processor = CCTVVideoProcessor(config=app_config)

    output_path = None if args.no_video else Path(args.output)
    metrics = processor.process_video(
        video_path=video_path,
        output_video_path=output_path,
        roi_config_path=Path(args.roi_config),
    )

    if args.meta:
        meta_p = Path(args.meta)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved processing telemetry to %s", meta_p)

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


if __name__ == "__main__":
    sys.exit(main())

