"""Video stream inspection and telemetry tool.

Deeply inspects video containers, decoding integrity, resolution,
FPS, and stream continuity without assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inspect_video")


def compute_md5(file_path: Path, block_size: int = 65536) -> str:
    """Compute the MD5 checksum of a file.

    Args:
        file_path: Path to the target file.
        block_size: Buffer block size in bytes.

    Returns:
        Hexadecimal MD5 digest string.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_aspect_ratio(width: int, height: int) -> str:
    """Compute the simplified aspect ratio string.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        String representing simplified aspect ratio (e.g., '16:9').
    """
    if width <= 0 or height <= 0:
        return "unknown"
    divisor = math.gcd(width, height)
    simplified_w = width // divisor
    simplified_h = height // divisor
    # Detect standard surveillance approximations
    ratio_float = width / height
    if abs(ratio_float - 16 / 9) < 0.02:
        return "16:9"
    if abs(ratio_float - 4 / 3) < 0.02:
        return "4:3"
    return f"{simplified_w}:{simplified_h}"


def decode_fourcc(fourcc_val: float) -> str:
    """Convert OpenCV fourcc float/int value to a 4-character string.

    Args:
        fourcc_val: The numeric fourcc value from cv2.CAP_PROP_FOURCC.

    Returns:
        Human-readable fourcc code string.
    """
    try:
        val = int(fourcc_val)
        return "".join([chr((val >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return "UNKNOWN"


def inspect_video_stream(
    video_path: Path,
    output_path: Optional[Path] = None,
    verify_every_frame: bool = True,
) -> Dict[str, Any]:
    """Inspect video file container metadata and verify frame decode continuity.

    Args:
        video_path: Path to the target video file.
        output_path: Optional path to write JSON telemetry results.
        verify_every_frame: Whether to decode all frames sequentially to verify integrity.

    Returns:
        Dictionary of stream telemetry and integrity metrics.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV fails to open the video stream.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    file_size_bytes = video_path.stat().st_size
    md5_hash = compute_md5(video_path)

    logger.info("Opening video stream: %s", video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video stream at: {video_path}")

    # Read container header metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    header_fps = float(cap.get(cv2.CAP_PROP_FPS))
    header_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
    codec = decode_fourcc(fourcc_val)
    aspect_ratio = compute_aspect_ratio(width, height)

    header_duration_s = (
        round(header_frame_count / header_fps, 3)
        if header_fps > 0 and header_frame_count > 0
        else 0.0
    )

    logger.info(
        "Container Header: %dx%d (%s) @ %.2f FPS, %d frames (~%.2fs), Codec: %s",
        width,
        height,
        aspect_ratio,
        header_fps,
        header_frame_count,
        header_duration_s,
        codec,
    )

    decoded_frames = 0
    dropped_or_corrupt_frames = 0
    start_decode_time = time.perf_counter()

    if verify_every_frame:
        logger.info("Decoding all frames sequentially to test stream continuity...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0 or frame.shape[0] != height or frame.shape[1] != width:
                dropped_or_corrupt_frames += 1
            decoded_frames += 1

    elapsed_decode_time = time.perf_counter() - start_decode_time
    cap.release()

    decode_throughput_fps = (
        round(decoded_frames / elapsed_decode_time, 2)
        if elapsed_decode_time > 0
        else 0.0
    )
    actual_duration_s = (
        round(decoded_frames / header_fps, 3) if header_fps > 0 else 0.0
    )

    stream_valid = (
        decoded_frames > 0
        and dropped_or_corrupt_frames == 0
        and (header_frame_count == 0 or abs(decoded_frames - header_frame_count) <= 1)
    )

    telemetry: Dict[str, Any] = {
        "file_name": video_path.name,
        "file_path": str(video_path.resolve()),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "md5_hash": md5_hash,
        "container_metadata": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "header_fps": header_fps,
            "header_frame_count": header_frame_count,
            "header_duration_s": header_duration_s,
            "codec_fourcc": codec,
            "fourcc_numeric": int(fourcc_val),
        },
        "decode_verification": {
            "decoded_frames": decoded_frames,
            "dropped_or_corrupt_frames": dropped_or_corrupt_frames,
            "actual_duration_s": actual_duration_s,
            "elapsed_decode_seconds": round(elapsed_decode_time, 4),
            "decode_throughput_fps": decode_throughput_fps,
            "is_frame_count_consistent": (decoded_frames == header_frame_count),
            "stream_valid": stream_valid,
        },
    }

    logger.info(
        "Verification Complete: %d decoded frames, %d corrupted, %.1f FPS decode speed. Stream Valid: %s",
        decoded_frames,
        dropped_or_corrupt_frames,
        decode_throughput_fps,
        stream_valid,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        logger.info("Saved telemetry report to: %s", output_path)

    return telemetry


def main() -> int:
    """CLI entrypoint for video inspection tool."""
    parser = argparse.ArgumentParser(
        description="Inspect video container metadata and verify sequential frame integrity."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video file (e.g. data/sample.mp4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON telemetry results (e.g. outputs/video_meta.json)",
    )
    parser.add_argument(
        "--no-verify-frames",
        action="store_true",
        help="Skip exhaustive sequential frame decoding (header inspection only).",
    )

    args = parser.parse_args()

    try:
        inspect_video_stream(
            video_path=args.video,
            output_path=args.output,
            verify_every_frame=not args.no_verify_frames,
        )
        return 0
    except Exception as exc:
        logger.error("Video inspection failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

