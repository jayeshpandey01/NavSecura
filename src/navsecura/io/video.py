"""Video stream ingestion and deterministic frame reader.

Provides frame-by-frame decoding, timestamp synchronization,
safe resource management, and container verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Generator, Iterator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("navsecura.io.video")


class VideoStreamReader:
    """Deterministic video stream reader with container validation and timebase tracking.

    Ensures timestamps are mathematically anchored to frame indices:
        timestamp_s = frame_idx / native_fps
    """

    def __init__(
        self,
        source: Union[str, Path, int],
        target_fps: Optional[float] = None,
        target_size: Optional[Tuple[int, int]] = None,
        to_rgb: bool = False,
    ) -> None:
        """Initialize the video reader.

        Args:
            source: Path to video file, RTSP/HTTP URL, or webcam device index.
            target_fps: Optional downsampled frame rate (defaults to native stream FPS).
            target_size: Optional resize dimensions as (width, height).
            to_rgb: If True, convert OpenCV BGR frames to RGB.

        Raises:
            FileNotFoundError: If a file path source does not exist.
            ValueError: If OpenCV fails to open the video stream or if container metadata is invalid.
        """
        self.source = str(source) if isinstance(source, Path) else source
        self.target_fps = target_fps
        self.target_size = target_size
        self.to_rgb = to_rgb

        if isinstance(self.source, str) and not self.source.startswith(("rtsp://", "http://", "https://")):
            source_path = Path(self.source)
            if not source_path.exists():
                raise FileNotFoundError(f"Video source not found: {source_path.resolve()}")

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._native_fps: float = 30.0
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0
        self._duration_s: float = 0.0

        self._open_stream()

    def _open_stream(self) -> None:
        """Open container and validate stream properties."""
        logger.info("Opening video source: %s", self.source)
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise ValueError(f"OpenCV failed to open video source: {self.source}")

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        header_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._native_fps = header_fps if header_fps > 0 else 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration_s = (
            self._total_frames / self._native_fps if self._native_fps > 0 and self._total_frames > 0 else 0.0
        )
        self._frame_idx = 0

        logger.info(
            "Stream validated: %dx%d @ %.2f FPS, %d frames (~%.2fs)",
            self._width,
            self._height,
            self._native_fps,
            self._total_frames,
            self._duration_s,
        )

    @property
    def width(self) -> int:
        """Stream frame width in pixels."""
        return self.target_size[0] if self.target_size else self._width

    @property
    def height(self) -> int:
        """Stream frame height in pixels."""
        return self.target_size[1] if self.target_size else self._height

    @property
    def native_fps(self) -> float:
        """Native stream frame rate."""
        return self._native_fps

    @property
    def effective_fps(self) -> float:
        """Effective processing frame rate after optional downsampling."""
        if self.target_fps is not None and 0 < self.target_fps < self._native_fps:
            return self.target_fps
        return self._native_fps

    @property
    def fps(self) -> float:
        """Effective processing frame rate (alias for effective_fps)."""
        return self.effective_fps

    @property
    def total_frames(self) -> int:
        """Total frame count reported by container header."""
        return self._total_frames

    @property
    def duration_s(self) -> float:
        """Total stream duration in seconds."""
        return self._duration_s

    @property
    def is_opened(self) -> bool:
        """Whether the underlying capture handle is open."""
        return self._cap is not None and self._cap.isOpened()

    def seek(self, frame_idx: int) -> bool:
        """Seek directly to a zero-indexed frame.

        Args:
            frame_idx: Target frame index.

        Returns:
            True if seek operation succeeded.
        """
        if self._cap is None or not self._cap.isOpened():
            return False
        ret = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if ret:
            self._frame_idx = frame_idx
        return bool(ret)

    def read_frame(self) -> Tuple[bool, Optional[Tuple[int, float, np.ndarray]]]:
        """Read the next sequential frame.

        Returns:
            Tuple of (success: bool, payload: Optional[(frame_idx, timestamp_s, frame)]).
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None

        current_idx = self._frame_idx
        timestamp_s = current_idx / self._native_fps
        self._frame_idx += 1

        if self.target_size is not None and (frame.shape[1] != self.target_size[0] or frame.shape[0] != self.target_size[1]):
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return True, (current_idx, round(timestamp_s, 4), frame)

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield frames sequentially as (frame_idx, timestamp_s, frame)."""
        frame_interval = 1.0
        if self.target_fps is not None and 0 < self.target_fps < self._native_fps:
            frame_interval = self._native_fps / self.target_fps

        next_target_frame = 0.0

        while self.is_opened:
            success, payload = self.read_frame()
            if not success or payload is None:
                break

            current_idx, timestamp_s, frame = payload

            if frame_interval > 1.0:
                if current_idx < int(round(next_target_frame)):
                    continue
                next_target_frame += frame_interval

            yield current_idx, timestamp_s, frame

    def __enter__(self) -> "VideoStreamReader":
        """Context manager entry."""
        if not self.is_opened:
            self._open_stream()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit: release capture handle safely."""
        self.close()

    def close(self) -> None:
        """Release video capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Closed video source: %s", self.source)

