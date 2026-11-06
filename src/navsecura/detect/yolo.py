"""YOLO object detector adapter implementing the Detector protocol.

Licensing Disclosure:
    The underlying Ultralytics YOLO implementation is licensed under AGPL-3.0.
    In accordance with the NavSecura architectural decoupling standards, this
    adapter isolates all model-specific calls behind the abstract Detector
    protocol. Commercial deployments requiring permissive licensing can drop in
    an Apache-2.0 detector (e.g., RF-DETR or RT-DETR) by implementing the same
    protocol without altering any downstream tracker or state machine logic.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from ultralytics import YOLO

from navsecura.detect.base import Detection, Detector

logger = logging.getLogger("navsecura.detect.yolo")


class YOLODetector(Detector):
    """Ultralytics YOLO detector adapter conforming to the Detector protocol."""

    def __init__(
        self,
        weights: Union[str, Path] = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.70,
        classes: Optional[Sequence[int]] = (0,),  # COCO class 0: person
        device: str = "cpu",
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            weights: Path to model weights file (.pt).
            confidence_threshold: Minimum detection confidence score [0.0, 1.0].
            iou_threshold: Non-Maximum Suppression (NMS) IoU intersection threshold.
            classes: Sequence of COCO class indices to retain (defaults to [0] for person).
            device: Accelerator device string ('cpu', 'mps', 'cuda').
        """
        self._weights_path = Path(weights) if isinstance(weights, str) else weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = list(classes) if classes is not None else [0]
        self._device = device

        self._last_inference_ms: float = 0.0
        self._total_inferences: int = 0
        self._cumulative_latency_ms: float = 0.0

        logger.info(
            "Loading YOLO detector: weights='%s', device='%s', conf=%.2f, iou=%.2f, classes=%s",
            self._weights_path,
            self._device,
            self.confidence_threshold,
            self.iou_threshold,
            self.classes,
        )

        self._model = YOLO(str(self._weights_path))
        logger.info("YOLO detector loaded successfully.")

    @property
    def name(self) -> str:
        """Detector name and model variant."""
        return f"YOLO-{self._weights_path.stem}"

    @property
    def last_inference_ms(self) -> float:
        """Latency of the most recent inference pass in milliseconds."""
        return self._last_inference_ms

    @property
    def average_latency_ms(self) -> float:
        """Mean inference latency across all processed frames in milliseconds."""
        if self._total_inferences == 0:
            return 0.0
        return self._cumulative_latency_ms / self._total_inferences

    @property
    def device(self) -> str:
        """Active inference device."""
        return self._device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run object detection on an image frame.

        Args:
            frame: Input image array as numpy ndarray (H, W, C) in BGR format.

        Returns:
            List of confirmed Detection objects for target classes.
        """
        if frame is None or frame.size == 0:
            return []

        start_time = time.perf_counter()

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self._device,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._last_inference_ms = elapsed_ms
        self._total_inferences += 1
        self._cumulative_latency_ms += elapsed_ms

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.int().cpu().numpy()

        for box, conf, cid in zip(boxes_xyxy, confs, cls_ids):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            class_name = self._model.names.get(int(cid), "person") if hasattr(self._model, "names") else "person"
            detections.append(
                Detection(
                    box=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                    confidence=round(float(conf), 4),
                    class_id=int(cid),
                    class_name=class_name,
                )
            )

        return detections

