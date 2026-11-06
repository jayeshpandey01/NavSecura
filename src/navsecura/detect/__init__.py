"""Object detection protocols and model adapters."""

from navsecura.detect.base import Detection, Detector
from navsecura.detect.yolo import YOLODetector

__all__ = ["Detection", "Detector", "YOLODetector"]
