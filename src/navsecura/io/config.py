"""Dynamic configuration management engine for NavSecura Video Analytics.

Eliminates hardcoded paths, resolutions, and frame rates: dynamically
adapts parameters based on stream characteristics, CLI overrides, and manifests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("navsecura.config")


class StitchingConfig(BaseModel):
    """Spatio-temporal tracklet stitching configuration."""

    enabled: bool = True
    max_time_gap_s: float = 5.0
    max_distance_px: float = 150.0
    max_distance_norm: Optional[float] = None  # Normalized to diagonal


class PipelineMetaConfig(BaseModel):
    """Pipeline metadata configuration."""

    name: str = "navsecura-video-analytics"
    version: str = "1.0.0"
    random_seed: int = 42


class VideoConfig(BaseModel):
    """Video ingestion stream configuration."""

    source: str = "data/sample.mp4"
    camera_id: str = "cam_entrance_01"
    target_fps: Optional[float] = None  # None preserves native stream FPS


class DetectorConfig(BaseModel):
    """Object detector configuration."""

    model_type: Literal["yolo", "rfdetr"] = "yolo"
    weights: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.70
    classes: List[int] = Field(default_factory=lambda: [0])  # 0 = person
    device: str = "cpu"


class TrackerConfig(BaseModel):
    """Multi-object tracker configuration."""

    type: Literal["bytetrack", "botsort"] = "bytetrack"
    track_high_thresh: float = 0.50
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.60
    match_thresh: float = 0.80
    track_buffer: int = 30  # Baseline frames to retain lost tracks
    track_buffer_s: Optional[float] = 2.5  # Dynamic buffer duration in seconds
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.25
    with_reid: bool = False


class ROIConfigRef(BaseModel):
    """ROI reference and foot-point mode configuration."""

    config_path: str = "configs/roi_example.json"
    roi_id: Optional[str] = None
    foot_point_position: str = "bottom_center"


class EventsConfig(BaseModel):
    """Visit lifecycle and temporal hysteresis configuration."""

    enter_consecutive_frames: int = 3
    grace_period_s: float = 2.0
    min_dwell_s: float = 1.0
    stitching: StitchingConfig = Field(default_factory=StitchingConfig)


class StoreConfig(BaseModel):
    """Event database persistence configuration."""

    db_path: str = "outputs/events.db"
    wal_mode: bool = True


class EvidenceConfig(BaseModel):
    """Evidence extraction and clip buffer configuration."""

    output_dir: str = "outputs"
    save_snapshots: bool = True
    save_clips: bool = True
    pre_roll_s: float = 3.0
    post_roll_s: float = 3.0


class AppConfig(BaseModel):
    """Root application configuration."""

    pipeline: PipelineMetaConfig = Field(default_factory=PipelineMetaConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    roi: ROIConfigRef = Field(default_factory=ROIConfigRef)
    events: EventsConfig = Field(default_factory=EventsConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)

    @classmethod
    def load(
        cls,
        config_path: Union[str, Path] = "configs/default.yaml",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "AppConfig":
        """Load configuration dynamically from YAML file with overrides.

        Args:
            config_path: Path to default or custom YAML configuration.
            overrides: Optional key-value dictionary to override nested settings.

        Returns:
            Instantiated and validated AppConfig.
        """
        path = Path(config_path)
        data: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        # Apply overrides dynamically
        if overrides:
            for key, val in overrides.items():
                if val is None:
                    continue
                if "." in key:
                    section, subkey = key.split(".", 1)
                    if section not in data:
                        data[section] = {}
                    data[section][subkey] = val
                else:
                    data[key] = val

        # Handle alias resolution from configs/videos.yaml if source is an alias
        source = data.get("video", {}).get("source", "")
        videos_manifest = Path("configs/videos.yaml")
        if videos_manifest.exists() and source and not Path(source).exists():
            with open(videos_manifest, "r", encoding="utf-8") as vf:
                vmanifest = yaml.safe_load(vf) or {}
            scenarios = vmanifest.get("scenarios", {})
            if source in scenarios:
                sc_info = scenarios[source]
                data["video"]["source"] = sc_info.get("video_path", source)
                data["video"]["camera_id"] = sc_info.get("camera_id", data["video"].get("camera_id"))
                if "roi_id" in sc_info:
                    data["roi"]["roi_id"] = sc_info["roi_id"]
                    data["roi"]["config_path"] = "configs/roi_registry.json"
                logger.info("Resolved scenario alias '%s' -> %s", source, data["video"]["source"])

        return cls(**data)

    def adapt_to_stream(self, native_fps: float, width: int, height: int) -> None:
        """Dynamically adapt temporal and spatial parameters to inspected stream.

        Args:
            native_fps: Verified stream frame rate.
            width: Stream width in pixels.
            height: Stream height in pixels.
        """
        if native_fps <= 0:
            return

        # Adapt tracker buffer dynamically to frame rate (e.g. 2.5s of retention)
        if self.tracker.track_buffer_s:
            self.tracker.track_buffer = max(10, int(round(native_fps * self.tracker.track_buffer_s)))

        # Adapt spatial stitching distance to resolution diagonal if not explicitly set
        diagonal = (width**2 + height**2) ** 0.5
        if self.events.stitching.max_distance_norm is None:
            self.events.stitching.max_distance_norm = 0.15  # 15% of frame diagonal
        self.events.stitching.max_distance_px = round(diagonal * self.events.stitching.max_distance_norm, 1)

        logger.info(
            "Adapted config to stream @ %.2f FPS, %dx%d: track_buffer=%d, max_stitch_dist=%.1fpx",
            native_fps,
            width,
            height,
            self.tracker.track_buffer,
            self.events.stitching.max_distance_px,
        )

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of configuration parameters."""
        import hashlib
        config_json = self.model_dump_json()
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]


