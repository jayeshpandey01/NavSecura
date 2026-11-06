"""Dynamic ROI registry and auto-resolver for multi-camera video analytics.

Allows dynamic lookup, resolution adaptation, and automated ROI generation
for any video stream or camera ID without hardcoding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from navsecura.roi.geometry import PolygonROI, ROIConfig

logger = logging.getLogger("navsecura.roi.registry")


class ROIRegistry:
    """Dynamic catalog and resolver for Regions of Interest across multiple cameras."""

    def __init__(self, registry_file: Optional[Union[str, Path]] = None) -> None:
        """Initialize the registry from an optional configuration file or defaults."""
        self._configs: Dict[str, ROIConfig] = {}
        if registry_file is not None:
            self.load_registry(registry_file)

    def register(self, config: ROIConfig) -> None:
        """Register an ROI configuration dynamically."""
        self._configs[config.roi_id] = config
        logger.debug("Registered ROI: %s (%s)", config.roi_id, config.name)

    def load_registry(self, file_path: Union[str, Path]) -> None:
        """Load multiple ROI configurations from a JSON or YAML manifest."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ROI registry file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        if isinstance(data, list):
            for item in data:
                self.register(ROIConfig(**item))
        elif isinstance(data, dict):
            # Check if dict of roi_id -> config dict or a single config
            if "roi_id" in data:
                self.register(ROIConfig(**data))
            elif "rois" in data and isinstance(data["rois"], list):
                for item in data["rois"]:
                    self.register(ROIConfig(**item))
            else:
                for roi_id, item in data.items():
                    if isinstance(item, dict):
                        item.setdefault("roi_id", roi_id)
                        self.register(ROIConfig(**item))

    def get_config(
        self,
        identifier: str,
        fallback_video_stem: Optional[str] = None,
    ) -> Optional[ROIConfig]:
        """Dynamically retrieve an ROIConfig by roi_id, camera_id, or video stem."""
        # Exact match on roi_id
        if identifier in self._configs:
            return self._configs[identifier]

        # Match on camera_id
        for cfg in self._configs.values():
            if cfg.camera_id and cfg.camera_id == identifier:
                return cfg

        # Match on video stem or partial roi_id
        if fallback_video_stem:
            stem_clean = fallback_video_stem.lower().replace("-", "_")
            for key, cfg in self._configs.items():
                if stem_clean in key.lower() or (cfg.camera_id and stem_clean in cfg.camera_id.lower()):
                    return cfg

        return None

    def resolve_roi(
        self,
        roi_target: Optional[Union[str, Path, ROIConfig]] = None,
        video_path: Optional[Union[str, Path]] = None,
        frame_width: int = 768,
        frame_height: int = 432,
    ) -> PolygonROI:
        """Dynamically resolve and instantiate a PolygonROI for the given stream.

        Resolution hierarchy:
        1. Explicit ROIConfig object passed directly.
        2. File path to JSON/YAML ROI configuration.
        3. Registry lookup by identifier (roi_id or camera_id).
        4. Auto-lookup by video filename/stem.
        5. Dynamic procedural fallback tailored to frame aspect ratio.
        """
        # Case 1: Already an ROIConfig
        if isinstance(roi_target, ROIConfig):
            return PolygonROI(roi_target, frame_width, frame_height)

        # Case 2: File path provided
        if roi_target is not None:
            target_path = Path(roi_target)
            if target_path.exists() and target_path.is_file():
                cfg = ROIConfig.from_file(target_path)
                return PolygonROI(cfg, frame_width, frame_height)

            # Case 3: Identifier lookup in registry
            cfg = self.get_config(str(roi_target))
            if cfg is not None:
                return PolygonROI(cfg, frame_width, frame_height)

        # Case 4: Lookup by video path stem
        if video_path is not None:
            v_stem = Path(video_path).stem
            cfg = self.get_config(f"roi_{v_stem}", fallback_video_stem=v_stem)
            if cfg is not None:
                return PolygonROI(cfg, frame_width, frame_height)

        # Case 5: Dynamic procedural fallback ROI (normalized center/entrance corridor)
        logger.warning(
            "No ROI found for target='%s', video='%s'. Generating dynamic normalized fallback ROI.",
            roi_target,
            video_path,
        )
        v_stem = Path(video_path).stem if video_path else "stream"
        fallback_cfg = ROIConfig(
            roi_id=f"roi_dynamic_{v_stem}",
            name=f"Dynamic Auto-Generated Ground Zone ({v_stem})",
            description="Dynamically generated resolution-independent ground plane polygon",
            coordinates_norm=[
                (0.12, 0.50),  # Upper left ground
                (0.88, 0.50),  # Upper right ground
                (0.94, 0.96),  # Lower right foreground
                (0.06, 0.96),  # Lower left foreground
            ],
            foot_point_mode="bottom_center",
        )
        return PolygonROI(fallback_cfg, frame_width, frame_height)

