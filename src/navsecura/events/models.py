"""Data models and schemas for visit lifecycle events.

Defines state enumerations, reason codes, and strongly typed visit
event schemas adhering strictly to PEP 484 and Pydantic validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field, model_validator


class VisitState(str, Enum):
    """Lifecycle states of a tracked entity with respect to an ROI polygon."""

    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"
    GRACE = "GRACE"
    CLOSED = "CLOSED"


class CloseReason(str, Enum):
    """Reason for closing an active visit session."""

    EXITED = "exited"
    TRACK_LOST = "track_lost"
    STREAM_END = "stream_end"
    MERGED = "merged"


class Visit(BaseModel):
    """Comprehensive record of a single, coherent zone visit.

    Aggregates multi-frame detections into a single business event,
    recording entry/exit timebases, ground-plane coordinates, confidence,
    and spatio-temporal stitching lineage.
    """

    visit_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID v4 identifying the visit event.",
    )
    camera_id: str = Field(
        default="camera_0",
        description="Identifier of the capturing sensor or video stream.",
    )
    roi_id: str = Field(
        ...,
        description="Identifier of the polygon Region of Interest.",
    )
    track_id: int = Field(
        ...,
        description="Primary temporary multi-object tracker ID.",
    )
    merged_track_ids: List[int] = Field(
        default_factory=list,
        description="List of tracker IDs stitched into this visit session.",
    )
    parent_visit_id: Optional[str] = Field(
        None,
        description="Reference to previous visit ID for re-entering visitors.",
    )
    person_id: Optional[str] = Field(
        None,
        description="Resolved human identity (e.g. P1, P2, P6) from ground truth or re-ID.",
    )
    entered_at: Optional[str] = Field(
        None,
        description="ISO-8601 wall-clock timestamp of zone entry.",
    )
    exited_at: Optional[str] = Field(
        None,
        description="ISO-8601 wall-clock timestamp of zone exit.",
    )
    video_t_enter_s: float = Field(
        ...,
        description="Offset in seconds from video start when subject entered ROI.",
    )
    video_t_exit_s: float = Field(
        ...,
        description="Offset in seconds from video start when subject last seen in ROI.",
    )
    frame_enter: int = Field(
        ...,
        description="Frame index of polygon entry.",
    )
    frame_exit: int = Field(
        ...,
        description="Frame index where subject was last confirmed inside ROI.",
    )
    duration_s: float = Field(
        ...,
        description="Net dwell duration in seconds (strictly video_t_exit_s - video_t_enter_s).",
    )
    mean_conf: float = Field(
        default=0.0,
        description="Average detector confidence across all in-zone observations.",
    )
    peak_conf: float = Field(
        default=0.0,
        description="Peak detector confidence observed during the visit.",
    )
    peak_frame: Optional[int] = Field(
        default=None,
        description="Frame index of peak confidence observation.",
    )
    close_reason: str = Field(
        default=CloseReason.EXITED.value,
        description="Reason code: exited, track_lost, stream_end, or merged.",
    )
    entry_box: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="Bounding box [x1, y1, x2, y2] upon confirmed entry.",
    )
    entry_foot_point: Optional[Tuple[float, float]] = Field(
        None,
        description="Ground contact point (x, y) upon confirmed entry.",
    )
    peak_box: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="Bounding box [x1, y1, x2, y2] corresponding to peak confidence.",
    )
    last_box: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="Bounding box [x1, y1, x2, y2] at last confirmed in-zone frame.",
    )
    last_foot_point: Optional[Tuple[float, float]] = Field(
        None,
        description="Ground contact point (x, y) at last confirmed in-zone frame.",
    )
    color_histogram: Optional[List[float]] = Field(
        None,
        description="Normalized appearance color histogram for re-ID stitching.",
    )
    snapshot_entry: Optional[str] = Field(
        None,
        description="Relative filepath to the entry evidence snapshot image.",
    )
    snapshot_peak: Optional[str] = Field(
        None,
        description="Relative filepath to the peak confidence snapshot image.",
    )
    clip_path: Optional[str] = Field(
        None,
        description="Relative filepath to the evidence video clip.",
    )
    config_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the configuration parameters used.",
    )
    model_version: Optional[str] = Field(
        None,
        description="Model identifier/version used for detection.",
    )
    notes: Optional[str] = Field(
        None,
        description="Human-readable contextual annotations or operational tags.",
    )

    @model_validator(mode="after")
    def _ensure_track_ids(self) -> "Visit":
        """Ensure merged_track_ids contains at least the primary track_id."""
        if not self.merged_track_ids:
            self.merged_track_ids = [self.track_id]
        elif self.track_id not in self.merged_track_ids:
            self.merged_track_ids.insert(0, self.track_id)
        return self

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert visit model to a flat dictionary suitable for CSV export."""
        return {
            "visit_id": self.visit_id,
            "person_id": self.person_id or f"T{self.track_id}",
            "frame_enter": self.frame_enter,
            "frame_exit": self.frame_exit,
            "t_enter_s": round(self.video_t_enter_s, 3),
            "t_exit_s": round(self.video_t_exit_s, 3),
            "duration_s": round(self.duration_s, 3),
            "roi_id": self.roi_id,
            "notes": self.notes or self.close_reason,
        }
