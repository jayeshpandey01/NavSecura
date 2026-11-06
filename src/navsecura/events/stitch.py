"""Spatio-temporal ID stitching and re-entry resolution engine.

Resolves multi-object tracker imperfections across two temporal regimes:
1. Short Occlusion / ID-Switch Regime (Delta_t <= stitch_window_s):
   Merges fragmented tracklets into a single coherent visit event.
2. Re-Entry Regime (Delta_t > reentry_threshold_s):
   Emits a distinct visit event session, but links 'parent_visit_id' and
   retains persistent 'person_id' via appearance and trajectory memory.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from navsecura.events.models import CloseReason, Visit

logger = logging.getLogger("navsecura.events.stitch")


class TrackletStitcher:
    """Performs spatio-temporal stitching and re-entry linking on visits."""

    def __init__(
        self,
        stitch_window_s: float = 3.0,
        max_stitch_dist_px: float = 120.0,
        reentry_threshold_s: float = 5.0,
        max_reentry_window_s: float = 120.0,
        min_appearance_sim: float = 0.80,
    ) -> None:
        """Initialize the stitcher.

        Args:
            stitch_window_s: Maximum temporal gap (seconds) to merge fragmented tracklets into one visit.
            max_stitch_dist_px: Maximum Euclidean spatial distance (pixels) for merge candidates.
            reentry_threshold_s: Minimum absence duration (seconds) that constitutes a re-entry.
            max_reentry_window_s: Maximum temporal lookback (seconds) for re-entry linkage.
            min_appearance_sim: Minimum cosine similarity of normalized color histograms to confirm identity.
        """
        self.stitch_window_s = stitch_window_s
        self.max_stitch_dist_px = max_stitch_dist_px
        self.reentry_threshold_s = reentry_threshold_s
        self.max_reentry_window_s = max_reentry_window_s
        self.min_appearance_sim = min_appearance_sim

        self.committed_visits: List[Visit] = []
        self._next_person_idx: int = 1
        self._track_to_person: Dict[int, str] = {}

    def process_visit(self, candidate_visit: Visit) -> Visit:
        """Process a newly closed visit, evaluating merge or re-entry linkage.

        Args:
            candidate_visit: Newly closed Visit event.

        Returns:
            Processed Visit with resolved person_id and parent_visit_id (or updated existing visit).
        """
        # Regime 1: Check for short-gap merge into an existing visit
        for prev in reversed(self.committed_visits):
            delta_t = candidate_visit.video_t_enter_s - prev.video_t_exit_s
            if delta_t < 0:
                # Overlapping visit -> cannot be a sequential continuation
                continue
            if delta_t > self.stitch_window_s:
                break

            spatial_dist = self._compute_spatial_distance(prev, candidate_visit)
            app_sim = self._compute_appearance_similarity(prev, candidate_visit)

            is_spatial_ok = spatial_dist <= self.max_stitch_dist_px
            is_app_ok = app_sim is None or app_sim >= self.min_appearance_sim

            if is_spatial_ok and is_app_ok:
                # Merge into previous visit!
                logger.info(
                    "Stitching track %d (frame %d-%d) into visit %s (track %d, frame %d-%d). delta_t=%.3fs, dist=%.1fpx, sim=%s",
                    candidate_visit.track_id,
                    candidate_visit.frame_enter,
                    candidate_visit.frame_exit,
                    prev.visit_id[:8],
                    prev.track_id,
                    prev.frame_enter,
                    prev.frame_exit,
                    delta_t,
                    spatial_dist,
                    f"{app_sim:.3f}" if app_sim is not None else "N/A",
                )
                prev.merged_track_ids.append(candidate_visit.track_id)
                prev.frame_exit = candidate_visit.frame_exit
                prev.video_t_exit_s = candidate_visit.video_t_exit_s
                prev.duration_s = round(prev.video_t_exit_s - prev.video_t_enter_s, 3)
                prev.last_box = candidate_visit.last_box
                prev.last_foot_point = candidate_visit.last_foot_point
                if candidate_visit.peak_conf > prev.peak_conf:
                    prev.peak_conf = candidate_visit.peak_conf
                    prev.peak_box = candidate_visit.peak_box
                prev.close_reason = CloseReason.MERGED.value
                return prev

        # Regime 2: Check for re-entry linkage (absence > reentry_threshold_s)
        reentry_parent: Optional[Visit] = None
        best_sim: float = -1.0

        for prev in reversed(self.committed_visits):
            delta_t = candidate_visit.video_t_enter_s - prev.video_t_exit_s
            if delta_t < self.reentry_threshold_s:
                continue
            if delta_t > self.max_reentry_window_s:
                break

            app_sim = self._compute_appearance_similarity(prev, candidate_visit)
            if app_sim is not None and app_sim >= self.min_appearance_sim:
                if app_sim > best_sim:
                    best_sim = app_sim
                    reentry_parent = prev

        if reentry_parent is not None:
            # Re-entry confirmed
            candidate_visit.parent_visit_id = reentry_parent.visit_id
            candidate_visit.person_id = (
                reentry_parent.person_id or f"P{reentry_parent.track_id}"
            )
            candidate_visit.notes = (
                f"re-entry of {candidate_visit.person_id} after "
                f"{candidate_visit.video_t_enter_s - reentry_parent.video_t_exit_s:.2f}s absence"
            )
            logger.info(
                "Resolved re-entry: Track %d linked to %s (%s, sim=%.3f)",
                candidate_visit.track_id,
                reentry_parent.visit_id[:8],
                candidate_visit.person_id,
                best_sim,
            )
        else:
            # Fresh visitor identity
            if candidate_visit.track_id in self._track_to_person:
                candidate_visit.person_id = self._track_to_person[candidate_visit.track_id]
            else:
                p_label = f"P{self._next_person_idx}"
                self._next_person_idx += 1
                self._track_to_person[candidate_visit.track_id] = p_label
                candidate_visit.person_id = p_label

        self.committed_visits.append(candidate_visit)
        return candidate_visit

    def _compute_spatial_distance(self, v1: Visit, v2: Visit) -> float:
        """Compute Euclidean distance between v1 last foot-point and v2 entry foot-point."""
        pt1 = v1.last_foot_point or v1.entry_foot_point
        pt2 = v2.entry_foot_point or v2.last_foot_point
        if pt1 is None or pt2 is None:
            return 0.0
        return float(math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))

    def _compute_appearance_similarity(
        self, v1: Visit, v2: Visit
    ) -> Optional[float]:
        """Compute cosine similarity between normalized color histograms."""
        if v1.color_histogram is None or v2.color_histogram is None:
            return None
        h1 = np.array(v1.color_histogram, dtype=np.float32)
        h2 = np.array(v2.color_histogram, dtype=np.float32)
        norm1 = np.linalg.norm(h1)
        norm2 = np.linalg.norm(h2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return None
        dot_product = float(np.dot(h1, h2) / (norm1 * norm2))
        return max(0.0, min(1.0, dot_product))
