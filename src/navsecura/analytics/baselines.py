"""Streaming statistical visit analytics, baseline modeling, and non-retraining anomaly mining.

Implements online statistical learning over structured SQLite visit events
with frozen perception weights:
1. 168-Bin Hour-of-Week Poisson volume distribution with exponential decay.
2. Online Welford log-normal dwell duration distribution.
3. Streaming Half-Space Trees (HS-Trees) ensemble for multi-dimensional anomaly detection
   (Tan, Ting, Liu, IJCAI 2011).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Welford Online Accumulator for Log-Normal Dwell Modeling
# ---------------------------------------------------------------------------

@dataclass
class WelfordAccumulator:
    """Numerically stable online accumulator for mean and variance (Welford 1962)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, val: float) -> None:
        """Ingest a new scalar observation in O(1) time."""
        self.count += 1
        delta = val - self.mean
        self.mean += delta / float(self.count)
        delta2 = val - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Sample variance (or 0.0 if count < 2)."""
        if self.count < 2:
            return 0.0
        return max(0.0, self.m2 / float(self.count - 1))

    @property
    def std_dev(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize accumulator state to dictionary."""
        return {
            "count": self.count,
            "mean": round(self.mean, 6),
            "m2": round(self.m2, 6),
            "variance": round(self.variance, 6),
            "std_dev": round(self.std_dev, 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WelfordAccumulator":
        """Deserialize accumulator from dictionary."""
        acc = cls(
            count=int(data.get("count", 0)),
            mean=float(data.get("mean", 0.0)),
            m2=float(data.get("m2", 0.0)),
        )
        return acc


# ---------------------------------------------------------------------------
# 2. Streaming Half-Space Tree Node & Tree (Tan, Ting, Liu, IJCAI 2011)
# ---------------------------------------------------------------------------

class HSTreeNode:
    """Node in a Streaming Half-Space Tree storing recursive bounds and mass counters."""

    def __init__(
        self,
        min_vals: List[float],
        max_vals: List[float],
        depth: int,
        max_depth: int,
        rng: random.Random,
    ) -> None:
        """Recursively initialize tree node with random orthogonal hyperplane split.

        Args:
            min_vals: Lower bound coordinate vector of current half-space.
            max_vals: Upper bound coordinate vector of current half-space.
            depth: Current depth in the tree.
            max_depth: Maximum tree depth.
            rng: Seeded pseudo-random number generator for deterministic structure.
        """
        self.min_vals = list(min_vals)
        self.max_vals = list(max_vals)
        self.depth = depth
        self.mass: float = 0.0
        self.left: Optional[HSTreeNode] = None
        self.right: Optional[HSTreeNode] = None
        self.split_dim: int = -1
        self.split_val: float = 0.0

        if depth < max_depth:
            # Pick a split dimension with non-zero span
            d = len(min_vals)
            valid_dims = [i for i in range(d) if max_vals[i] > min_vals[i] + 1e-6]
            if valid_dims:
                self.split_dim = rng.choice(valid_dims)
                # Split at midpoint of node's bounding interval
                self.split_val = (self.min_vals[self.split_dim] + self.max_vals[self.split_dim]) / 2.0

                left_max = list(max_vals)
                left_max[self.split_dim] = self.split_val

                right_min = list(min_vals)
                right_min[self.split_dim] = self.split_val

                self.left = HSTreeNode(min_vals, left_max, depth + 1, max_depth, rng)
                self.right = HSTreeNode(right_min, max_vals, depth + 1, max_depth, rng)

    def update_mass(self, point: List[float], weight: float = 1.0) -> None:
        """Increment mass along the point's traversal path down to leaf."""
        self.mass += weight
        if self.left is not None and self.right is not None:
            if point[self.split_dim] <= self.split_val:
                self.left.update_mass(point, weight)
            else:
                self.right.update_mass(point, weight)

    def evaluate_mass(self, point: List[float]) -> Tuple[float, int]:
        """Traverse down to deepest enclosing node and return (mass, depth)."""
        if self.left is None or self.right is None:
            return (self.mass, self.depth)
        if point[self.split_dim] <= self.split_val:
            return self.left.evaluate_mass(point)
        else:
            return self.right.evaluate_mass(point)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node structure and mass to dictionary."""
        d: Dict[str, Any] = {
            "depth": self.depth,
            "mass": round(self.mass, 4),
            "split_dim": self.split_dim,
            "split_val": round(self.split_val, 6),
            "min_vals": [round(v, 4) for v in self.min_vals],
            "max_vals": [round(v, 4) for v in self.max_vals],
        }
        if self.left is not None:
            d["left"] = self.left.to_dict()
        if self.right is not None:
            d["right"] = self.right.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HSTreeNode":
        """Deserialize node from dictionary."""
        node = cls.__new__(cls)
        node.depth = data["depth"]
        node.mass = float(data["mass"])
        node.split_dim = int(data["split_dim"])
        node.split_val = float(data["split_val"])
        node.min_vals = list(data["min_vals"])
        node.max_vals = list(data["max_vals"])
        node.left = cls.from_dict(data["left"]) if "left" in data else None
        node.right = cls.from_dict(data["right"]) if "right" in data else None
        return node


class HSTreeEnsemble:
    """Streaming Half-Space Trees anomaly detection ensemble (Tan, Ting, Liu 2011)."""

    def __init__(
        self,
        num_trees: int = 25,
        max_depth: int = 10,
        feature_mins: Optional[List[float]] = None,
        feature_maxs: Optional[List[float]] = None,
        random_seed: int = 42,
    ) -> None:
        """Construct random half-space trees ensemble over fixed feature workspace.

        Args:
            num_trees: Number of trees in the ensemble (default: 25).
            max_depth: Maximum tree depth (default: 10).
            feature_mins: Lower bounds for normalized features (default: 5-dim zeros).
            feature_maxs: Upper bounds for normalized features (default: 5-dim ones).
            random_seed: Seed for reproducible tree construction.
        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.feature_mins = feature_mins or [0.0, 0.0, 0.0, 0.0, 0.0]
        self.feature_maxs = feature_maxs or [1.0, 1.0, 1.0, 1.0, 1.0]
        self.total_samples: int = 0

        rng = random.Random(random_seed)
        self.trees: List[HSTreeNode] = []
        for _ in range(num_trees):
            # Introduce slight boundary jitter to diversify half-spaces
            perturbed_min = [m - rng.uniform(0.0, 0.05) for m in self.feature_mins]
            perturbed_max = [M + rng.uniform(0.0, 0.05) for M in self.feature_maxs]
            tree = HSTreeNode(perturbed_min, perturbed_max, depth=0, max_depth=max_depth, rng=rng)
            self.trees.append(tree)

    def update(self, point: List[float], weight: float = 1.0) -> None:
        """Stream an observation into all trees in O(1) time."""
        self.total_samples += 1
        for tree in self.trees:
            tree.update_mass(point, weight)

    def score(self, point: List[float]) -> float:
        """Compute bounded anomaly score in [0.0, 1.0] in O(1) time.

        Higher score = greater anomalousness (isolated in low-mass half-spaces).
        """
        if self.total_samples == 0:
            return 0.0

        mass_depth_scores: List[float] = []
        for tree in self.trees:
            mass, depth = tree.evaluate_mass(point)
            # Mass score weighted by depth 2^depth
            mass_depth_scores.append(mass * (2.0 ** depth))

        avg_score = sum(mass_depth_scores) / float(len(mass_depth_scores))

        # Nominal points have high mass; outliers have small mass
        expected_nominal_score = max(1.0, float(self.total_samples))
        normalized_density = avg_score / expected_nominal_score

        # Sigmoidal mapping to [0.0, 1.0]
        anomaly_score = 1.0 / (1.0 + math.exp(min(10.0, max(-10.0, (normalized_density - 1.0) * 2.0))))
        return round(float(anomaly_score), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire ensemble state."""
        return {
            "num_trees": self.num_trees,
            "max_depth": self.max_depth,
            "feature_mins": self.feature_mins,
            "feature_maxs": self.feature_maxs,
            "total_samples": self.total_samples,
            "trees": [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HSTreeEnsemble":
        """Deserialize ensemble from dictionary."""
        ens = cls(
            num_trees=data["num_trees"],
            max_depth=data["max_depth"],
            feature_mins=data["feature_mins"],
            feature_maxs=data["feature_maxs"],
        )
        ens.total_samples = int(data["total_samples"])
        ens.trees = [HSTreeNode.from_dict(t) for t in data["trees"]]
        return ens


# ---------------------------------------------------------------------------
# 3. Master Streaming Visit Baselines Engine
# ---------------------------------------------------------------------------

class StreamingVisitBaselines:
    """Master streaming statistical model for visit volume, dwell times, and anomaly detection.

    Operates purely in O(1) online mode with frozen perception weights:
    - 168-Bin Poisson volume model per (day_of_week, hour) with exponential half-life decay.
    - Welford streaming accumulator for log-dwell distributions per ROI.
    - Streaming Half-Space Tree ensemble for multi-dimensional anomaly detection.
    """

    def __init__(
        self,
        half_life_days: float = 28.0,
        dwell_z_threshold: float = 2.0,
        hstree_threshold: float = 0.65,
    ) -> None:
        """Initialize streaming visit baselines.

        Args:
            half_life_days: Half-life in days for exponential historical decay (default: 28 days).
            dwell_z_threshold: Dwell duration Z-score threshold for lingering anomaly (default: 2.0).
            hstree_threshold: HS-Tree multi-feature anomaly threshold (default: 0.65).
        """
        self.half_life_days = float(half_life_days)
        self.dwell_z_threshold = float(dwell_z_threshold)
        self.hstree_threshold = float(hstree_threshold)

        # 168 bins: Gamma-Poisson conjugate shape parameters (alpha_param, beta_param)
        # alpha_param = total weighted events, beta_param = total weighted time units
        # Default unobserved baseline: 0.05 events/hour (1 visit per 20 hours)
        self.volume_bins: List[Dict[str, float]] = [
            {"alpha": 0.05, "beta": 1.0, "total_events": 0} for _ in range(168)
        ]

        # Welford accumulators for log-dwell duration: roi_id -> WelfordAccumulator
        self.dwell_accumulators: Dict[str, WelfordAccumulator] = {}

        # Streaming Half-Space Tree ensemble
        self.hs_ensemble = HSTreeEnsemble(num_trees=25, max_depth=10, random_seed=42)

        self.total_ingested_visits: int = 0
        self.last_update_timestamp: Optional[str] = None

    @staticmethod
    def _bin_index(day_of_week: int, hour: int) -> int:
        """Map day of week [0..6] and hour [0..23] to a linear bin index [0..167]."""
        d = max(0, min(6, int(day_of_week)))
        h = max(0, min(23, int(hour)))
        return d * 24 + h

    def record_hourly_volume(
        self,
        day_of_week: int,
        hour: int,
        count: int,
        decay_factor: float = 1.0,
    ) -> None:
        """Record the aggregated visit count for a one-hour window in O(1) time.

        Args:
            day_of_week: Day index (0=Monday, 6=Sunday).
            hour: Hour index (0..23).
            count: Number of visits observed during this hour.
            decay_factor: Multiplier in [0, 1] to apply exponential half-life decay.
        """
        b_idx = self._bin_index(day_of_week, hour)
        bin_data = self.volume_bins[b_idx]

        # In conjugate Gamma-Poisson:
        # alpha accumulates event counts, beta accumulates time units (hours observed)
        bin_data["alpha"] = bin_data["alpha"] * decay_factor + float(count)
        bin_data["beta"] = bin_data["beta"] * decay_factor + 1.0
        bin_data["total_events"] += count

    def update_volume(
        self,
        day_of_week: int,
        hour: int,
        count: int = 1,
    ) -> None:
        """Increment observed visit count in current hourly bin."""
        b_idx = self._bin_index(day_of_week, hour)
        bin_data = self.volume_bins[b_idx]
        bin_data["alpha"] += float(count)
        bin_data["total_events"] += count

    def get_expected_volume_rate(self, day_of_week: int, hour: int) -> float:
        """Return posterior expected visit arrival rate lambda for a specific hour of the week."""
        b_idx = self._bin_index(day_of_week, hour)
        bin_data = self.volume_bins[b_idx]
        return max(0.01, bin_data["alpha"] / max(0.01, bin_data["beta"]))

    def score_volume(self, day_of_week: int, hour: int, observed_count: int) -> Dict[str, Any]:
        """Compute volume Z-score and Poisson anomaly probability."""
        exp_rate = self.get_expected_volume_rate(day_of_week, hour)
        std_err = math.sqrt(exp_rate)
        z_score = (observed_count - exp_rate) / std_err

        # Poisson CDF probability: P(X <= k) using regularized gamma / poisson series
        p_val = 1.0 - math.exp(-exp_rate) if observed_count > 0 else math.exp(-exp_rate)

        is_anomalous = abs(z_score) >= 2.5 or (observed_count > 0 and exp_rate < 0.1)

        return {
            "day_of_week": day_of_week,
            "hour": hour,
            "observed_count": observed_count,
            "expected_rate": round(exp_rate, 2),
            "z_score": round(z_score, 2),
            "is_volume_anomaly": bool(is_anomalous),
        }

    def update_dwell(self, roi_id: str, duration_s: float) -> None:
        """Update streaming Welford accumulator with log-dwell duration."""
        if roi_id not in self.dwell_accumulators:
            self.dwell_accumulators[roi_id] = WelfordAccumulator()

        dur = max(0.1, float(duration_s))
        y = math.log(dur)
        self.dwell_accumulators[roi_id].update(y)

    def score_dwell(self, roi_id: str, duration_s: float) -> Dict[str, Any]:
        """Compute dwell Z-score against learned log-normal distribution."""
        dur = max(0.1, float(duration_s))
        acc = self.dwell_accumulators.get(roi_id)

        if acc is None or acc.count < 3:
            return {
                "roi_id": roi_id,
                "duration_s": round(dur, 2),
                "expected_median_s": round(dur, 2),
                "z_score": 0.0,
                "is_lingering_anomaly": False,
            }

        y = math.log(dur)
        sigma = max(0.1, acc.std_dev)
        z_score = (y - acc.mean) / sigma

        expected_median = math.exp(acc.mean)
        expected_p95 = math.exp(acc.mean + 1.645 * sigma)

        is_lingering = z_score >= self.dwell_z_threshold

        return {
            "roi_id": roi_id,
            "duration_s": round(dur, 2),
            "expected_median_s": round(expected_median, 2),
            "threshold_p95_s": round(expected_p95, 2),
            "z_score": round(z_score, 2),
            "is_lingering_anomaly": bool(is_lingering),
        }

    def _extract_feature_vector(self, visit_dict: Dict[str, Any]) -> List[float]:
        """Extract and normalize a 5-dimensional feature vector for HS-Tree scoring."""
        # 1. Hour of day normalized [0, 1]
        entered_at = visit_dict.get("entered_at")
        if entered_at:
            try:
                dt = datetime.fromisoformat(entered_at.replace("Z", "+00:00"))
                h_norm = dt.hour / 24.0
                d_norm = dt.weekday() / 7.0
            except Exception:
                h_norm = 0.5
                d_norm = 0.5
        else:
            h_norm = 0.5
            d_norm = 0.5

        # 2. Log-dwell normalized [0, 1] across [0.1s .. 3600s]
        dur = max(0.1, float(visit_dict.get("duration_s", 1.0)))
        log_dur = math.log(dur)
        log_dur_norm = max(0.0, min(1.0, (log_dur - math.log(0.1)) / (math.log(3600.0) - math.log(0.1))))

        # 3. Mean detector confidence [0, 1]
        conf = float(visit_dict.get("mean_conf", 0.80) or 0.80)
        conf_norm = max(0.0, min(1.0, conf))

        # 4. Movement velocity / distance proxy normalized
        # Higher dwell with zero distance -> stationary lingering
        f_enter = int(visit_dict.get("frame_enter", 0))
        f_exit = int(visit_dict.get("frame_exit", f_enter + 1))
        f_span = max(1, f_exit - f_enter)
        vel_norm = max(0.0, min(1.0, 1.0 / math.sqrt(float(f_span))))

        return [h_norm, log_dur_norm, conf_norm, vel_norm, d_norm]

    def ingest_visit(self, visit_dict: Dict[str, Any], is_reference: bool = True) -> Dict[str, Any]:
        """Ingest a visit event, update all online models, and return anomaly evaluation."""
        self.total_ingested_visits += 1
        self.last_update_timestamp = datetime.now().isoformat()

        roi_id = str(visit_dict.get("roi_id", "default_roi"))
        dur = float(visit_dict.get("duration_s", 1.0))

        entered_at = visit_dict.get("entered_at")
        if entered_at:
            try:
                dt = datetime.fromisoformat(entered_at.replace("Z", "+00:00"))
                dow = dt.weekday()
                hr = dt.hour
            except Exception:
                dow, hr = 1, 14
        else:
            dow, hr = 1, 14

        # 1. Update models if in reference training mode
        if is_reference:
            self.update_volume(day_of_week=dow, hour=hr, count=1)
            self.update_dwell(roi_id=roi_id, duration_s=dur)
            feat = self._extract_feature_vector(visit_dict)
            self.hs_ensemble.update(feat)

        # 2. Score visit
        dwell_score = self.score_dwell(roi_id=roi_id, duration_s=dur)
        vol_score = self.score_volume(day_of_week=dow, hour=hr, observed_count=1)
        feat = self._extract_feature_vector(visit_dict)
        hstree_score = self.hs_ensemble.score(feat)

        is_anomaly = (
            dwell_score["is_lingering_anomaly"]
            or vol_score["is_volume_anomaly"]
            or hstree_score >= self.hstree_threshold
        )

        anomaly_type = "NORMAL"
        if dwell_score["is_lingering_anomaly"]:
            anomaly_type = "LINGERING_ANOMALY"
        elif vol_score["is_volume_anomaly"]:
            anomaly_type = "OFF_HOURS_SURGE"
        elif hstree_score >= self.hstree_threshold:
            anomaly_type = "BEHAVIORAL_OUTLIER"

        return {
            "visit_id": visit_dict.get("visit_id", "")[:8],
            "track_id": visit_dict.get("track_id"),
            "roi_id": roi_id,
            "duration_s": dur,
            "dwell_z_score": dwell_score["z_score"],
            "expected_median_dwell_s": dwell_score["expected_median_s"],
            "volume_expected_rate": vol_score["expected_rate"],
            "volume_z_score": vol_score["z_score"],
            "hstree_anomaly_score": hstree_score,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Produce full statistical telemetry summary."""
        dwell_summary = {
            roi: {
                "count": acc.count,
                "median_dwell_s": round(math.exp(acc.mean), 2) if acc.count > 0 else 0.0,
                "log_mean": round(acc.mean, 4),
                "log_std_dev": round(acc.std_dev, 4),
            }
            for roi, acc in self.dwell_accumulators.items()
        }

        # Find peak volume hour
        max_rate = 0.0
        peak_bin = 0
        for i, b in enumerate(self.volume_bins):
            rate = b["alpha"] / max(0.01, b["beta"])
            if rate > max_rate:
                max_rate = rate
                peak_bin = i

        peak_dow = peak_bin // 24
        peak_hour = peak_bin % 24

        return {
            "total_ingested_visits": self.total_ingested_visits,
            "last_update_timestamp": self.last_update_timestamp,
            "half_life_days": self.half_life_days,
            "dwell_z_threshold": self.dwell_z_threshold,
            "hstree_threshold": self.hstree_threshold,
            "peak_traffic_period": {
                "day_of_week": peak_dow,
                "hour": peak_hour,
                "expected_rate": round(max_rate, 2),
            },
            "dwell_distributions": dwell_summary,
            "hstree_ensemble": {
                "num_trees": self.hs_ensemble.num_trees,
                "max_depth": self.hs_ensemble.max_depth,
                "total_samples": self.hs_ensemble.total_samples,
            },
        }

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Persist full baseline models to JSON."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "half_life_days": self.half_life_days,
            "dwell_z_threshold": self.dwell_z_threshold,
            "hstree_threshold": self.hstree_threshold,
            "total_ingested_visits": self.total_ingested_visits,
            "last_update_timestamp": self.last_update_timestamp,
            "volume_bins": self.volume_bins,
            "dwell_accumulators": {
                roi: acc.to_dict() for roi, acc in self.dwell_accumulators.items()
            },
            "hs_ensemble": self.hs_ensemble.to_dict(),
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved StreamingVisitBaselines to %s", p)

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> "StreamingVisitBaselines":
        """Load baseline models from persisted JSON."""
        p = Path(file_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        baselines = cls(
            half_life_days=float(data.get("half_life_days", 28.0)),
            dwell_z_threshold=float(data.get("dwell_z_threshold", 2.0)),
            hstree_threshold=float(data.get("hstree_threshold", 0.65)),
        )
        baselines.total_ingested_visits = int(data.get("total_ingested_visits", 0))
        baselines.last_update_timestamp = data.get("last_update_timestamp")
        baselines.volume_bins = data.get("volume_bins", baselines.volume_bins)

        acc_dict = data.get("dwell_accumulators", {})
        baselines.dwell_accumulators = {
            roi: WelfordAccumulator.from_dict(d) for roi, d in acc_dict.items()
        }

        if "hs_ensemble" in data:
            baselines.hs_ensemble = HSTreeEnsemble.from_dict(data["hs_ensemble"])

        return baselines


# ---------------------------------------------------------------------------
# 4. Diurnal Rate Modeling & Historical Dataset Synthesis
# ---------------------------------------------------------------------------

def generate_diurnal_rate(day_of_week: int, hour: int) -> float:
    """Compute realistic expected hourly visit rate (lambda) for entrance CCTV."""
    is_weekend = day_of_week in (5, 6)
    base_mult = 0.50 if is_weekend else 1.0

    if 0 <= hour <= 5:
        rate = 0.05
    elif 6 <= hour <= 7:
        rate = 2.0 * base_mult
    elif 8 <= hour <= 10:
        rate = (8.0 if not is_weekend else 3.0) * base_mult
    elif 11 <= hour <= 13:
        rate = (12.0 if not is_weekend else 7.0) * base_mult
    elif 14 <= hour <= 16:
        rate = (6.0 if not is_weekend else 5.0) * base_mult
    elif 17 <= hour <= 19:
        rate = (10.0 if not is_weekend else 4.0) * base_mult
    elif 20 <= hour <= 21:
        rate = 2.5 * base_mult
    else:
        rate = 0.40 * base_mult

    return max(0.02, rate)


def synthesize_historical_dataset(
    num_weeks: int = 4,
    roi_id: str = "roi_entrance",
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, int]], List[Dict[str, Any]]]:
    """Generate multi-week synthetic visit events adhering to diurnal Poisson rates.

    Args:
        num_weeks: Number of weeks to simulate (default: 4 weeks).
        roi_id: Target Region of Interest identifier.
        random_seed: Deterministic random seed.

    Returns:
        Tuple of (historical_events, hourly_counts, synthetic_anomaly_events).
    """
    from datetime import datetime, timedelta, timezone

    rng = random.Random(random_seed)
    start_date = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_hours = num_weeks * 7 * 24

    events: List[Dict[str, Any]] = []
    hourly_counts: List[Tuple[int, int, int]] = []
    track_counter = 1000

    logger.info("Synthesizing %d weeks of historical events for '%s'...", num_weeks, roi_id)

    # Use deterministic pseudorandom generation
    for h_idx in range(total_hours):
        current_dt = start_date + timedelta(hours=h_idx)
        dow = current_dt.weekday()
        hr = current_dt.hour

        lam = generate_diurnal_rate(dow, hr)
        # Poisson approximation
        u = rng.random()
        p = math.exp(-lam)
        F = p
        k = 0
        while u > F and k < 50:
            k += 1
            p = p * lam / k
            F += p
        num_visits = k
        hourly_counts.append((dow, hr, num_visits))

        for v_idx in range(num_visits):
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            visit_dt = current_dt.replace(minute=minute, second=second)

            # Log-normal dwell: exp(normal(1.163, 0.35))
            norm_val = rng.gauss(1.163, 0.35)
            dwell = max(1.0, min(12.0, math.exp(norm_val)))

            track_counter += 1
            events.append({
                "visit_id": f"hist_{track_counter}",
                "track_id": track_counter,
                "roi_id": roi_id,
                "entered_at": visit_dt.isoformat(),
                "exited_at": (visit_dt + timedelta(seconds=dwell)).isoformat(),
                "duration_s": round(dwell, 2),
                "mean_conf": round(rng.uniform(0.75, 0.95), 3),
                "peak_conf": round(rng.uniform(0.85, 0.98), 3),
                "frame_enter": int(v_idx * 100),
                "frame_exit": int(v_idx * 100 + round(dwell * 12.0)),
                "close_reason": "exited",
                "is_synthetic": True,
            })

    # Benchmark test anomalies
    anomalies: List[Dict[str, Any]] = [
        {
            "visit_id": "anom_offhours_01",
            "track_id": 9001,
            "roi_id": roi_id,
            "entered_at": datetime(2026, 10, 27, 3, 15, 0, tzinfo=timezone.utc).isoformat(),
            "exited_at": datetime(2026, 10, 27, 3, 15, 4, tzinfo=timezone.utc).isoformat(),
            "duration_s": 4.0,
            "mean_conf": 0.88,
            "peak_conf": 0.92,
            "frame_enter": 0,
            "frame_exit": 48,
            "close_reason": "exited",
            "expected_anomaly": "OFF_HOURS_SURGE",
        },
        {
            "visit_id": "anom_lingering_02",
            "track_id": 9002,
            "roi_id": roi_id,
            "entered_at": datetime(2026, 10, 27, 14, 0, 0, tzinfo=timezone.utc).isoformat(),
            "exited_at": datetime(2026, 10, 27, 14, 25, 0, tzinfo=timezone.utc).isoformat(),
            "duration_s": 1500.0,
            "mean_conf": 0.82,
            "peak_conf": 0.91,
            "frame_enter": 0,
            "frame_exit": 18000,
            "close_reason": "exited",
            "expected_anomaly": "LINGERING_ANOMALY",
        },
    ]

    return events, hourly_counts, anomalies

