"""Unit tests for streaming visit baselines, Welford log-dwell, and HS-Trees anomaly detection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from navsecura.analytics.baselines import (
    HSTreeEnsemble,
    StreamingVisitBaselines,
    WelfordAccumulator,
    generate_diurnal_rate,
    synthesize_historical_dataset,
)


def test_welford_accumulator_numerical_stability() -> None:
    """Verify Welford streaming accumulator matches numpy sample variance on random stream."""
    acc = WelfordAccumulator()
    assert acc.count == 0
    assert acc.variance == 0.0
    assert acc.std_dev == 0.0

    # Stream 1,000 synthetic log-durations
    rng = np.random.default_rng(12345)
    samples = rng.normal(loc=1.163, scale=0.35, size=1000).tolist()

    for x in samples:
        acc.update(x)

    np_mean = float(np.mean(samples))
    np_var = float(np.var(samples, ddof=1))
    np_std = float(np.std(samples, ddof=1))

    assert acc.count == 1000
    assert pytest.approx(acc.mean, abs=1e-5) == np_mean
    assert pytest.approx(acc.variance, abs=1e-5) == np_var
    assert pytest.approx(acc.std_dev, abs=1e-5) == np_std

    # Dictionary serialization roundtrip
    d = acc.to_dict()
    restored = WelfordAccumulator.from_dict(d)
    assert restored.count == acc.count
    assert pytest.approx(restored.mean, abs=1e-5) == acc.mean
    assert pytest.approx(restored.variance, abs=1e-5) == acc.variance


def test_poisson_volume_bins_and_z_scores() -> None:
    """Verify 168-bin Poisson volume tracking, expected arrival rates, and Z-scores."""
    baselines = StreamingVisitBaselines()

    # Verify unobserved off-hour rate
    rate_off = baselines.get_expected_volume_rate(day_of_week=0, hour=3)
    assert rate_off < 0.10

    # Record 4 weeks of observations for Monday 9:00 AM (e.g. 10 visits/hr)
    for _ in range(4):
        baselines.record_hourly_volume(day_of_week=0, hour=9, count=10)

    exp_rate = baselines.get_expected_volume_rate(day_of_week=0, hour=9)
    # 4 observations with 10 events each -> alpha ~= 40, beta ~= 4 -> rate ~= 8-10
    assert exp_rate > 7.0

    # Normal observation should have low Z-score
    normal_score = baselines.score_volume(day_of_week=0, hour=9, observed_count=9)
    assert abs(normal_score["z_score"]) < 1.0
    assert not normal_score["is_volume_anomaly"]

    # Massive surge (50 visits in an hour where 8-10 expected)
    surge_score = baselines.score_volume(day_of_week=0, hour=9, observed_count=50)
    assert surge_score["z_score"] > 3.0
    assert surge_score["is_volume_anomaly"]

    # Off-hours entry at 3:00 AM
    off_score = baselines.score_volume(day_of_week=0, hour=3, observed_count=1)
    assert off_score["is_volume_anomaly"]


def test_dwell_distribution_and_lingering_detection() -> None:
    """Verify log-normal dwell time modeling and linger anomaly detection."""
    baselines = StreamingVisitBaselines(dwell_z_threshold=2.0)
    roi_id = "test_zone"

    # Ingest 200 nominal doorway crossings (median ~3.0s)
    rng = np.random.default_rng(42)
    durations = rng.lognormal(mean=1.10, sigma=0.25, size=200)

    for dur in durations:
        baselines.update_dwell(roi_id=roi_id, duration_s=float(dur))

    # Evaluate nominal 3.1s transit
    nom_eval = baselines.score_dwell(roi_id=roi_id, duration_s=3.1)
    assert not nom_eval["is_lingering_anomaly"]
    assert abs(nom_eval["z_score"]) < 1.5

    # Evaluate lingering subject (300s = 5 minutes in a 3s doorway)
    lingering_eval = baselines.score_dwell(roi_id=roi_id, duration_s=300.0)
    assert lingering_eval["is_lingering_anomaly"]
    assert lingering_eval["z_score"] > 3.0


def test_hstree_ensemble_mass_and_anomaly_ranking() -> None:
    """Verify Streaming Half-Space Trees mass updates and anomaly score ranking."""
    ensemble = HSTreeEnsemble(num_trees=20, max_depth=8, random_seed=99)
    assert ensemble.total_samples == 0

    # Stream 300 nominal cluster points around [0.5, 0.5, 0.8, 0.2, 0.5]
    rng = np.random.default_rng(77)
    for _ in range(300):
        pt = [
            float(rng.normal(0.5, 0.05)),
            float(rng.normal(0.5, 0.05)),
            float(rng.normal(0.8, 0.05)),
            float(rng.normal(0.2, 0.05)),
            float(rng.normal(0.5, 0.05)),
        ]
        # Clamp to [0, 1]
        clamped = [max(0.0, min(1.0, val)) for val in pt]
        ensemble.update(clamped)

    assert ensemble.total_samples == 300

    # Score nominal point near cluster center
    nominal_pt = [0.5, 0.5, 0.8, 0.2, 0.5]
    nom_score = ensemble.score(nominal_pt)

    # Score extreme outlier point
    outlier_pt = [0.05, 0.99, 0.1, 0.95, 0.05]
    outlier_score = ensemble.score(outlier_pt)

    # Outlier must receive strictly higher anomaly score than nominal point
    assert outlier_score > nom_score
    assert outlier_score > 0.50


def test_exponential_decay_adaptation() -> None:
    """Verify exponential half-life decay adapts to shifts in arrival rate."""
    baselines = StreamingVisitBaselines(half_life_days=28.0)

    # Initial regime: 20 visits/hr for 4 weeks
    for _ in range(4):
        baselines.record_hourly_volume(day_of_week=2, hour=12, count=20)
    initial_rate = baselines.get_expected_volume_rate(day_of_week=2, hour=12)
    assert initial_rate > 15.0

    # New regime: store volume drops to 2 visits/hr with decay factor 0.80
    for _ in range(8):
        baselines.record_hourly_volume(day_of_week=2, hour=12, count=2, decay_factor=0.80)

    adapted_rate = baselines.get_expected_volume_rate(day_of_week=2, hour=12)
    # Rate should have decayed significantly toward 2.0
    assert adapted_rate < initial_rate / 2.0


def test_baseline_json_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify full baseline model serialization to/from JSON."""
    out_file = tmp_path / "test_model.json"

    baselines = StreamingVisitBaselines()
    baselines.record_hourly_volume(day_of_week=1, hour=14, count=8)
    baselines.update_dwell(roi_id="zone_01", duration_s=4.5)
    baselines.save_to_json(out_file)

    assert out_file.exists()
    restored = StreamingVisitBaselines.load_from_json(out_file)

    assert restored.half_life_days == baselines.half_life_days
    assert "zone_01" in restored.dwell_accumulators
    assert restored.dwell_accumulators["zone_01"].count == 1

