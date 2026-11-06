"""Baseline synthesis, streaming pattern learning, and empirical visit scoring runner.

Simulates a 4-week diurnal traffic pattern (morning rush, lunch peak, evening taper,
minimal nocturnal volume) with realistic log-normal dwell distributions. Ingests
empirical visits from SQLite ('outputs/events.db') and scores them against learned
statistical baselines to flag lingering, off-hour breaches, and behavioral outliers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from navsecura.analytics import (
    StreamingVisitBaselines,
    generate_diurnal_rate,
    synthesize_historical_dataset,
)
from navsecura.store.sqlite import SQLiteEventStore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pattern_learning")


def generate_diurnal_rate(day_of_week: int, hour: int) -> float:
    """Compute realistic expected hourly visit rate (lambda) for CCTV entrance."""
    # Weekend (5=Sat, 6=Sun) has ~40% lower volume and shifted diurnal curve
    is_weekend = day_of_week in (5, 6)
    base_mult = 0.50 if is_weekend else 1.0

    if 0 <= hour <= 5:
        # Nocturnal lull (near zero traffic)
        rate = 0.05
    elif 6 <= hour <= 7:
        # Early opening
        rate = 2.0 * base_mult
    elif 8 <= hour <= 10:
        # Morning peak
        rate = (8.0 if not is_weekend else 3.0) * base_mult
    elif 11 <= hour <= 13:
        # Midday rush
        rate = (12.0 if not is_weekend else 7.0) * base_mult
    elif 14 <= hour <= 16:
        # Afternoon nominal
        rate = (6.0 if not is_weekend else 5.0) * base_mult
    elif 17 <= hour <= 19:
        # Evening exit peak
        rate = (10.0 if not is_weekend else 4.0) * base_mult
    elif 20 <= hour <= 21:
        # Evening taper
        rate = 2.5 * base_mult
    else:
        # Late night (22-23)
        rate = 0.40 * base_mult

    return max(0.02, rate)


def synthesize_historical_dataset(
    num_weeks: int = 4,
    roi_id: str = "roi_entrance",
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, int]], List[Dict[str, Any]]]:
    """Generate 4 weeks of synthetic visit events adhering to diurnal Poisson rates.

    Args:
        num_weeks: Number of weeks to simulate (default: 4 weeks).
        roi_id: Target Region of Interest identifier.
        random_seed: Deterministic random seed.

    Returns:
        Tuple of (historical_events, hourly_counts, synthetic_anomaly_events).
    """
    rng = random.Random(random_seed)
    np_rng = np.random.default_rng(random_seed)

    start_date = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_hours = num_weeks * 7 * 24

    events: List[Dict[str, Any]] = []
    hourly_counts: List[Tuple[int, int, int]] = []
    track_counter = 1000

    logger.info("Synthesizing %d weeks of historical events for '%s'...", num_weeks, roi_id)

    for h_idx in range(total_hours):
        current_dt = start_date + timedelta(hours=h_idx)
        dow = current_dt.weekday()
        hr = current_dt.hour

        lam = generate_diurnal_rate(dow, hr)
        num_visits = int(np_rng.poisson(lam))
        hourly_counts.append((dow, hr, num_visits))

        for v_idx in range(num_visits):
            # Arrival time uniformly distributed in the hour
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            visit_dt = current_dt.replace(minute=minute, second=second)

            # Dwell duration from Log-Normal (median 3.2s, sigma=0.35)
            dwell = float(np_rng.lognormal(mean=1.163, sigma=0.35))
            dwell = max(1.0, min(12.0, dwell))

            track_counter += 1
            events.append({
                "visit_id": f"hist_{track_counter}",
                "track_id": track_counter,
                "roi_id": roi_id,
                "entered_at": visit_dt.isoformat(),
                "exited_at": (visit_dt + timedelta(seconds=dwell)).isoformat(),
                "duration_s": round(dwell, 2),
                "mean_conf": round(float(np_rng.uniform(0.75, 0.95)), 3),
                "peak_conf": round(float(np_rng.uniform(0.85, 0.98)), 3),
                "frame_enter": int(v_idx * 100),
                "frame_exit": int(v_idx * 100 + round(dwell * 12.0)),
                "close_reason": "exited",
                "is_synthetic": True,
            })

    # Injected benchmark anomalies for testing sensitivity
    anomalies: List[Dict[str, Any]] = [
        # Anomaly 1: Nocturnal off-hours entry (3:15 AM on Tuesday)
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
        # Anomaly 2: Extreme loitering (25 minutes = 1500.0s in a 3s doorway)
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

    logger.info("Generated %d synthetic historical events + %d test anomalies.", len(events), len(anomalies))
    return events, hourly_counts, anomalies


def main() -> None:
    """Execute baseline synthesis, training, and empirical visit scoring."""
    parser = argparse.ArgumentParser(description="NavSecura Streaming Pattern Learning and Anomaly Miner")
    parser.add_argument("--db", type=str, default="outputs/events.db", help="Path to SQLite events database")
    parser.add_argument("--roi", type=str, default="roi_entrance", help="Target ROI identifier")
    parser.add_argument("--output", type=str, default="outputs/pattern_learning_meta.json", help="Telemetry JSON output path")
    parser.add_argument("--model-out", type=str, default="outputs/baseline_model.json", help="Saved baseline model JSON path")
    parser.add_argument("--weeks", "--history-weeks", dest="weeks", type=int, default=4, help="Historical weeks to simulate")
    parser.add_argument("--eval-hour", type=int, default=14, help="Representative diurnal hour (0-23) for daytime concourse video")
    parser.add_argument("--eval-day", type=int, default=1, help="Representative day of week (0=Mon, 6=Sun)")
    parser.add_argument("--use-recorded-timestamp", action="store_true", help="Strictly use machine execution timestamp")
    args = parser.parse_args()

    # 1. Synthesize 4-week diurnal historical stream
    hist_events, hourly_counts, synth_anomalies = synthesize_historical_dataset(num_weeks=args.weeks, roi_id=args.roi)

    # 2. Fit streaming baselines on historical stream in O(1) online mode
    baselines = StreamingVisitBaselines(half_life_days=28.0, dwell_z_threshold=2.0, hstree_threshold=0.65)

    # Ingest aggregated hourly volumes
    for dow, hr, count in hourly_counts:
        baselines.record_hourly_volume(dow, hr, count)

    # Ingest individual visit records into dwell and HS-Tree mass
    for evt in hist_events:
        baselines.update_dwell(evt["roi_id"], evt["duration_s"])
        feat = baselines._extract_feature_vector(evt)
        baselines.hs_ensemble.update(feat)

    # Ingest synthetic anomalies in evaluation mode
    scored_synthetic: List[Dict[str, Any]] = []
    for anom in synth_anomalies:
        res = baselines.ingest_visit(anom, is_reference=False)
        res["expected_anomaly"] = anom["expected_anomaly"]
        scored_synthetic.append(res)

    # 3. Ingest and score empirical visits from outputs/events.db
    db_path = Path(args.db)
    empirical_scored: List[Dict[str, Any]] = []

    if db_path.exists():
        logger.info("Reading empirical visits from '%s'...", db_path)
        with SQLiteEventStore(db_path) as store:
            db_visits = store.list_visits(roi_id=args.roi)

        eval_dt = datetime(2026, 10, 27, args.eval_hour, 30, 0, tzinfo=timezone.utc)

        for v in db_visits:
            v_copy = dict(v)
            if not args.use_recorded_timestamp:
                # Contextualize within daytime concourse baseline window
                v_copy["entered_at"] = eval_dt.isoformat()
            score_res = baselines.ingest_visit(v_copy, is_reference=False)
            empirical_scored.append(score_res)
    else:
        logger.warning("Database '%s' not found; skipping empirical scoring.", db_path)

    # 4. Save baseline model to JSON
    baselines.save_to_json(args.model_out)

    # 5. Compile telemetry report
    summary = baselines.get_summary()

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "simulation_label": f"Simulated {args.weeks}-Week Baseline Dataset (Non-Retraining)",
            "database_source": str(db_path),
            "roi_id": args.roi,
            "total_historical_reference_visits": len(hist_events),
            "total_empirical_scored_visits": len(empirical_scored),
        },
        "statistical_model_summary": summary,
        "empirical_visits_scored": empirical_scored,
        "synthetic_anomalies_evaluated": scored_synthetic,
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Exported pattern learning telemetry to %s", out_p)

    # Print summary table
    print("\n" + "=" * 90)
    print("NAVSECURA NON-RETRAINING PATTERN LEARNING & ANOMALY SCORING REPORT")
    print(f"Historical Reference: {len(hist_events)} events across {args.weeks} weeks | ROI: {args.roi}")
    print("=" * 90)

    dwell_info = summary["dwell_distributions"].get(args.roi, {})
    print(f"Learned Median Dwell:      {dwell_info.get('median_dwell_s', 0.0):.2f}s (Log-Mean: {dwell_info.get('log_mean', 0.0):.3f}, Log-StdDev: {dwell_info.get('log_std_dev', 0.0):.3f})")
    peak_info = summary["peak_traffic_period"]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"Peak Diurnal Period:       {dow_names[peak_info['day_of_week']]} @ {peak_info['hour']:02d}:00 (Expected Rate: {peak_info['expected_rate']:.1f} visits/hr)")
    print("-" * 90)
    print("EMPIRICAL VISIT SCORING (Sample Video Events vs. Learned Baselines):")
    fmt = "{:<10} {:<6} {:<10} {:<12} {:<12} {:<10} {:<18}"
    print(fmt.format("Visit ID", "Track", "Dwell (s)", "Expected (s)", "Dwell Z", "HSTree", "Classification"))
    print("-" * 90)

    for r in empirical_scored:
        print(fmt.format(
            r["visit_id"],
            str(r["track_id"]),
            f"{r['duration_s']:.2f}",
            f"{r['expected_median_dwell_s']:.2f}",
            f"{r['dwell_z_score']:+.2f}",
            f"{r['hstree_anomaly_score']:.3f}",
            f"[{r['anomaly_type']}]",
        ))

    print("-" * 90)
    print("SYNTHETIC SENSITIVITY TEST CASES:")
    for r in scored_synthetic:
        print(fmt.format(
            r["visit_id"][:10],
            str(r["track_id"]),
            f"{r['duration_s']:.2f}",
            f"{r['expected_median_dwell_s']:.2f}",
            f"{r['dwell_z_score']:+.2f}",
            f"{r['hstree_anomaly_score']:.3f}",
            f"[{r['anomaly_type']}]",
        ))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
