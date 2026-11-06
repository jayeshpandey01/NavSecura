---
name: pattern-learning-miner
description: Implements streaming visit baselines, hour-of-week count models, EWMA dwell distributions, and anomaly scoring over emitted events without neural network retraining. Use when developing analytics, baseline metrics, or addressing the non-retraining pattern learning requirement.
---

# Streaming Pattern Miner

## Goal
Demonstrate how an operational video intelligence system extracts recurring behavioral patterns, establishes normal activity baselines, and flags anomalous visits without daily deep learning retraining.

## Instructions
1. **Principle:** Keep perception weights frozen. Treat detections and tracking as a sensor; perform statistical learning directly on structured event records.
2. **Hour-of-Week Visit Volume Baselines:**
   - Partition continuous time into 168 hourly bins `(day_of_week, hour)`.
   - Maintain running visit count statistics using an online Gamma-Poisson update or exponential moving average (EWMA).
   - Compute Z-scores for incoming hourly visit volumes to detect unusual spikes or lulls.
3. **Dwell Time Distribution Modeling:**
   - Maintain running mean and variance of log-dwell times (`log(duration_s)`) per ROI using Welford's streaming algorithm.
   - Flag visits whose dwell duration exceeds the 95th percentile ($\mu + 2\sigma$) as lingering anomalies.
4. **Streaming Multi-Dimensional Anomaly Detection (Half-Space Trees):**
   - Stream event feature vectors: `[hour_of_day, duration_s, mean_confidence, entry_velocity]`.
   - Utilize a Streaming Half-Space Tree (HS-Tree) ensemble to compute mass-based anomaly scores in $O(1)$ time and constant memory.
5. **Drift Adaptation:**
   - Apply half-life decay weights to historical counts to adapt to gradual seasonal shifts without manual intervention.

## Examples
- Run baseline calculation: `python src/navsecura/analytics/baselines.py --db outputs/events.db`
- Anomaly scoring: evaluate a 45-minute dwell visit against historical distribution and emit anomaly flag.

## Constraints
- Do not invoke neural network backpropagation or weight fine-tuning in this module.
- All streaming algorithms must operate strictly in online mode ($O(1)$ time complexity per event).
- Explicitly document that evaluation on short test videos utilizes simulated historical seeds.

