---
name: cv-tracker-evaluator
description: Runs and benchmarks computer vision detection and multi-object tracking pipelines (ByteTrack vs BoT-SORT). Computes FPS, latency, and visit-level evaluation metrics (Precision, Recall, Fragmentation Rate, Duration Error). Use whenever running model inference, tracker comparisons, or evaluation sweeps in the NavSecura assessment.
---

# CV Tracker Evaluator

## Goal
Provide rigorous, reproducible evaluation of detection and tracking pipelines, comparing ByteTrack and BoT-SORT(+ReID) on both frame-level tracking efficiency and visit-level event accuracy.

## Instructions
1. Inspect the input video stream properties (resolution, FPS, total frames).
2. Run tracking under fixed configuration seeds:
   - Run A: ByteTrack (person-only detection, low-score association enabled).
   - Run B: BoT-SORT (motion compensation + ReID enabled if compute permits).
3. Log runtime telemetry: detector inference latency, tracker update latency, total FPS, and GPU/CPU memory consumption.
4. Load hand-annotated ground truth visits from `eval/ground_truth.csv` (schema: `person_id, t_enter_s, t_exit_s`).
5. Match predicted visits against ground truth using temporal Intersection over Union (t-IoU >= 0.5):
   - **Visit Precision:** `matched_visits / total_predicted_visits`
   - **Visit Recall:** `matched_visits / total_ground_truth_visits`
   - **Fragmentation Rate:** `total_predicted_visits / total_ground_truth_visits` (Ideal: 1.0)
   - **Duration MAE:** Mean absolute difference in dwell duration (seconds).
   - **Event Reduction Ratio:** `total_raw_person_detections / total_emitted_visits`
6. Output formatted comparison tables and save structured results to `eval/results.json`.

## Examples
- Run comparative evaluation: `python eval/evaluate.py --video data/sample.mp4 --ground-truth eval/ground_truth.csv`
- Compute metrics for ByteTrack vs. BoT-SORT: output comparative markdown table detailing precision, recall, fragmentation, and FPS.

## Constraints
- Never fabricate FPS, latency, or accuracy numbers.
- Maintain identical detection confidence thresholds when comparing trackers.
- Always report hardware specifications alongside FPS measurements.

