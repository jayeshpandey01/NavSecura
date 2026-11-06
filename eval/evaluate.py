"""Formal Visit-Level Evaluation Engine.

Computes temporal Intersection-over-Union (t-IoU) matching between hand-labeled
ground truth visits and pipeline-predicted visits. Evaluates Visit Precision, Recall,
F1 Score, Fragmentation Rate, Duration MAE, Entry Time MAE, and Event Reduction Ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_visits")


def compute_temporal_iou(
    p_enter: float,
    p_exit: float,
    g_enter: float,
    g_exit: float,
) -> float:
    """Compute temporal Intersection-over-Union (t-IoU) between two time intervals.

    Args:
        p_enter: Predicted enter timestamp (seconds).
        p_exit: Predicted exit timestamp (seconds).
        g_enter: Ground truth enter timestamp (seconds).
        g_exit: Ground truth exit timestamp (seconds).

    Returns:
        Intersection over union ratio in [0.0, 1.0].
    """
    intersection_start = max(p_enter, g_enter)
    intersection_end = min(p_exit, g_exit)
    intersection = max(0.0, intersection_end - intersection_start)

    union_start = min(p_enter, g_enter)
    union_end = max(p_exit, g_exit)
    union = max(0.0, union_end - union_start)

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


class VisitEvaluator:
    """Evaluates predicted visit events against hand-annotated ground truth."""

    def __init__(self, iou_threshold: float = 0.50) -> None:
        """Initialize evaluator with temporal matching threshold.

        Args:
            iou_threshold: Minimum temporal IoU required for a valid visit match.
        """
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        predicted_visits: List[Dict[str, Any]],
        ground_truth_visits: List[Dict[str, Any]],
        total_raw_detections: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform bipartite matching and compute all visit-level metrics.

        Args:
            predicted_visits: List of predicted visit dicts containing t_enter_s/video_t_enter_s and t_exit_s/video_t_exit_s.
            ground_truth_visits: List of ground-truth visit dicts from ground_truth.csv.
            total_raw_detections: Optional total count of raw bounding box detections.

        Returns:
            Dictionary containing precision, recall, F1, fragmentation, MAE, and match audit.
        """
        n_gt = len(ground_truth_visits)
        n_pred = len(predicted_visits)

        if n_gt == 0:
            return {
                "total_ground_truth_visits": 0,
                "total_predicted_visits": n_pred,
                "matched_visits": 0,
                "visit_precision": 0.0,
                "visit_recall": 0.0,
                "visit_f1_score": 0.0,
                "fragmentation_rate": 0.0,
                "fragmentation_ratio": 0.0,
                "mean_duration_mae_s": 0.0,
                "mean_entry_mae_s": 0.0,
                "matches": [],
            }

        # Greedy bipartite matching on highest temporal IoU
        matched_gt_indices = set()
        matched_pred_indices = set()
        match_records: List[Dict[str, Any]] = []

        duration_errors: List[float] = []
        entry_errors: List[float] = []

        # Sort candidate pairs by t-IoU descending
        all_pairs = []
        for p_idx, pred in enumerate(predicted_visits):
            p_enter = float(pred.get("video_t_enter_s", pred.get("t_enter_s", 0.0)))
            p_exit = float(pred.get("video_t_exit_s", pred.get("t_exit_s", 0.0)))
            p_dur = float(pred.get("duration_s", p_exit - p_enter))

            for g_idx, gt in enumerate(ground_truth_visits):
                g_enter = float(gt["t_enter_s"])
                g_exit = float(gt["t_exit_s"])
                g_dur = float(gt.get("duration_s", g_exit - g_enter))

                iou = compute_temporal_iou(p_enter, p_exit, g_enter, g_exit)
                all_pairs.append((iou, p_idx, g_idx, p_enter, p_exit, p_dur, g_enter, g_exit, g_dur))

        all_pairs.sort(key=lambda x: x[0], reverse=True)

        for iou, p_idx, g_idx, p_enter, p_exit, p_dur, g_enter, g_exit, g_dur in all_pairs:
            if p_idx in matched_pred_indices or g_idx in matched_gt_indices:
                continue
            if iou >= self.iou_threshold:
                matched_pred_indices.add(p_idx)
                matched_gt_indices.add(g_idx)

                dur_err = abs(p_dur - g_dur)
                ent_err = abs(p_enter - g_enter)
                duration_errors.append(dur_err)
                entry_errors.append(ent_err)

                pred_obj = predicted_visits[p_idx]
                gt_obj = ground_truth_visits[g_idx]

                match_records.append({
                    "ground_truth_id": gt_obj.get("visit_id", f"GT{g_idx+1}"),
                    "predicted_visit_id": str(pred_obj.get("visit_id", f"P{p_idx+1}"))[:8],
                    "person_id_gt": gt_obj.get("person_id", "N/A"),
                    "person_id_pred": pred_obj.get("person_id", "N/A"),
                    "temporal_iou": round(iou, 4),
                    "gt_interval": f"{g_enter:.2f}s - {g_exit:.2f}s ({g_dur:.2f}s)",
                    "pred_interval": f"{p_enter:.2f}s - {p_exit:.2f}s ({p_dur:.2f}s)",
                    "duration_error_s": round(dur_err, 3),
                    "entry_error_s": round(ent_err, 3),
                })

        n_matched = len(matched_gt_indices)
        precision = n_matched / float(n_pred) if n_pred > 0 else 0.0
        recall = n_matched / float(n_gt) if n_gt > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Fragmentation Rate: (pred - gt) / gt (ideal 0.0)
        frag_rate = (n_pred - n_gt) / float(n_gt)
        # Fragmentation Ratio: pred / gt (ideal 1.0)
        frag_ratio = n_pred / float(n_gt)

        event_reduction = (
            f"{round(total_raw_detections / float(n_pred), 1)}:1"
            if (total_raw_detections is not None and n_pred > 0)
            else "N/A"
        )

        return {
            "total_ground_truth_visits": n_gt,
            "total_predicted_visits": n_pred,
            "matched_visits": n_matched,
            "temporal_iou_threshold": self.iou_threshold,
            "visit_precision": round(precision, 4),
            "visit_recall": round(recall, 4),
            "visit_f1_score": round(f1, 4),
            "fragmentation_rate": round(frag_rate, 4),
            "fragmentation_ratio": round(frag_ratio, 4),
            "mean_duration_mae_s": round(float(np.mean(duration_errors)), 3) if duration_errors else 0.0,
            "mean_entry_mae_s": round(float(np.mean(entry_errors)), 3) if entry_errors else 0.0,
            "total_raw_detections": total_raw_detections,
            "event_reduction_ratio": event_reduction,
            "match_records": match_records,
        }


def load_ground_truth_csv(file_path: Path, roi_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load and parse ground truth visit rows from CSV."""
    visits = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if roi_id is not None and r.get("roi_id") != roi_id:
                continue
            visits.append({
                "visit_id": r["visit_id"],
                "person_id": r["person_id"],
                "frame_enter": int(r["frame_enter"]),
                "frame_exit": int(r["frame_exit"]),
                "t_enter_s": float(r["t_enter_s"]),
                "t_exit_s": float(r["t_exit_s"]),
                "duration_s": float(r["duration_s"]),
                "roi_id": r.get("roi_id", ""),
                "notes": r.get("notes", ""),
            })
    return visits


def load_predicted_visits(file_path: Path) -> List[Dict[str, Any]]:
    """Load predicted visits from JSON or CSV file."""
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("visits", [])
    elif file_path.suffix.lower() == ".csv":
        visits = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                visits.append({
                    "visit_id": r.get("visit_id", ""),
                    "person_id": r.get("person_id", ""),
                    "t_enter_s": float(r.get("t_enter_s", r.get("video_t_enter_s", 0.0))),
                    "t_exit_s": float(r.get("t_exit_s", r.get("video_t_exit_s", 0.0))),
                    "duration_s": float(r.get("duration_s", 0.0)),
                    "roi_id": r.get("roi_id", ""),
                })
        return visits
    raise ValueError(f"Unsupported file format: {file_path}")


def main() -> None:
    """Run formal visit evaluation and produce formatted report."""
    parser = argparse.ArgumentParser(description="Formal Visit-Level Evaluation Engine")
    parser.add_argument("--ground-truth", type=str, default="eval/ground_truth.csv", help="Ground truth CSV path")
    parser.add_argument("--visits", "--predictions", dest="visits", type=str, default="outputs/database/sample_visits.json", help="Predicted visits JSON or CSV path")
    parser.add_argument("--roi-id", type=str, default="roi_entrance", help="Target ROI ID to filter")
    parser.add_argument("--output", type=str, default="eval/results.json", help="Results output JSON path")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="Temporal IoU threshold")
    parser.add_argument("--raw-detections", type=int, default=402, help="Raw detections count for reduction ratio")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    pred_path = Path(args.visits)
    if not pred_path.exists() and str(pred_path) == "outputs/sample_visits.json":
        cat_p = Path("outputs/database/sample_visits.json")
        if cat_p.exists():
            pred_path = cat_p

    if not gt_path.exists():
        logger.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)
    if not pred_path.exists():
        logger.error("Predicted visits file not found: %s", pred_path)
        sys.exit(1)

    gt_visits = load_ground_truth_csv(gt_path, roi_id=args.roi_id)
    pred_visits = load_predicted_visits(pred_path)

    evaluator = VisitEvaluator(iou_threshold=args.iou_thresh)
    results = evaluator.evaluate(
        predicted_visits=pred_visits,
        ground_truth_visits=gt_visits,
        total_raw_detections=args.raw_detections,
    )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation results to %s", out_file)

    # Print formatted Markdown table
    print("\n" + "=" * 80)
    print("NAVSECURA VISIT-LEVEL EVENT EVALUATION REPORT")
    print("=" * 80)
    print(f"Ground Truth Dataset:     {gt_path} (ROI: {args.roi_id})")
    print(f"Predictions File:         {pred_path}")
    print(f"Temporal IoU Threshold:   {results['temporal_iou_threshold']:.2f}")
    print("-" * 80)
    print("EVALUATION METRICS:")
    print(f"  * Total Ground Truth Visits:  {results['total_ground_truth_visits']}")
    print(f"  * Total Predicted Visits:     {results['total_predicted_visits']}")
    print(f"  * Matched Visits:             {results['matched_visits']}")
    print(f"  * Visit Precision:            {results['visit_precision'] * 100:.2f}%")
    print(f"  * Visit Recall:               {results['visit_recall'] * 100:.2f}%")
    print(f"  * Visit F1-Score:             {results['visit_f1_score'] * 100:.2f}%")
    print(f"  * Fragmentation Rate:         {results['fragmentation_rate']:.4f} (Ideal: 0.0)")
    print(f"  * Mean Duration Error (MAE):  {results['mean_duration_mae_s']:.3f} seconds")
    print(f"  * Mean Entry Time Error (MAE):{results['mean_entry_mae_s']:.3f} seconds")
    print(f"  * Event Reduction Ratio:      {results['event_reduction_ratio']}")
    print("-" * 80)
    print("INDIVIDUAL VISIT MATCH AUDIT:")
    print(f"{'GT ID':<6} | {'Pred ID':<10} | {'Person':<8} | {'t-IoU':<6} | {'Duration Error':<16} | {'Entry Error':<12}")
    print("-" * 80)
    for m in results["match_records"]:
        print(
            f"{m['ground_truth_id']:<6} | {m['predicted_visit_id']:<10} | "
            f"{m['person_id_gt']:<8} | {m['temporal_iou']:<6.3f} | "
            f"{m['duration_error_s']:>12.3f} s  | {m['entry_error_s']:>8.3f} s"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

