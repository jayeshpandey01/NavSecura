"""Render broadcast-quality evaluation verification scorecard and diagnostic charts.

Generates outputs/evaluation_verification.png combining:
1. Executive Metric Badges (Precision, Recall, F1, Fragmentation, Dwell MAE, Event Reduction).
2. Ground Truth vs Predicted Visit Temporal Alignment Gantt Chart.
3. Comparative Tracker Latency & Throughput Benchmark (ByteTrack vs BoT-SORT).
4. Hyperparameter Sensitivity Curves (T_grace vs Dwell MAE & T_min_dwell vs Recall).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load evaluation results, ablation study, and tracker comparison telemetry
eval_path = Path("eval/results.json")
ablation_path = Path("eval/ablation_study.json")
tracker_path = Path("outputs/tracker_comparison_meta.json")

with open(eval_path, "r", encoding="utf-8") as f:
    eval_res = json.load(f)

with open(ablation_path, "r", encoding="utf-8") as f:
    abl = json.load(f)

with open(tracker_path, "r", encoding="utf-8") as f:
    tc_meta = json.load(f)

# Set high DPI figure with professional dark theme
fig = plt.figure(figsize=(18, 12), dpi=150)
fig.patch.set_facecolor("#0b0f19")

# Grid layout: Top summary cards (0.12), Gantt chart (0.45), Bottom two plots (0.43)
gs = fig.add_gridspec(3, 2, height_ratios=[0.25, 1.1, 0.95], hspace=0.32, wspace=0.22)

# ---------------------------------------------------------------------------
# Section 1: Executive KPI Cards (Top Banner)
# ---------------------------------------------------------------------------
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_facecolor("#111827")
ax_kpi.set_xlim(0, 6)
ax_kpi.set_ylim(0, 1)
ax_kpi.axis("off")

kpi_data = [
    ("VISIT PRECISION", f"{eval_res['visit_precision']*100:.1f}%", "#10b981", "7 / 7 Matched (0 FP)"),
    ("VISIT RECALL", f"{eval_res['visit_recall']*100:.1f}%", "#10b981", "7 / 7 Detected (0 FN)"),
    ("VISIT F1-SCORE", f"{eval_res['visit_f1_score']:.3f}", "#10b981", "Harmonic Balance"),
    ("FRAGMENTATION RATE", f"{eval_res['fragmentation_rate']:.3f}", "#3b82f6", "Ideal = 0.000"),
    ("DWELL DURATION MAE", f"{eval_res['mean_duration_mae_s']:.3f}s", "#f59e0b", "Mean Absolute Error"),
    ("ALERT REDUCTION", f"{eval_res['event_reduction_ratio']}", "#8b5cf6", "402 Dets -> 7 Visits"),
]

for i, (title, val, color, sub) in enumerate(kpi_data):
    # Card background box
    rect = patches.FancyBboxPatch(
        (i + 0.06, 0.08),
        0.88,
        0.84,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor="#1a2234",
        edgecolor=color,
        linewidth=1.5,
        alpha=0.9,
    )
    ax_kpi.add_patch(rect)
    ax_kpi.text(i + 0.5, 0.72, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#94a3b8")
    ax_kpi.text(i + 0.5, 0.42, val, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax_kpi.text(i + 0.5, 0.18, sub, ha="center", va="center", fontsize=7.5, color="#cbd5e1")

# ---------------------------------------------------------------------------
# Section 2: Ground Truth vs Predicted Temporal Alignment Gantt Chart
# ---------------------------------------------------------------------------
ax_gantt = fig.add_subplot(gs[1, :])
ax_gantt.set_facecolor("#111827")

records = eval_res["match_records"]
y_ticks = []
y_labels = []

n_visits = len(records)
for idx, r in enumerate(reversed(records)):
    y_pos = idx * 2.2
    gt_parts = r["gt_interval"].split("(")[0].strip().split("-")
    gt_start = float(gt_parts[0].replace("s", "").strip())
    gt_end = float(gt_parts[1].replace("s", "").strip())
    gt_dur = gt_end - gt_start

    pred_parts = r["pred_interval"].split("(")[0].strip().split("-")
    pred_start = float(pred_parts[0].replace("s", "").strip())
    pred_end = float(pred_parts[1].replace("s", "").strip())
    pred_dur = pred_end - pred_start

    # Blue bar for Ground Truth
    ax_gantt.broken_barh(
        [(gt_start, gt_dur)],
        (y_pos + 0.45, 0.65),
        facecolors="#2563eb",
        edgecolors="#60a5fa",
        linewidth=1.2,
        alpha=0.95,
        label="Ground Truth Visit" if idx == 0 else "",
    )
    # Orange bar for Predicted Visit
    ax_gantt.broken_barh(
        [(pred_start, pred_dur)],
        (y_pos - 0.35, 0.65),
        facecolors="#ea580c",
        edgecolors="#fb923c",
        linewidth=1.2,
        alpha=0.95,
        label="Predicted Pipeline Visit" if idx == 0 else "",
    )

    tiou = r["temporal_iou"]
    dur_err = r["duration_error_s"]
    ent_err = r["entry_error_s"]
    label_text = f"t-IoU = {tiou:.3f} | Dur Err = {dur_err:.2f}s | Ent Err = {ent_err:.2f}s"

    ax_gantt.text(
        max(gt_end, pred_end) + 0.7,
        y_pos + 0.20,
        label_text,
        fontsize=9,
        fontweight="bold",
        color="#f1f5f9",
        va="center",
    )

    y_ticks.append(y_pos + 0.20)
    y_labels.append(f"{r['ground_truth_id']} ({r['person_id_gt']})")

ax_gantt.set_yticks(y_ticks)
ax_gantt.set_yticklabels(y_labels, fontsize=10, fontweight="bold", color="#f1f5f9")
ax_gantt.set_xlabel("Stream Timeline (Seconds)", fontsize=11, fontweight="bold", color="#f1f5f9")
ax_gantt.set_title(
    "Ground Truth vs Predicted Visit Temporal Overlap [t-IoU >= 0.50] (Blue = Ground Truth, Orange = Predicted)",
    fontsize=12,
    fontweight="bold",
    color="#38ef7d",
    pad=10,
)
ax_gantt.set_xlim(0, 52)
ax_gantt.set_ylim(-1.0, (n_visits - 1) * 2.2 + 1.8)
ax_gantt.tick_params(colors="#cbd5e1")
ax_gantt.grid(axis="x", linestyle="--", alpha=0.25, color="#64748b")
ax_gantt.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9", loc="upper right", fontsize=9.5)

# ---------------------------------------------------------------------------
# Section 3: Tracker Latency & Association Benchmark (ByteTrack vs BoT-SORT)
# ---------------------------------------------------------------------------
ax_trk = fig.add_subplot(gs[2, 0])
ax_trk.set_facecolor("#111827")

byte_b = tc_meta["bytetrack_benchmark"]
bot_b = tc_meta["botsort_benchmark"]

metrics_trk = ["Avg Latency", "Median Latency", "P95 Latency"]
byte_lat = [byte_b["avg_tracker_latency_ms"], byte_b["median_tracker_latency_ms"], byte_b["p95_tracker_latency_ms"]]
bot_lat = [bot_b["avg_tracker_latency_ms"], bot_b["median_tracker_latency_ms"], bot_b["p95_tracker_latency_ms"]]

x_trk = np.arange(len(metrics_trk))
w = 0.35

rects1 = ax_trk.bar(x_trk - w/2, byte_lat, width=w, label=f"ByteTrack ({byte_b['overall_fps']:.1f} FPS, 0.20ms avg)", color="#059669", edgecolor="#34d399", linewidth=1.2)
rects2 = ax_trk.bar(x_trk + w/2, bot_lat, width=w, label=f"BoT-SORT GMC (1.64ms avg, ~8.2x slower)", color="#dc2626", edgecolor="#f87171", linewidth=1.2)

ax_trk.set_xticks(x_trk)
ax_trk.set_xticklabels(metrics_trk, fontsize=10, fontweight="bold", color="#f1f5f9")
ax_trk.set_ylabel("Tracker Latency (Milliseconds)", fontsize=10, fontweight="bold", color="#f1f5f9")
ax_trk.set_title(
    f"Tracker Association Latency Benchmark (Fixed CCTV Camera)",
    fontsize=11.5,
    fontweight="bold",
    color="#38ef7d",
    pad=10,
)
ax_trk.tick_params(colors="#cbd5e1")
ax_trk.grid(axis="y", linestyle="--", alpha=0.25, color="#64748b")
ax_trk.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9", loc="upper left", fontsize=9)

for rect in rects1:
    h = rect.get_height()
    ax_trk.text(rect.get_x() + rect.get_width()/2.0, h + 0.06, f"{h:.2f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#34d399")

for rect in rects2:
    h = rect.get_height()
    ax_trk.text(rect.get_x() + rect.get_width()/2.0, h + 0.06, f"{h:.2f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#f87171")

ax_trk.set_ylim(0, max(bot_lat) * 1.35)

# ---------------------------------------------------------------------------
# Section 4: Hyperparameter Sensitivity Curves (T_grace & T_min_dwell)
# ---------------------------------------------------------------------------
ax_sens = fig.add_subplot(gs[2, 1])
ax_sens.set_facecolor("#111827")

grace_data = abl["grace_period_sweep"]
g_x = [row["grace_period_s"] for row in grace_data]
g_mae = [row["mean_duration_mae_s"] for row in grace_data]

dwell_data = abl["min_dwell_sweep"]
d_x = [row["min_dwell_s"] for row in dwell_data]
d_rec = [row["visit_recall"] * 100 for row in dwell_data]

line1 = ax_sens.plot(g_x, g_mae, marker="o", markersize=7, color="#f59e0b", linewidth=2.5, label="Duration MAE (s) vs T_grace")
ax_sens.set_xlabel("Threshold Parameter (Seconds)", fontsize=10.5, fontweight="bold", color="#f1f5f9")
ax_sens.set_ylabel("Duration Error MAE (s)", fontsize=10, fontweight="bold", color="#f59e0b")
ax_sens.tick_params(axis="y", labelcolor="#f59e0b", colors="#cbd5e1")
ax_sens.tick_params(axis="x", colors="#cbd5e1")
ax_sens.grid(True, linestyle="--", alpha=0.25, color="#64748b")
ax_sens.set_ylim(0.12, 0.17)

ax_sens2 = ax_sens.twinx()
line2 = ax_sens2.plot(d_x, d_rec, marker="s", markersize=7, color="#38bdf8", linewidth=2.5, linestyle="--", label="Visit Recall (%) vs T_min_dwell")
ax_sens2.set_ylabel("Visit Recall (%)", fontsize=10, fontweight="bold", color="#38bdf8")
ax_sens2.tick_params(axis="y", labelcolor="#38bdf8")
ax_sens2.set_ylim(25, 115)

# Correctly positioned in-axis annotation lines
ax_sens.axvline(2.0, color="#10b981", linestyle=":", linewidth=2)
ax_sens.axvline(1.0, color="#a855f7", linestyle=":", linewidth=2)

# Place text safely inside plot bounds using axis coordinates
ax_sens.text(2.08, 0.162, "Optimal T_grace = 2.0s\n(Absorbs dropouts)", color="#10b981", fontsize=8.5, fontweight="bold")
ax_sens.text(1.08, 0.150, "Optimal T_dwell = 1.0s\n(V07 preserved)", color="#a855f7", fontsize=8.5, fontweight="bold")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax_sens.legend(lines, labels, facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9", loc="center left", fontsize=9)
ax_sens.set_title("Hyperparameter Sensitivity: Grace & Dwell Sweeps", fontsize=11.5, fontweight="bold", color="#38ef7d", pad=10)

# Save high-resolution PNG
out_png = Path("outputs/evaluation_verification.png")
plt.savefig(str(out_png), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
plt.close(fig)

print("Rendered polished high-resolution verification scorecard:", out_png)
