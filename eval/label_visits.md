# NavSecura Visit Ground Truth Annotation Protocol

> **Specification Standard:** Operational Ground Truth Protocol for Video Analytics & Tracking  
> **Target Schema:** `eval/ground_truth.csv`  
> **Authoritative Baseline:** Milestone 0 (Task 0.4) & `visit-event-guard` Skill

---

## 1. Objectives & Ground Truth Definition

The purpose of this protocol is to define an unambiguous, mathematically reproducible procedure for labeling human visit events in video streams. In video surveillance, naive frame-level bounding boxes are insufficient to evaluate visit semantics. Ground truth must represent the **visit lifecycle** of an individual within a designated spatial Region of Interest (ROI).

A ground truth **Visit Event** is defined by:
1. A unique subject identity (`person_id`).
2. The instant the subject enters the designated polygon ground area ($t_{\text{enter}}$).
3. The instant the subject exits the designated polygon ground area ($t_{\text{exit}}$).
4. The true dwell duration: $\Delta t = t_{\text{exit}} - t_{\text{enter}}$.

---

## 2. Core Operational Rules

### 2.1 The Ground-Plane Foot-Point Rule
Perspective projections in CCTV cameras cause person bounding boxes to overlap polygons long before the subject physically enters the designated ground zone. 

* **Rule:** A person is defined as **INSIDE** the ROI if and only if their ground-plane contact point lies within or on the boundary of the defined ROI polygon.
* **Foot-Point Formulation:**
  $$P_{\text{foot}} = \left( \frac{x_1 + x_2}{2},\, y_2 \right)$$
  where $[x_1, y_1, x_2, y_2]$ is the tight bounding box encompassing the person, and $y_2$ represents the ground contact level (bottom of shoes).
* **Two Feet Edge Case:** When a walking subject takes a step with one foot inside the polygon and one foot outside, the contact point is taken as the midpoint between both feet on the ground plane. Entry occurs on the first frame where this contact point crosses the polygon boundary.

### 2.2 Deterministic Frame-Based Timestamping
To guarantee zero drift across variable execution speeds or player clocks, all timestamps are strictly derived from the discrete video frame index:
$$t_{\text{seconds}} = \frac{\text{frame\_idx}}{\text{native\_fps}}$$
* **`frame_enter`**: The first frame index where $P_{\text{foot}} \in \text{ROI}$.
* **`frame_exit`**: The first frame index where $P_{\text{foot}} \notin \text{ROI}$ following an inside sequence.
* **`t_enter_s`**: `frame_enter / native_fps` rounded to 3 decimal places.
* **`t_exit_s`**: `frame_exit / native_fps` rounded to 3 decimal places.
* **`duration_s`**: `t_exit_s - t_enter_s`.

### 2.3 Re-Entry Policy & Temporal Gap Separation
* **Continuous Visit:** If an individual is temporarily occluded (e.g., passes behind a column, pillar, or another person) or steps out for a duration $\le 5.0$ seconds before resuming their presence in the ROI, the event is treated as **one continuous visit**.
* **Separate Visit (Re-entry):** If an individual exits the ROI and remains outside for $> 5.0$ seconds before returning to the ROI, the return is cataloged as a **new, distinct visit** ($V_{n+1}$) associated with the same `person_id`.
* In downstream evaluation, re-entries allow assessing whether the system correctly links `parent_visit_id` without falsely merging disconnected sessions.

### 2.4 Minimum Dwell Threshold ($t_{\text{dwell}} \ge 1.0\text{s}$)
* Transits across corners lasting fewer than 1.0 second (or $\approx 30$ frames at 30 FPS) are categorized as transient transits. They are recorded in ground truth with notes indicating `transient_corner_cut` to benchmark against the automated filter `min_dwell_s: 1.0`.

---

## 3. Handling Ambiguities & Failure Modes

| Case / Failure Mode | Visual Scenario | Ground Truth Labeling Instruction |
|---|---|---|
| **F1: Partial Occlusion** | Subject torso visible behind pillar, feet obscured | Interpolate foot position along ground contact trajectory. Dwell continues uninterrupted. |
| **F2: Dynamic Crossing** | Two individuals cross paths inside ROI | Preserve persistent human identities ($P_1, P_2$). Each maintains independent entry/exit timestamps. |
| **F3: Boundary Hesitation** | Subject paces along perimeter, stepping in/out within 1–2s | Count as single visit; exit is only logged when subject clearly departs from the perimeter zone. |
| **F4: Re-entry** | Subject leaves ROI, walks away, returns after 10 seconds | Label as two distinct visit rows with identical `person_id`. |
| **F5: Stream Termination** | Subject still inside ROI at final video frame | Set `frame_exit` = final video frame; document `notes: truncated_by_stream_end`. |

---

## 4. Ground Truth File Specification (`eval/ground_truth.csv`)

The ground truth dataset is stored in standard CSV format conforming to RFC 4180 with UTF-8 encoding.

### Schema:
- `visit_id` (string): Unique identifier for the visit event (e.g. `V01`, `V02`).
- `person_id` (string): Persistent identifier of the human subject (e.g. `P1`, `P2`).
- `frame_enter` (int): Zero-indexed video frame of entry.
- `frame_exit` (int): Zero-indexed video frame of exit.
- `t_enter_s` (float): Entry timestamp in seconds from video start.
- `t_exit_s` (float): Exit timestamp in seconds from video start.
- `duration_s` (float): Duration in seconds (`t_exit_s - t_enter_s`).
- `roi_id` (string): Region of interest identifier (e.g. `roi_entrance`).
- `notes` (string): Scenario classification, failure mode exercised, and qualitative observations.

---

## 5. Labeler Agreement & Verification Procedure

1. Two independent reviews are conducted using frame-accurate video inspection.
2. If frame discrepancy between labelers is $\le 2$ frames ($\le 66\text{ms}$ at 30 FPS), the mean frame index is recorded.
3. If discrepancy exceeds 2 frames, the ground plane contact polygon intersection is verified frame-by-frame using coordinate inspection tools.

