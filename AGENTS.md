# NavSecura CV Assessment: Agent Architecture and Operational Guidelines

This project implements the **NavSecura Computer Vision Engineering Assessment: Video Understanding, Tracking and Event Memory**.
The target standard is 10/10 Code Quality, Empirical Rigor, and Zero-Defect CV Systems Execution.

## Operating Principles
1. **Empirical Truth:** Never fabricate benchmark numbers, FPS, or bounding box coordinates.
2. **Deep Inspection:** Always verify video stream properties (resolution, FPS, codec) and coordinate geometries.
3. **Event Semantics:** Aggregate frame detections into one coherent event per visit with entry time and dwell time.
4. **Tolerance to Tracker Imperfections:** Absorb tracker noise with hysteresis, grace periods, and spatio-temporal ID stitching.
5. **Living Architecture:** Respect module decoupling in `src/navsecura/` (`io/`, `detect/`, `track/`, `roi/`, `events/`, `store/`, `evidence/`, `analytics/`).

## Workspace Skills & Specialized Roles
The following specialized skills are available under `.agents/skills/`:
- **`cv-tracker-evaluator`**: Multi-object tracking evaluation, inference benchmarking (ByteTrack vs. BoT-SORT), and visit-level metric computation.
- **`visit-event-guard`**: State machine lifecycle (`OUTSIDE` -> `INSIDE` -> `GRACE` -> `CLOSED`), foot-point polygon testing, and synthetic unit testing.
- **`pattern-learning-miner`**: Streaming count baselines, EWMA dwell modeling, and anomaly detection over events without retraining.
- **`navsecura-submission`**: README packaging, licensing disclosure, engineering discussion answers, limitation analysis, and clean-clone verification.

Refer to `docs/instruction.md` and `docs/plan.md` for full implementation protocols and milestone criteria.

