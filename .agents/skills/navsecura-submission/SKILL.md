---
name: navsecura-submission
description: Governs the assembly and verification of the final NavSecura submission package: repository structure, README.md, attribution tables, engineering discussion answers, limitation analysis, and clean-clone verification. Use in preparation for final project handoff.
---

# NavSecura Packager

## Goal
Package the complete assessment solution into an exceptional, reproducible, and transparent engineering repository that exceeds all NavSecura criteria.

## Instructions
1. **Repository Layout Enforcement:**
   - Verify presence of: `README.md`, `requirements.txt`, `configs/default.yaml`, `configs/roi_example.json`, `src/navsecura/`, `tools/`, `tests/`, `eval/`, and `docs/`.
2. **README Construction:**
   - Clear problem statement: converting repeated frame detections into coherent visit events.
   - Architecture diagram (Mermaid flow).
   - Quickstart guide (clone, venv, install, run sample video in one command).
   - Configuration parameter reference (`grace_s`, `min_dwell`, `enter_frames`).
   - SQLite schema documentation.
   - Evaluation benchmark table (ByteTrack vs. BoT-SORT).
   - Third-party model, library, and video attribution with explicit license disclosures.
3. **Engineering Discussion Document (`docs/engineering_discussion.md`):**
   - Provide comprehensive, authoritative answers to the four assessment prompts:
     1. Why detection and tracking approach was chosen (YOLO + ByteTrack rationale, AGPL vs. Apache alternatives).
     2. Handling of occlusion and leave/re-enter dynamics (three-tier defense: tracker Kalman, event grace, spatio-temporal stitching).
     3. Multi-camera site expansion (edge extraction, camera topology graphs, time-window gating, MTMC association).
     4. Recurring activity learning without neural retraining (streaming Gamma-Poisson baselines, EWMA dwell models, HS-Trees).
4. **Limitation Analysis (`docs/limitations.md`):**
   - Document at least one authentic failure case observed during video testing (e.g., ID switch during crossing, prolonged occlusion splitting).
   - Provide root cause diagnosis, empirical traces, and practical engineering remediations.
5. **Clean-Clone Reproduction Verification:**
   - Execute an isolated verification in a fresh directory to ensure all dependencies install and the test command runs end-to-end without errors.

## Examples
- Verification command: `python tools/verify_submission.py --repo .`
- Export summary: generate final completion checklist with pass/fail status for all requirements.

## Constraints
- Ensure no proprietary, unlicensed, or private videos are committed.
- Ensure all third-party code and model weights are formally attributed with licenses.
- No claims without reproducible code or empirical evidence.

