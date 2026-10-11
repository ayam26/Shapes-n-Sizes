# Shape-Biased Detector — Footage & Audit Implementation Plan (Plan 3 of 3)

> Executed inline, TDD per module. Compact form (interfaces, behaviors, tests).

**Goal:** Tools for gate G7 (spec §4.4, §6): unfiltered frame sampling from the user's drives,
footage predictions per arm at the full-val equal-recall thresholds, a blind audit page,
audit scoring (Wilson CIs, estimated real / invented detections), recall on hand-labeled frames,
and optional inter-rater agreement.

**Spec:** `docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md` §4.4, §6.

## Tasks

### Task 1: `data/footage.py` + `snz footage-sample`
- `choose_interval(total_seconds, min_frames=2000) -> float`: largest whole-second N ≥ 1 with
  ⌊total/N⌋ ≥ min_frames (fewer frames than that only if the footage is too short; N = 1).
- `sample_times(duration, every) -> list[float]`: 0, N, 2N, … < duration.
- `extract_frames(video, drive_id, every, out_dir) -> (rows, drops)` with PyAV; a frame that
  cannot be decoded or is all black (mean < 2/255) is dropped and logged with the reason.
- Input `drives.csv` (drive_id, file, time_of_day, weather — tagged by the user); outputs
  `footage.csv` manifest (name, sha256, drive, t, time_of_day, weather) and `footage_drops.csv`.
- Tests: interval choice; sample times; a synthetic video (PyAV, mpeg4) with a black segment →
  expected frame names, drop log, manifest.

### Task 2: footage predictions + `audit/sample.py`
- `snz evaluate`-like `snz footage-predict` (split name `footage`, guarded by `prereg-v1`), no labels.
- `sample_audit(dets_by_arm, thresholds, n_per_arm=150, seed=0) -> (items, key)`: uniform
  sample of thresholded detections per arm, all shuffled together, item ids `a000…`; items carry
  frame name, box, class, but no arm or score; key maps id → (arm, frame, det index).
- Tests: per-arm counts; no arm/score in items; deterministic per seed; fewer detections than
  n → all of them.

### Task 3: `audit/page.py`
- `build_audit_page(items, frames_dir, out_dir)`: one JPEG per item with the box drawn, and a
  static `index.html` with the four labels per item (keyboard 1–4), progress kept in
  localStorage, and "Download CSV" (item_id,label).
- Tests: files written; HTML lists every item id; no arm names in the HTML.

### Task 4: `audit/score.py` + `snz audit-score`
- `wilson(k, n, z=1.96) -> (p, lo, hi)`; `score_audit(labels_csv, key, total_detections) ->
  per arm {n, unsure, precision (Wilson), invented share (Wilson), est_real, est_invented
  (count × proportion and its CI bounds)}`; `cohen_kappa(a, b)`.
- `load_coco_labels(path) -> list[Frame]` for the 100 hand-labeled frames (CVAT / Label Studio
  COCO export), then per arm: recall and false detections per frame at its threshold (reusing
  `match_split`, `recall_at`, `false_detections`).
- Tests: Wilson against hand values; scoring a constructed audit; kappa; COCO import.
