# Shape-Biased Detector — Analysis Implementation Plan (Plan 2 of 3)

> Executed inline (superpowers:executing-plans), TDD per module. Compact form: interfaces,
> behaviors and test cases are specified; code is written test-first during execution.

**Goal:** Everything needed for gates G4–G6: test-split prediction (incl. T-texture),
per-image matching, vectorized AP for the bootstrap, equal-recall operating points,
false-detection types, paired bootstrap + Holm, the `snz analyze` report, and the
preregistration draft.

**Spec:** `docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md` §3, §4.2, §7.
**Depends on:** Plan 1 (`docs/superpowers/plans/2026-10-11-snz-training-pipeline.md`).

## Global constraints (from the spec)

- Primary mAP = COCO mAP@[.5:.95] averaged over the 9 `EVAL_CLASSES`; AP50 and per-class AP reported.
- Vectorized AP must agree with pycocotools within 0.1 mAP points on full splits (tests: within 1e-6 on synthetic data).
- Arm metric = mean over its 3 seeds. Paired bootstrap over test images, B = 2000, the same resampled image multiset for every run in a replicate; 95% percentile CIs of arm − BASE; one-sided p in the predicted direction.
- Confirmatory family {H1b-STY, H1b-SIN, H2, H3, H4-STY, H4-SIN}, Holm, α = 0.05. BOTH and additivity exploratory (CIs, no tests).
- R* = BASE seed-0 recall on pooled T-shift at score ≥ 0.5, IoU ≥ 0.5, 9 classes pooled. Each run's threshold gives recall = R*; runs that cannot reach R* are evaluated at max recall and flagged. Full-val R* (same construction on all BDD val) gives the footage thresholds.
- False-detection types, in order: duplicate, wrong class, poor localization, invented (spec §7.3). Detections on ignore regions are neither TP nor false.
- No test split is evaluated before `prereg-v1`.

## Tasks

### Task 1: More manifests + prediction on stylized/test images
- `build_and_write_splits` also writes descriptive splits `t-dusk` (val dawn/dusk, any weather) and `t-overcast` (val daytime ∧ overcast) — included in the disjointness check — and `val-all.csv` (all val frames; not in the check). Existing manifests are unchanged (deterministic).
- `BDDDetection(..., file_name: Callable[[str], str] | None)` for eval: T-texture reads `stylized/t-texture/<stem>_s1.jpg` while records keep the original name.
- `snz evaluate --stylized-copy N`: split `t-texture` takes manifest `t-clean.csv` (guard accepts this pairing only).
- Tests: descriptive splits membership; val-all count; file_name mapping; guard accepts (t-texture, t-clean.csv) and rejects (t-texture, id-val.csv).

### Task 2: `eval/matching.py`
- `IOU_THRESHOLDS = np.linspace(.5, .95, 10)`; `box_iou(a, b)`; COCO crowd IoU (intersection / det area) for ignore regions.
- `Matches` (dataclass): `classes, image_names, det_image, det_class, det_score, det_tp[N,T], det_ignored[N,T], det_fd_type[N] (at IoU .5), gt_count[n_images, K]`; `Matches.subset(image_names)` re-indexes to a subset of images.
- `match_split(frames, detections, classes=EVAL_CLASSES, max_dets=100) -> Matches`: COCO greedy matching per image × class, dets by descending score (stable), truncated to max_dets; prefers non-ignore GT, may match ignore regions repeatedly.
- `det_fd_type` codes: 0 TP, 1 duplicate, 2 wrong class, 3 poor localization, 4 invented, 5 near another class (0.1 ≤ IoU < .5 with another class only; not named by the spec), 6 ignore region. Wrong-class/invented checks use all GT boxes of every category (incl. `train`).
- Tests: hand cases for each fd type; duplicate; ignore region absorbs a det; max_dets truncation.

### Task 3: `eval/ap.py`
- `average_precision(m, weights=None) -> {"mAP", "AP50", "per_class"}`: COCO 101-point interpolation; `weights[n_images]` are bootstrap multiplicities (weighted cumsums, sort order precomputed once per class).
- Tests: equals `coco_eval.evaluate` (pycocotools) within 1e-6 on randomized synthetic splits (several seeds); integer weights equal literal image duplication (within 1e-3); hand case 51/101.

### Task 4: `eval/operating.py`
- `recall_at(m, threshold, weights=None)` (IoU .5, pooled classes); `threshold_for_recall(m, target) -> OperatingPoint(threshold, recall, flagged)` (the highest threshold whose recall ≥ target; else the lowest score, flagged); `false_detections(m, threshold, weights=None) -> {"per_image", "by_type": {...}}`.
- Tests: hand cases; flagged case; weighted per-image means.

### Task 5: `eval/bootstrap.py`
- `bootstrap_weights(n, B, seed) -> np.ndarray[B, n]` (multinomial counts); `one_sided_p(diffs, direction)` = (1 + #{diff on the wrong side or 0}) / (B + 1); `percentile_ci(x)`; `holm(pvals: dict) -> dict`.
- Tests: reproducible with seed; recovers a planted difference (p < .05) and not a null one; Holm against a hand-computed example.

### Task 6: `eval/analyze.py` + `snz analyze`
- Inputs: `--runs DIR` with `<ARM>-s<seed>/eval/{val-all,t-texture}.detections.jsonl`, val labels, manifests. Slices val-all into t-clean, t-night, t-rain, t-snow, t-fog, t-dusk, t-overcast and pooled T-shift.
- Outputs `results/analysis.json` and `results/report.md`: per-run and per-arm (mean, SD, min/max over seeds) mAP/AP50/per-class for every subset; retention; R*, per-run thresholds and flags; false detections per image and by type; bootstrap CIs/p for the six confirmatory tests + Holm; exploratory BOTH/additivity CIs; full-val thresholds for the footage.
- Tests: end-to-end on a synthetic 4-arm × 2-seed fixture where one arm is planted better; missing runs reported, not crashing.

### Task 7: `prereg/prereg.md`
- Hypotheses, metrics, splits with counts and manifest hashes, frozen recipe (sha256), analysis exactly as above, the 2018-labels deviation, ignore-region handling, footage sampling rule and audit protocol (spec §4.4, §6). Committed and tagged `prereg-v1` only after G3 freezes the recipe.
