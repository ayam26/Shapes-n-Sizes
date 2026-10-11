# GNN Shape Verifier — Design Spec

- **Date:** 2026-10-11 · **Status:** approved in conversation; time-boxed to the 13:00 PDT deadline
- **Builds on:** the CNN detector study (`2026-10-11-shape-biased-detector-design.md`, frozen as `prereg-v1`). That study is stopped after its **BASE seed-0** run; BASE-s0 becomes stage 1 of this pipeline, and any other finished CNN arms (STY, …) are reported as existing baselines.
- **Plan:** `docs/superpowers/plans/2026-10-11-gnn-shape-verifier.md`
- **Preregistration:** `prereg/prereg-v2.md`, tag `prereg-v2` (frozen before any GNN output on a test split)

## 1. Idea

Camera detectors make texture-driven mistakes: a shadow, a reflection, or a patch of truck-colored texture becomes a "car". A **graph neural network that sees only an object's outline geometry and size — never its pixels' colors or texture —** is texture-blind by construction. We use it as a second stage that re-scores every detection:

```text
dashcam frame ─► BASE-s0 Faster R-CNN (all boxes with score ≥ 0.05, top 50 per image)
             ─► SAM 2 (box prompt) ─► outline mask ─► 64 points along the contour
             ─► shape graph (geometry node features + size features; no colors)
             ─► GNN (EdgeConv) ─► P(class) and P(not an object)
             ─► fusion with the detector score ─► new score ─► standard evaluation
```

Because the GNN can push down fake detections, the detector can run at a lower threshold and still keep its false-alarm rate: **more real objects at the same number of false alarms** — the project's "more objects than before, and are they real?" question.

**Known limit (stated in the paper):** a picture of a truck has a truck's outline, so flat depictions (billboards, truck-side graphics) pass the GNN. Shape-only checking targets texture-driven errors, not depiction confusion.

## 2. Questions and hypotheses (`prereg-v2`)

All confirmatory tests are on pooled **T-shift** (night, rain, snow, fog of BDD100K val; same slices as `prereg-v1`).

| ID | Prediction | Metric |
|---|---|---|
| **G1** | GNN < BASE | false detections per image at equal recall R\* |
| **G2** | GNN > BASE | recall at equal false-detection rate F\* |
| **G3** | GNN < RGB | false detections per image at equal recall R\* |

- **R\*** = BASE-s0 recall on pooled T-shift at score ≥ 0.5, IoU 0.5, 9 classes pooled — identical to `prereg-v1`. Each arm's threshold is the highest one reaching R\* (flag if unreachable).
- **F\*** = BASE-s0 false detections per image on pooled T-shift at score ≥ 0.5. Each arm's threshold is the **lowest** one whose false detections per image ≤ F\*; G2 compares recall there.
- Paired bootstrap over T-shift images (B = 2,000, seed 0), R\*, F\* and all thresholds re-estimated in every replicate; one-sided p; **Holm over {G1, G2, G3}**, α = 0.05. One seed per arm (time budget): stated as a limitation.

**Descriptive (CIs, no tests):** mAP on T-clean, T-shift, T-texture and every subset for every arm; GNN − GNN-SHAPE (what size cues add); GNN-COLOR − GNN (what adding color does under shift); T-texture retention; false-detection types (invented, wrong class, poor localization, duplicate, near another class); outline failure rate per subset.

## 3. Arms

All arms re-score the **same BASE-s0 detections**; boxes and class labels are unchanged, only scores change.

| Arm | Second-stage input | Purpose |
|---|---|---|
| BASE | none (detector score) | the "before" |
| **GNN** | outline graph + size features | the method |
| RGB | the image crop (ResNet-18) | control: same pipeline, sees texture (G3) |
| GNN-SHAPE | outline graph only, size features zeroed | ablation: value of size cues |
| GNN-COLOR | outline graph + size + mean RGB at each contour point | ablation: does adding texture hurt under shift |

Priority if time runs short: BASE, GNN (G1, G2) → RGB (G3) → GNN-SHAPE → GNN-COLOR.

## 4. Data

- **Detector:** `~/snz-data/runs/BASE-s0/final.pt` (frozen recipe, amendment 2 of `prereg-v1`).
- **Second-stage training data:** BASE-s0 detections on **id-val** (1,248 clear-daytime BDD train images the detector never trained on, so it makes realistic mistakes). Images split by name with seed 0 into **fold A (50%)**, which trains the GNN/RGB networks, and **fold B (50%)**, which fits the fusion. Written to `manifests/gnn_fold_{a,b}.csv`.
- **Targets per detection** (own IoU against id-val labels): best IoU ≥ 0.5 with a real box → that box's class; IoU < 0.3 with every box (ignore regions included) → `background`; otherwise (0.3–0.5, or ≥ 0.5 only with an ignore region) → excluded from network training. Fusion target (fold B): the detection is a true positive at IoU 0.5 under the `prereg-v1` matching (`Matches.det_tp[:, 0]`).
- **Test inputs:** BASE-s0 `val-all` and `t-texture` detections (the files `prereg-v1` G6 produces). Footage (comma2k19) is optional follow-up.
- Training uses only clear-daytime images, so a gain under night/rain/snow/fog is a generalization result.

## 5. Components

### 5.1 Outlines (`snz gnn-outlines`)

- SAM 2.1 **hiera-small** via `SAM2ImagePredictor`, one `set_image` per image, all selected boxes as a batched box prompt, `multimask_output=False`, bf16 autocast on CUDA.
- Mask → largest external contour (OpenCV, `CHAIN_APPROX_NONE`) → **64 points** equally spaced by arc length, clockwise, starting at the top-most point.
- Failure (empty mask or < 3 contour points) → the box rectangle is used and `outline_ok = False`; failure rate is reported per subset.
- Also stored: box, score, label, image size, and the mean RGB of a 5×5 patch at each contour point (used **only** by GNN-COLOR).
- Fallback if SAM 2 cannot be installed in 15 minutes: OpenCV GrabCut initialized from the box on the crop resized to ≤ 128 px (5 iterations, multiprocessing). Recorded in `prereg-v2` if used.
- Output: one `.npz` per split. Computing outlines does not evaluate anything and is allowed before `prereg-v2`.

### 5.2 Shape graph (`snz.gnn.graph`)

- **Nodes (64):** contour points. Features (5): x and y relative to the centroid divided by s = √(polygon area) (translation/scale invariant); turning angle / π; turning angle × 64 / (perimeter / s) (discrete curvature); distance to centroid / s.
- **Edges:** fixed ring neighbors at offsets ±1, ±2, ±4 (K = 6).
- **Size features (6, per graph):** box width / image width, box height / image height, box bottom y / image height, box center x / image width, log(box width / box height), polygon area / box area.
- No pixel values in the GNN or GNN-SHAPE input. A unit test asserts features are identical for two images that differ only in color.

### 5.3 Networks (`snz.gnn.model`)

- **GNN:** plain PyTorch (no PyTorch Geometric: fixed edges, so no compiled extensions are needed on the GX10's Arm CPU). 3 EdgeConv layers (64, 64, 128; message MLP([hᵢ, hⱼ − hᵢ]) → max over neighbors), concatenated (256) → linear 512 → global max ‖ mean (1024) ‖ size MLP (6 → 32) → MLP 256 → dropout 0.3 → 10 logits (9 `EVAL_CLASSES` + `background`).
- **Training:** AdamW lr 1e-3, weight decay 1e-4, batch 512, 30 epochs, cosine schedule, cross-entropy with label smoothing 0.05; augmentation: random cyclic shift of the start point, horizontal mirror (x → −x, point order reversed), rotation ±10°, Gaussian noise σ = 0.01 on size features. Seed 0.
- **RGB control:** torchvision ResNet-18 `IMAGENET1K_V1`, crop = box enlarged 10% per side, resized to 112×112, 10 classes, AdamW lr 3e-4, batch 256, 8 epochs, horizontal flip; seed 0.
- **Fusion (per arm):** logistic regression on fold B, features [logit(detector score), log p(detected class), log p(background), log max p(other class)], L2 = 1e-4, Newton/IRLS. New score = predicted P(true positive).

### 5.4 Re-scoring and analysis

- `snz gnn-rescore`: writes `~/snz-data/gnn/runs/<ARM>-s0/eval/{val-all,t-texture}.detections.jsonl` in the `prereg-v1` format (same boxes/labels, new scores). Refuses test splits until tag `prereg-v2` exists (guard generalized with a `tag` argument).
- `snz gnn-analyze`: reuses `match_split`, `average_precision`, `RecallCurve`, `false_detections`, `bootstrap_weights`, `one_sided_p`, `percentile_ci`, `holm`; adds `threshold_for_false_rate` (lowest threshold with false detections per image ≤ target) for G2. Outputs `results/gnn/analysis.json` and `results/gnn/report.md`.

## 6. Roles and timeline (today, deadline 13:00 PDT)

| When | GX10 (Enoch) | Laptops (Aaron + Claude) |
|---|---|---|
| now | Stop G5 once `runs/BASE-s0/run.json` exists; run BASE-s0 on id-val (+ val-all, t-texture if missing); install SAM 2 | Implement Plan Tasks 1–4 |
| +45 min | Pull; `gnn-outlines` id-val, val-all, t-texture | Implement Tasks 5–7 |
| +75 min | `gnn-train` (GNN, RGB, ablations) | Write `prereg-v2` with model checksums; tag |
| +100 min | `gnn-rescore`, `gnn-analyze`; commit `results/gnn/` | README + paper results |

Compute: SAM 2 on ≈ 15k images at ≤ 50 boxes each (≈ 25–40 min); GNN and RGB training a few minutes each; analysis minutes.

## 7. Risks

| Risk | Mitigation |
|---|---|
| SAM 2 does not install on the GX10 | `SAM2_BUILD_CUDA=0`; checkpoint via Hugging Face or direct URL; GrabCut fallback after 15 min |
| Outlines poor at night | Report outline failure rate per subset; a null G1/G2 is reported as such |
| Too little time for all arms | Priority order in §3; un-run arms listed as not run |
| Small training set (fold A) | ≈ 15–20k detections expected; report fold-B AUROC of the fusion as a sanity check |
| GNN rejects real but unusual shapes | False-detection types and per-class AP show what changed |

## 8. Follow-up (after the deadline)

Cross-fitted detectors for more training data, 3 seeds, SAM outline quality against BDD's 10K instance masks, GNN re-scoring of the comma2k19 footage with 150 more blind-audit items, nuImages as a second test set.
