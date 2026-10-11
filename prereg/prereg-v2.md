# Preregistration v2 — Texture-blind GNN shape verifier

**Status:** frozen at git tag `prereg-v2` (2026-10-11), after the second-stage networks were
trained on id-val and **before** any GNN/RGB score was computed on a test split. `snz
gnn-rescore` and `snz gnn-analyze` refuse every split except id-val until this tag exists.

- Design spec: `docs/superpowers/specs/2026-10-11-gnn-shape-verifier-design.md`
- Plan: `docs/superpowers/plans/2026-10-11-gnn-shape-verifier.md`
- Builds on `prereg/prereg.md` (`prereg-v1`, amendment 2): BASE-s0 is stage 1; CNN training
  stopped after BASE-s0 and STY-s0 (STY is reported only as an existing CNN baseline).

## 1. Pipeline

BASE-s0 Faster R-CNN detections (score ≥ 0.05, top 50 per image, 9 evaluated classes) →
SAM 2.1 hiera-small, one box prompt per detection → largest external contour → 64 points by arc
length (clockwise, from the top-most point) → shape graph → second stage → logistic fusion with
the detector score → new score. Boxes and labels never change; only scores do. **The same
selected detection set is evaluated for every arm, BASE included.**

Segmenter used: **SAM 2.1 hiera-small** (`facebook/sam2.1-hiera-small`), masks thresholded on
the GPU. Outline success rate on id-val: **1.000** (59,587 outlines on 1,248 images).

## 2. Arms

| Arm | Second-stage input |
|---|---|
| BASE | none (detector score over the selected set) |
| GNN | outline graph (5 node features) + 6 size features; **no pixel values** |
| RGB | image crop (box + 10%, 112×112), ResNet-18 `IMAGENET1K_V1` (control that sees texture) |
| GNN-SHAPE | outline graph only (size features zeroed) |
| GNN-COLOR | outline graph + size + mean RGB of a 5×5 patch at each contour point |

Networks: GNN = EdgeConv 64-64-128 on a fixed ring graph (offsets ±1, ±2, ±4), linear 512,
max‖mean pooling, size MLP 6→32, MLP 256, dropout 0.3, 10 outputs (9 classes + background);
AdamW lr 1e-3, wd 1e-4, batch 512, 30 epochs, cosine, label smoothing 0.05, augmentation
(cyclic shift, mirror, ±10° rotation, size noise σ 0.01). RGB = ResNet-18, AdamW lr 3e-4,
batch 256, 8 epochs, horizontal flip. Fusion = logistic regression (IRLS, L2 1e-4) on
[logit(detector score), log p(detected class), log p(background), log max p(other class)].
Seed 0 everywhere; one seed per arm (time budget) — a stated limitation.

## 3. Data

- Second-stage training: BASE-s0 detections on **id-val** (1,248 clear-daytime BDD train images
  BASE never trained on). Split by image name (seed 0) into fold A (networks) and fold B
  (fusion): `manifests/gnn_fold_a.csv` (sha256 `ec6d03252d349e3b72ba856aaa3ca8912e255b276ddf96d145a6fcd88e8d4d87`), `manifests/gnn_fold_b.csv`
  (sha256 `b24ceaeeb7910bf629c51e922bbbdc77feb9682309ca6bc9df7c3ff122be4ef4`).
- Network targets: best IoU ≥ 0.5 with a real box → its class; IoU < 0.3 with every box (ignore
  regions included) → background; otherwise excluded. Fold A counts: `excluded 5,833, pedestrian 528, rider 60, car 6,542, truck 666, bus 314, motorcycle 102, bicycle 67, traffic light 1,073, traffic sign 1,653, background 12,916`.
- Fusion target: true positive at IoU 0.5 under `prereg-v1` matching. Fold B: `8,563 true positives of 29,833 detections`.
- Fold-B AUROC of the fused score (sanity check, not a hypothesis): `BASE 0.9080, GNN 0.9100, RGB 0.9176, GNN-SHAPE 0.9101, GNN-COLOR 0.9102`.
- Model checksums (sha256): `GNN.pt `7ef04f4b2274dd3fefd0ca7bcce10983553e91fafb3d009d5255a94cf96ff89a`; RGB.pt `314afa5aa0c078445232f6ec711d85235d23382d2ee454cf8483604a200d1215`; GNN-SHAPE.pt `2b525fcf06ea94f14dee66912a9c107d8dedc8ff56d59d2b98e84dafc898dd22`; GNN-COLOR.pt `bd4bfb38bb5dfcddf3fa831cb1708afbe2b5022f46173b331f0e4755d1b07d78``.
- Test: BDD100K val (`val-all`, sliced into the `prereg-v1` subsets) and T-texture.

## 4. Hypotheses (confirmatory; pooled T-shift = night ∪ rain ∪ snow ∪ fog; Holm over G1–G3, α = 0.05)

| ID | Prediction | Metric |
|---|---|---|
| G1 | GNN < BASE | false detections per image at equal recall R* |
| G2 | GNN > BASE | recall at equal false-detection rate F* |
| G3 | GNN < RGB | false detections per image at equal recall R* |

- R* = BASE-s0 recall on pooled T-shift at score ≥ 0.5, IoU 0.5, 9 classes pooled (as in
  `prereg-v1`); each arm's threshold = highest reaching R* (flagged if unreachable).
- F* = BASE-s0 false detections per image on pooled T-shift at score ≥ 0.5; each arm's
  threshold = lowest with false detections per image ≤ F*; G2 compares recall there.
- Paired bootstrap over T-shift images, B = 2,000, seed 0; R*, F* and every threshold
  re-estimated in each replicate; one-sided p = (1 + #replicates not in the predicted
  direction) / (B + 1); Holm over {G1, G2, G3}; a missing arm enters with p = 1.

**Descriptive (CIs, no tests):** mAP per subset and arm; GNN − GNN-SHAPE (value of size cues);
GNN-COLOR − GNN (adding color under shift); RGB − BASE; T-texture retention; false-detection
types at R*; outline success rate per subset.

**Known limit:** a picture of a truck has a truck's outline, so flat depictions pass a
shape-only check. The GNN targets texture-driven errors, not depiction confusion.

## 5. Deviations log

| Date | Deviation | Reason |
|---|---|---|
| 2026-10-11 | `sam2` installed into the GX10 venv, not locked in `uv.lock` | sdist-only package that imports torch at build time |
| 2026-10-11 | SAM 2 masks thresholded and cropped to each box (+10%) on the GPU before copying | 8× speed-up; contours come from inside the box either way |
| 2026-10-11 (after G1–G3, before any footage label) | Footage audit (RQ5) run for BASE, GNN, RGB only: 100 blind detections per arm (300 items), thresholds = each arm's full-val equal-recall point (R*_val = BASE-s0 full-val recall at score ≥ 0.5). The 100-frame recall labeling is not run. | under 1 hour left; recall labeling takes 1.5–2.5 h |
