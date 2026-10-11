# Preregistration — Shape-biased road-object detection

**Status:** DRAFT. This document becomes binding when it is committed and tagged `prereg-v1`
(gate G4). No test split has been evaluated before that tag; `snz evaluate` refuses test
splits until the tag exists.

- Design spec: `docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md`
- Code: this repository at the tagged commit (`src/snz`), analysis in `src/snz/eval/analyze.py`
- Authors: the Shapes-n-Sizes team

## 1. Question

Do road-object detectors rely on texture rather than shape, and does shape-biased training
make them find more real objects and invent fewer fake ones under appearance shift?

## 2. Design

Four arms (2 × 2), three seeds each (0, 1, 2), 12 training runs:

| Arm | Backbone initialization | Training images |
|---|---|---|
| BASE | torchvision ResNet-50 `IMAGENET1K_V1` | original |
| STY | `IMAGENET1K_V1` | half stylized (P(original) = 0.5, P(each of 2 stylized copies) = 0.25) |
| SIN | Geirhos et al. Stylized-ImageNet ResNet-50 (`resnet50_train_60_epochs-c8e5653e`, sha256 `c8e5653e…0da3`) | original |
| BOTH | Stylized-ImageNet | half stylized |

Detector: torchvision Faster R-CNN ResNet-50 + FPN, freshly initialized FPN/RPN/ROI heads
(no COCO weights), frozen batch norm, 3 trainable backbone stages, bf16. Every arm uses the
identical recipe and number of iterations; the **final checkpoint** is evaluated (no selection).

Frozen recipe (`configs/recipe.toml`, recipe sha256 `<FILLED AT G4>`):

| Item | Value |
|---|---|
| batch size | `<G3>` |
| learning rate (SGD, momentum 0.9, wd 1e-4) | `<G3>` |
| iterations | `<G3>` |
| warmup / LR steps | 1000 iterations / `<G3>` |
| scale jitter (short side) | `<G3>`; evaluation at 720 (native 1280×720) |
| horizontal flip | p = 0.5 |

Stylization: AdaIN (pretrained `vgg_normalised.pth` / `decoder.pth`, α = 1.0); style images
from the Describable Textures Dataset (5,640), split 80/20 with seed 0 into training styles and
held-out test styles (`manifests/styles_{train,test}.csv`). T-texture uses test styles only.

## 3. Data

BDD100K 100K images with the **2018 detection labels**. Deviation from the spec, which named
the 2020 (det_20) labels: in October 2026 the official BDD100K servers were offline and the
det_20 labels could not be obtained; the maintainer's Berkeley mirror (`dl.yf.io`) serves the
images and the 2018 labels. Checksums: `manifests/downloads.md5`.

Label handling: 10 classes (pedestrian, rider, car, truck, bus, train, motorcycle, bicycle,
traffic light, traffic sign); boxes under 1 px dropped (483 train, 61 val). BDD's
`other vehicle` / `other person` / `trailer` boxes are ignore regions (mapped to car /
pedestrian / truck): not training targets, and detections on them are neither true nor false
positives (COCO crowd semantics). (The 2018 labels contain none of these categories.)

Splits (`manifests/*.csv`, image name + SHA-256; seed 0; no image in two splits, checked by
name and by hash; 0 cross-split duplicates found):

| Split | Source | Filter | Images |
|---|---|---|---|
| train | BDD train | daytime ∧ clear, minus id-val | 11,229 |
| id-val | BDD train | 10% of daytime ∧ clear | 1,248 |
| T-clean | BDD val | daytime ∧ clear | 1,764 |
| T-night | BDD val | night (any weather) | 3,929 |
| T-rain | BDD val | daytime ∧ rainy | 396 |
| T-snow | BDD val | daytime ∧ snowy | 422 |
| T-fog | BDD val | daytime ∧ foggy | 5 |
| T-shift (pooled) | union of night/rain/snow/fog | | 4,752 |
| T-texture | T-clean, stylized with held-out styles | | 1,764 |
| t-dusk, t-overcast (descriptive) | BDD val | dawn/dusk; daytime ∧ overcast | `<G4>` |
| val-all | all BDD val (footage operating point) | | 10,000 |

T-fog has 5 images and is reported descriptively only. Training and id-val use only
daytime ∧ clear images; id-val is the only data used to choose the recipe (G3).

## 4. Hypotheses (confirmatory family, Holm-corrected, α = 0.05)

Arm metric = mean over its 3 seeds. Each test compares one arm with BASE.

| ID | Prediction | Metric |
|---|---|---|
| H1b-STY | STY > BASE | retention = mAP(T-texture) / mAP(T-clean) |
| H1b-SIN | SIN > BASE | retention |
| H2 | STY > BASE | mAP on pooled T-shift |
| H3 | SIN > BASE | mAP on pooled T-shift |
| H4-STY | STY < BASE | false detections per image on T-shift at equal recall |
| H4-SIN | SIN < BASE | false detections per image on T-shift at equal recall |

Descriptive (no test): BASE retention with CI (H1a); clean-cost mAP difference on T-clean for
every arm (RQ4); BOTH vs BASE and additivity
((BOTH − BASE) − (STY − BASE) − (SIN − BASE)) on T-shift mAP; every subset separately.

T-corrupt (exploratory, point estimates only): a fixed random subset of 300 T-clean images
(seed 0, `manifests/t-corrupt.csv`), all 15 `imagecorruptions` corruptions at severities
1, 3 and 5 (reproducibly seeded per image), evaluated for the seed-0 run of each arm; reported
as mPC (mean mAP over corruption × severity) and rPC = mPC / clean mAP on the same subset.
Reduced from "all corruptions, severities 1–5, all of T-clean" to keep compute modest.

Known confound: STY and BOTH have seen stylization artifacts, which favors them on T-texture;
T-shift (natural shift) is therefore the primary test.

## 5. Metrics and analysis

- mAP = COCO mAP@[.5:.95] over the 9 classes excluding `train` (reported per class only);
  AP50 and per-class AP are reported. COCO matching (pycocotools semantics) computed once per
  run; AP from those matches, validated against pycocotools (agreement to 1e-6 in tests).
- Equal-recall operating point: R* = BASE seed-0 recall on pooled T-shift at score ≥ 0.5,
  IoU ≥ 0.5, 9 classes pooled. Each run's threshold is the highest score threshold whose
  T-shift recall ≥ R*; a run that cannot reach R* is evaluated at its maximum recall and
  flagged (flagged runs are listed).
- False detections per image at that threshold, by type, in order: duplicate (IoU ≥ 0.5 with
  an already-matched same-class box), wrong class (IoU ≥ 0.5 with a box of another class,
  including `train`), poor localization (0.1 ≤ IoU < 0.5 with a same-class box), invented
  (IoU < 0.1 with every box), and near another class (0.1 ≤ IoU < 0.5 with another class only;
  a residual type the spec did not name). Detections on ignore regions are excluded.
- Uncertainty: paired bootstrap over test images, B = 2,000, seed 0 (T-clean / T-texture
  share one weight matrix, T-shift another); the same resampled image multiset for every run
  in a replicate. 95% percentile CIs for arm − BASE; one-sided p = (1 + #replicates not in the
  predicted direction) / (B + 1). Thresholds are fixed from the full data, not re-estimated per
  replicate. Holm over the six tests; supported iff Holm-adjusted p < 0.05.
- Seed spread (SD, min, max) reported beside every estimate. Every subset is reported,
  including subsets where shape bias does not help.

Command: `snz analyze` (see `docs/runbook.md`, G6).

## 6. Unfiltered dashcam footage (RQ5, secondary, estimates with CIs, no test)

- Video only, no audio. Every drive recorded during a declared collection period; one frame
  every N seconds with N chosen so the total is ≥ 2,000 frames; no manual selection; only
  unreadable frames dropped, each drop logged; each drive tagged with time of day and weather.
- Models: seed-0 run of each arm; score threshold per arm = its full-val equal-recall threshold
  (R* defined on all BDD val by the same construction), fixed before any footage is processed.
- Counts of detections per frame and per class, per arm.
- Blind precision audit: 150 detections per arm sampled uniformly from all thresholded
  detections (600), shuffled, arm hidden, key stored separately. Labels: real object / correct
  class; real object / wrong class; nothing there (invented); unsure. Precision = correct /
  (items − unsure), Wilson 95% CI.
- Recall: 100 random frames fully labeled for the 9 classes without model pre-annotation.
- Headline: total detections; estimated real = detections × precision; estimated invented =
  detections × invented share; recall on labeled frames; all with CIs.
- Faces and plates are blurred in anything published.

## 7. Deviations log

| Date | Deviation | Reason |
|---|---|---|
| 2026-10-11 | 2018 BDD detection labels instead of det_20 | official servers offline; det_20 unobtainable |
| 2026-10-11 | false-detection residual type "near another class" | the spec's four types leave this case unclassified |
| 2026-10-11 | T-corrupt reduced to 300 images, severities 1/3/5, seed-0 runs | compute; exploratory only |
