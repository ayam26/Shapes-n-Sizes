# Preregistration — Shape-biased road-object detection

**Status:** FROZEN at git tag `prereg-v1` (gate G4, 2026-10-11). No test split had been
evaluated before this tag: `snz evaluate` refuses every split except train / id-val until the
tag exists, and only id-val was evaluated, for the G3 pilot (`results/pilot/`).

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

Frozen recipe (`configs/recipe.toml`, recipe sha256
`71248101b500052b140fee8814d049662af72a6111db2d9421acc974761854f2`), chosen by the G3 pilot
(BASE seed 0, 6,000 iterations, id-val mAP 24.5 at lr 0.01 vs. 25.9 at lr 0.02; 1.19 it/s):

| Item | Value |
|---|---|
| batch size | 8 |
| learning rate (SGD, momentum 0.9, wd 1e-4) | 0.02 |
| iterations | 18,000 (~12.8 epochs) |
| warmup / LR steps | 1,000 iterations linear warmup / ×0.1 at 12,060 and 16,020 |
| scale jitter (short side) | {576, 608, 640, 672, 704, 720}; evaluation at 720 (native 1280×720) |
| horizontal flip | p = 0.5 |
| precision | bf16 autocast; final checkpoint evaluated |

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
| t-dusk (descriptive) | BDD val | dawn/dusk (any weather) | 778 |
| t-overcast (descriptive) | BDD val | daytime ∧ overcast | 1,038 |
| val-all | all BDD val (footage operating point) | | 10,000 |

T-fog has 5 images and is reported descriptively only. One t-overcast image was
byte-identical to a t-rain image under another name; it was dropped from t-overcast
(`manifests/dropped_duplicates.csv`).

Manifest SHA-256 (first 16 hex): train `86b0fe420cb91c41`, id-val `0c8536f52b8eed9e`,
t-clean `615ee70f2395dde6`, t-night `999b57caec98c625`, t-rain `9fd175dcdc476ab5`,
t-snow `d47e5bcab7e12e68`, t-fog `3bdcc1133c65b4b6`, t-dusk `6ea405342a15d9e5`,
t-overcast `93e7c0d17e08ff68`, val-all `7141fbb5460338d3`, styles_train `9013f9468ec8c06c`,
styles_test `ac07114010a4561f`, stylized_train `d19281eeafc08d0d`,
stylized_t-texture `3c4957a7f5417eb2`. Training and id-val use only
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
  T-shift recall ≥ R*; a run that cannot reach R* is evaluated at its maximum recall (the
  highest threshold attaining it) and flagged (flagged runs are listed).
- False detections per image at that threshold, by type, in order: duplicate (IoU ≥ 0.5 with
  an already-matched same-class box), wrong class (IoU ≥ 0.5 with a box of another class,
  including `train`), poor localization (0.1 ≤ IoU < 0.5 with a same-class box), invented
  (IoU < 0.1 with every ground-truth box, ignore regions included), and near another class (0.1 ≤ IoU < 0.5 with another class only;
  a residual type the spec did not name). Detections on ignore regions are excluded.
- Uncertainty: paired bootstrap over test images, B = 2,000, seed 0 (T-clean / T-texture
  share one weight matrix, T-shift another); the same resampled image multiset for every run
  in a replicate. 95% percentile CIs for arm − BASE; one-sided p = (1 + #replicates not in the
  predicted direction) / (B + 1). R* and every run's equal-recall threshold are re-estimated
  inside each replicate, so threshold-estimation uncertainty is part of the H4 intervals.
  Holm over all six tests (a test that cannot be computed, e.g. a missing arm, enters the family
  with p = 1); supported iff Holm-adjusted p < 0.05. BASE retention (H1a) and every arm − BASE
  subset difference get 95% bootstrap CIs. Our AP is checked against pycocotools on every
  run's val-all and T-texture predictions (agreement within 0.1 mAP points required).
- Seed spread (SD, min, max) reported beside every estimate. Every subset is reported,
  including subsets where shape bias does not help.

Command: `snz analyze` (see `docs/runbook.md`, G6).

## 6. Unfiltered dashcam footage (RQ5, secondary, estimates with CIs, no test)

- Video only, no audio. Every drive recorded during a declared collection period; one frame
  every N seconds with N chosen so the total is ≥ 2,000 frames; no manual selection; only
  unreadable frames dropped, each drop logged; each drive tagged with time of day and weather.
- Models: seed-0 run of each arm; score threshold per arm = its full-val equal-recall threshold
  (R* defined on all BDD val by the same construction), fixed before any footage is processed.
- Counts of detections per frame and per class, per arm. As on BDD, only the 9 evaluated
  classes are counted, sampled and audited (`train` detections are left out).
- Blind precision audit: 150 detections per arm sampled uniformly from all thresholded
  detections (600), shuffled, arm hidden, key stored separately. Labels: real object / correct
  class; real object / wrong class; nothing there (invented); unsure. Precision = correct /
  (items − unsure), Wilson 95% CI.
- Recall: 100 frames drawn with `snz footage-label-sample` (seed 0) from the footage manifest,
  fully labeled for the 9 classes without model pre-annotation; recall and false detections
  per frame with bootstrap-over-frames 95% CIs.
- Headline: total detections; estimated real = detections × precision; estimated invented =
  detections × invented share; recall on labeled frames; all with CIs.
- Faces and plates are blurred in anything published.

## 7. Deviations log

| Date | Deviation | Reason |
|---|---|---|
| 2026-10-11 | 2018 BDD detection labels instead of det_20 | official servers offline; det_20 unobtainable |
| 2026-10-11 | false-detection residual type "near another class" | the spec's four types leave this case unclassified |
| 2026-10-11 | T-corrupt reduced to 300 images, severities 1/3/5, seed-0 runs | compute; exploratory only |
| 2026-10-11 | schedule 18,000 iterations (spec placeholder 27,000) | measured 1.19 it/s; 12 runs in ~2.1 GX10-days (spec §5.4) |
| 2026-10-11 | t-overcast drops 1 image identical to a t-rain image | content-hash disjointness |

### Amendment after `prereg-v1` (2026-10-11, before any footage was sampled or any model ran on it)

| Change | Reason |
|---|---|
| RQ5 footage: instead of the team's own drives, the open comma2k19 dataset (comma.ai, MIT license), **Chunk_1, every segment** (~3.3 h of commute driving, 1164×874 road camera, raw HEVC at a declared 20 fps). The sampling rule, thresholds, audit and recall protocol are otherwise unchanged. | Decision of the project lead: use open-source dashcam footage. Using a whole chunk keeps the "no hand selection" rule. |
| Drives are tagged time of day / weather "unknown" (comma2k19 has no such tags). | Not available in the source. |
| comma2k19 uses a different camera and region than BDD100K, so RQ5 also tests transfer across cameras. | Stated as a limitation. |
