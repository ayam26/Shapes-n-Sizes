# Does Shape Bias Help Road-Object Detectors See Through Night and Weather? A Preregistered 2×2 Study on BDD100K

*Draft — results sections are filled from `results/report.md` after gate G6. Numbers marked
⟨…⟩ are pending.*

## Abstract

ImageNet-trained convolutional networks classify largely by texture rather than shape, and
training on style-transferred images increases their shape bias. We ask whether shape bias
makes road-object detectors more robust when appearance changes but object shapes do not.
We train Faster R-CNN (ResNet-50 + FPN) detectors in a preregistered 2 × 2 design —
standard vs. Stylized-ImageNet backbone × normal vs. half-stylized training images, three
seeds each — on clear daytime BDD100K images only, and test them on night, rain, snow and
fog, on texture-scrambled copies of the clean test images, and on unfiltered dashcam
footage with a blind audit of real, invented and missed objects. ⟨Main result: H2/H3
supported or not, effect size with CI; H4; retention; clean cost.⟩ All code, split manifests,
the preregistration, and result summaries are released.

## 1. Introduction

Vision models used in driving can be confidently wrong: they miss real objects or report
objects that are not there. One proposed cause is texture bias. Geirhos et al. (2019) showed
that ImageNet-trained CNNs prefer texture over shape when the two conflict, and that training
on Stylized-ImageNet — ImageNet with textures replaced by AdaIN style transfer while shapes are
kept — increases shape bias and robustness to image distortions. Michaelis et al. (2019)
found that detectors trained on a mix of original and stylized images are more robust to
synthetic corruptions with little loss on clean data.

Natural appearance shifts in driving — darkness, rain, snow, fog — change texture and color
statistics far more than object outlines. If detectors rely on texture, shape-biased training
should help most exactly there. We test this directly, with three choices that make the answer
hard to fool ourselves about:

1. **Train on clear daytime only**, so every night and weather image is out of distribution.
2. **Separate the two levers** — a shape-biased backbone (SIN) and stylized fine-tuning data
   (STY) — in a 2 × 2 with three seeds per arm, identical recipes, and final checkpoints only.
3. **Preregister** hypotheses, metrics, operating points and the statistical analysis before any
   test image is evaluated, and enforce it in code (test splits are unreadable until the
   `prereg-v1` tag exists).

Beyond accuracy we measure *hallucinations*: false detections per image at equal recall, split
into duplicates, wrong classes, poor localization and invented objects, and, on our own
unfiltered dashcam footage, a blind audit of whether detections are real.

## 2. Related work

**Texture and shape bias.** Geirhos et al. (2019) introduced cue-conflict stimuli and
Stylized-ImageNet; later work studied shape bias across architectures and training recipes
(model-vs-human benchmark). **Robust detection.** Michaelis et al. (2019) benchmarked
detectors under corruptions (COCO-C, Cityscapes-C) and stylized training; Hendrycks &
Dietterich (2019) defined the corruption suite. **Driving under shift.** BDD100K (Yu et al.,
2020) provides weather and time-of-day tags; ACDC, DAWN, nuImages and SHIFT study adverse
conditions. **Hallucinations.** In vision-language models, POPE and DriveBench measure object
hallucination; for detectors we use false detections at matched recall.

## 3. Method

### 3.1 Detector and arms

torchvision Faster R-CNN with a ResNet-50 + FPN backbone; FPN, RPN and ROI heads are freshly
initialized in every arm (no COCO weights, which would carry ImageNet texture features into
the shape-biased arms). Batch norm is frozen; the last three backbone stages train.

| Arm | Backbone initialization | Fine-tuning images |
|---|---|---|
| BASE | ImageNet (`IMAGENET1K_V1`) | original |
| STY | ImageNet | half stylized |
| SIN | Stylized-ImageNet (Geirhos et al.) | original |
| BOTH | Stylized-ImageNet | half stylized |

Half stylized: each training sample is the original with probability 0.5, otherwise one of two
precomputed AdaIN-stylized copies (α = 1) made with styles from the Describable Textures
Dataset. Styles are split 80/20; held-out styles are used only for testing.

### 3.2 Data

BDD100K 100K images with its 2018 detection labels (10 classes; the 2020 labels were
unobtainable because the official servers were offline in 2026). Training: daytime ∧ clear
images (11,229), with 1,248 held out as id-val for recipe selection. Test sets from BDD val:
T-clean (daytime ∧ clear, 1,764), T-shift = night (3,929) ∪ daytime rain (396) ∪ snow (422)
∪ fog (5), T-texture (T-clean stylized with held-out styles), descriptive dawn/dusk and
overcast subsets, and an exploratory T-corrupt subset. Splits are content-hashed; no image
appears in two splits.

### 3.3 Training

Identical recipe for all 12 runs (⟨batch, LR, iterations, schedule⟩, chosen on id-val with
BASE seed 0 only), bf16, 1280×720 input with scale jitter and horizontal flips, on an ASUS
Ascent GX10 (NVIDIA GB10). Every random choice is seeded; runs resume exactly after
interruption.

### 3.4 Evaluation

COCO mAP@[.5:.95] over 9 classes (excluding the rare `train` class) with pycocotools
semantics. *Retention* = mAP(T-texture)/mAP(T-clean). *Equal-recall false detections*: R* is
BASE seed-0 recall on pooled T-shift at score ≥ 0.5 and IoU ≥ 0.5; each run is thresholded to
reach R*, and its unmatched detections per image are counted and typed. Uncertainty: paired
bootstrap over test images (B = 2,000), arm metric = mean over seeds, one-sided p in the
predicted direction, Holm correction over the six confirmatory tests; R* and every
threshold are re-estimated within each bootstrap replicate.

### 3.5 Footage audit

⟨Pending gate G7.⟩ Unfiltered frames every N seconds from all drives in a collection period;
seed-0 detectors thresholded at their full-val equal-recall points; 150 detections per arm
audited blind; 100 frames hand-labeled for recall.

## 4. Results

### 4.1 What was run

The study ran under a 5-hour budget, and every change was recorded before the data it affects
were evaluated (`prereg/prereg.md` amendment 2; `prereg/prereg-v2.md`). The CNN 2 × 2 was cut
to **one seed per arm at short side 384–448 with 4,500 iterations**. Training stopped after
BASE-s0 and STY-s0 when the project moved to the GNN shape verifier, which uses BASE-s0 as stage
1. The `prereg-v1` confirmatory tests (H1b–H4) therefore were **not run**: SIN and BOTH were
never trained. The GNN hypotheses G1–G3 (`prereg-v2`) were run in full: all five
second-stage arms (BASE, GNN, RGB, GNN-SHAPE, GNN-COLOR) re-scored the same 483,120 BASE-s0
detections on the 10,000 BDD100K val images. SAM 2 produced a usable outline for
99.998% of them (100.0% on every subset).

### 4.2 CNN arms (descriptive)

| Arm | val-all mAP / AP50 | T-texture mAP / AP50 |
|---|---|---|
| BASE-s0 | 17.0 / 36.0 | 2.3 / 5.0 |
| STY-s0 | 13.0 / 27.9 | 7.6 / 16.0 |

pycocotools on all detections (score ≥ 0.05). Training on half-stylized images more than
tripled accuracy on texture-scrambled images (2.3 → 7.6 mAP), but cost 4.0 mAP on ordinary val
images. Both detectors are weak by design (≈3 epochs at reduced input size), so these numbers
show direction, not ceiling performance. With one seed and no SIN arm, nothing here is a
confirmatory test.

### 4.3 Confirmatory tests (G1–G3)

Pooled T-shift (night ∪ rain ∪ snow ∪ fog, 4,752 images). R* = 0.548 (BASE-s0 recall at score
≥ 0.5); F* = 10.19 false detections per image (BASE-s0 at score ≥ 0.5). Paired bootstrap,
B = 2,000, with R*, F* and all thresholds re-estimated in every replicate; Holm over three tests.
No arm was flagged (every arm reached R* and F*).

| Test | Comparison | Estimate | 95% CI | Holm p | Supported |
|---|---|---|---|---|---|
| G1 | GNN − BASE, false detections/image at R* | −0.345 | [−0.429, −0.236] | 0.0015 | **yes** |
| G2 | GNN − BASE, recall at F* | +0.0036 | [+0.0026, +0.0045] | 0.0015 | **yes** |
| G3 | GNN − RGB, false detections/image at R* | +0.309 | [+0.160, +0.447] | 1.0 | **no** |

The texture-blind GNN removed 3.4% of BASE's false detections at equal recall (10.19 → 9.84
per image; 48,397 → 46,756 in total) and found slightly more objects at an equal false-alarm
rate (recall 0.548 → 0.551). G3 failed in the opposite direction from the hypothesis: the RGB
control, which sees texture, removed **6.4%** (10.19 → 9.53 per image).

### 4.4 What changed, by false-detection type (per image, at R*)

| Arm | duplicate | wrong class | poor localization | invented | near another class |
|---|---|---|---|---|---|
| BASE | 0.578 | 0.282 | 5.209 | 3.886 | 0.230 |
| GNN | 0.594 | 0.363 | 5.058 | **3.593** | 0.230 |
| GNN-SHAPE | 0.594 | 0.335 | 5.076 | 3.641 | 0.229 |
| GNN-COLOR | 0.603 | 0.362 | 5.071 | 3.691 | 0.234 |
| RGB | 0.707 | 0.304 | 5.316 | **3.009** | 0.193 |

Most of the GNN's gain is in *invented* objects (−7.5%) and poorly localized boxes (−2.9%); it
added a few wrong-class detections (+0.08 per image). The RGB control cut invented objects by
23% but kept more duplicates and poorly localized boxes. Poor localization is the largest error
type for every arm (≈ 51% of BASE's false detections), and no second stage that keeps the
boxes fixed can repair it.

### 4.5 Accuracy by subset (mAP × 100)

| Subset (images) | BASE | GNN | GNN-SHAPE | GNN-COLOR | RGB |
|---|---|---|---|---|---|
| T-clean (1,764) | 18.6 | 19.0 | 18.9 | 18.9 | 19.6 |
| T-shift (4,752) | 13.8 | 14.1 | 14.0 | 14.0 | 14.4 |
| night (3,929) | 12.4 | 12.7 | 12.7 | 12.6 | 12.8 |
| rain (396) | 19.5 | 19.7 | 19.6 | 19.7 | 20.0 |
| snow (422) | 17.1 | 17.0 | 17.1 | 17.0 | 18.0 |
| fog (5) | 23.2 | 21.9 | 22.1 | 23.2 | 23.9 |
| dusk (778) | 16.5 | 16.8 | 16.7 | 16.8 | 17.9 |
| overcast (1,038) | 18.6 | 18.8 | 18.8 | 18.8 | 19.5 |
| T-texture (1,764) | 2.3 | 2.3 | 2.4 | 2.4 | 2.3 |

These are computed on the selected detection set (top 50 per image), so BASE differs slightly
from §4.2. The second stages change mAP by at most +0.8 points, and the RGB control leads
on every subset except T-texture. Re-scoring does not rescue T-texture (≈ 2.4 mAP for every
arm): when stage 1 misses or mislabels objects on texture-scrambled images, re-ranking its boxes
cannot help.

### 4.6 Ablations (descriptive, 95% CIs)

- **Size cues add nothing measurable:** GNN − GNN-SHAPE = −0.036 false detections/image
  [−0.090, +0.041].
- **Adding color to the outline graph hurts under shift:** GNN-COLOR − GNN = +0.122
  [+0.026, +0.191] false detections/image and −0.0015 [−0.0022, −0.0006] recall at F*.
- **Shape alone still helps:** GNN-SHAPE − BASE = −0.310 [−0.400, −0.220].
- **Texture helps more:** RGB − BASE = −0.655 [−0.777, −0.494] false detections/image,
  +0.0065 [+0.0050, +0.0079] recall at F*, +0.53 [+0.29, +0.79] mAP points on T-shift.

On the clean id-val fold used to fit the fusion, the networks barely changed AUROC
(BASE 0.908, GNN 0.910, RGB 0.918), so the gains under shift are not an artifact of an
over-fitted fusion.

### 4.7 Unfiltered dashcam footage (blind audit)

⟨Pending: blind precision audit of BASE, GNN and RGB detections on comma2k19 footage (2,270
frames, 100 detections per arm). Filled from `results/gnn/footage.json`.⟩

### 4.8 Summary

Shape-only re-scoring reduced false detections at matched recall under night and weather shift,
and the effect is statistically clear (G1, G2) but small. A same-size second stage that sees
texture did better on every measure (G3 not supported), and adding color to the shape graph made
it worse. Under this setup we find **no evidence that discarding texture is what helps**. The
gains come from a second look at each box, and that second look works better with texture.

## 5. Discussion

⟨After results.⟩ Points to address regardless of outcome: whether fine-tuning washes out
backbone shape bias (H3 vs. H1b-SIN); the T-texture confound for STY/BOTH; what a null result
means for texture-bias explanations of detector failures at night.

## 6. Limitations

- One detector family and one dataset; 2018 rather than 2020 BDD labels.
- T-fog has 5 images; dawn/dusk and overcast are descriptive only.
- Stylized training uses DTD textures rather than Painter by Numbers.
- The footage audit is single-region and single-camera; auditors are the authors.
- No claim of improved vehicle safety is made.

## 7. Ethics and data

BDD100K is used under its research license; no images are redistributed. Our footage is
video only (no audio), kept private, and faces and plates are blurred in any published figure.

## References

- Geirhos, R., et al. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. ICLR.
- Michaelis, C., et al. (2019). Benchmarking robustness in object detection: Autonomous driving when winter is coming. arXiv:1907.07484.
- Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization. ICCV.
- Ren, S., et al. (2015). Faster R-CNN. NeurIPS. · Lin, T.-Y., et al. (2017). Feature pyramid networks. CVPR.
- Yu, F., et al. (2020). BDD100K. CVPR.
- Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. ICLR.
- Cimpoi, M., et al. (2014). Describing textures in the wild. CVPR.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scand. J. Statistics.
