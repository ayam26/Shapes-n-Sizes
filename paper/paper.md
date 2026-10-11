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
predicted direction, Holm correction over the six confirmatory tests.

### 3.5 Footage audit

⟨Pending gate G7.⟩ Unfiltered frames every N seconds from all drives in a collection period;
seed-0 detectors thresholded at their full-val equal-recall points; 150 detections per arm
audited blind; 100 frames hand-labeled for recall.

## 4. Results

⟨Filled from `results/report.md`.⟩

- 4.1 Confirmatory tests (Table: H1b-STY, H1b-SIN, H2, H3, H4-STY, H4-SIN).
- 4.2 Accuracy by subset (Table: mAP/AP50 per arm, mean ± SD over seeds).
- 4.3 False detections by type at equal recall.
- 4.4 Clean cost, BOTH and additivity (exploratory).
- 4.5 T-corrupt (exploratory).
- 4.6 Footage audit.

## 5. Discussion

⟨After results.⟩ Points to address regardless of outcome: whether fine-tuning washes out
backbone shape bias (H3 vs. H1b-SIN); the T-texture confound for STY/BOTH; what a null result
means for texture-bias explanations of detector failures at night.

## 6. Limitations

- One detector family and one dataset; 2018 rather than 2020 BDD labels.
- T-fog has 5 images; dawn/dusk and overcast are descriptive only.
- Stylized training uses DTD textures rather than Painter by Numbers.
- The footage audit is single-region and single-camera; auditors are the authors.
- Equal-recall thresholds are fixed from full data, not re-estimated per bootstrap replicate.
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
