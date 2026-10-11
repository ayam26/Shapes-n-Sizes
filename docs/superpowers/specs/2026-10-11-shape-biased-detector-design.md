# Shape-Biased Road-Object Detection — Research Design Spec

- **Date:** 2026-10-11
- **Status:** design approved in conversation; awaiting review of this written spec
- **Supersedes:** an earlier VLM depiction-cue design and the hackathon Perception Lab spec (both removed; see git history)
- **Pilot work:** `perception-lab/` (hackathon VLM testbed and results), cited as pilot evidence
- **Deliverable:** an arXiv / workshop-style paper with code, split manifests, preregistration, and result summaries

## 1. Summary

Vision models used for driving can make confident mistakes — missing real objects or "hallucinating" objects that are not there. One proposed cause is **texture bias**: ImageNet-trained convolutional networks classify largely by surface texture rather than object shape (Geirhos et al., 2019). Training on images whose textures are replaced by style transfer, while object shapes are preserved, increases shape bias, and Michaelis et al. (2019) showed that such stylized training makes object detectors more robust to image corruptions.

This project trains a **road-object detector on the ASUS Ascent GX10** in four arms — a 2×2 of *backbone initialization* (standard ImageNet vs. shape-biased Stylized-ImageNet) × *fine-tuning data* (normal vs. half-stylized dashcam images) — and asks whether shape-biased training:

1. reduces reliance on texture (premise test),
2. improves detection under natural appearance shifts (night, rain, snow, fog) when trained only on clear daytime images,
3. reduces invented detections ("hallucinations") at equal recall,
4. at what cost to clean daytime accuracy, and
5. whether the gains hold on **unfiltered dashcam footage recorded by the user**, measured as estimated real, invented, and missed objects.

The baseline ("before") is the same detector trained the standard way. A null or negative result is a valid result and is reported as such.

**Scope note.** Shape bias targets texture-driven errors (unusual paint or wraps, weather, night, confusing backgrounds). It does not address flat-depiction confusion (a picture of a truck still has a truck's outline); the Perception Lab renders appear only as a small exploratory probe.

## 2. Goals, non-goals, success criteria

**Goals**

- A preregistered 2×2 training study (3 seeds per arm, 12 runs) on BDD100K, evaluated on held-out conditions.
- A blind audit of detections on the user's unfiltered dashcam footage.
- Reproducible splits (manifests with hashes), frozen recipe, and recomputable result summaries.
- A paper draft.

**Non-goals**

- A driving controller, collision prediction, or any claim of improved vehicle safety.
- Vision-language models as the trained system (the Perception Lab covers VLMs; not extended here).
- Training a backbone on Stylized-ImageNet ourselves (published weights are used).
- Redistributing BDD100K images or labels.

**Success criteria**

- Recipe and preregistration frozen (git tag) before any evaluation on test splits.
- All 12 runs evaluated on every test split; all hypotheses reported regardless of outcome.
- Footage audit completed under the preregistered sampling rule and blinding.

## 3. Research questions and hypotheses

Frozen in `prereg/prereg.md` (tag `prereg-v1`) at gate G4.

Arms: **BASE** = ImageNet backbone + normal data; **STY** = ImageNet backbone + half-stylized data; **SIN** = Stylized-ImageNet backbone + normal data; **BOTH** = Stylized-ImageNet backbone + half-stylized data.

**RQ1 — Texture reliance (premise test).** *Retention* = mAP(T-texture) / mAP(T-clean).

- **H1a (descriptive):** BASE retention, reported with 95% CI.
- **H1b:** retention is higher for STY than BASE, and higher for SIN than BASE (two tests).

**RQ2 — Robustness to natural appearance shift (primary).** All arms train on clear daytime only.

- **H2:** STY > BASE in mAP on T-shift (pooled).
- **H3:** SIN > BASE in mAP on T-shift (pooled).
- Exploratory: BOTH vs. BASE; additivity of the two levers; each shift subset separately.

**RQ3 — Hallucinations at equal recall.**

- **H4:** at the T-shift equal-recall operating point (§7.3), STY and SIN each produce fewer false detections per image than BASE (two tests). The primary false-detection type of interest, *invented objects*, is reported alongside the total.

**RQ4 — Clean cost.** mAP difference on T-clean for each arm vs. BASE, reported with CIs (no directional hypothesis).

**RQ5 — Unfiltered footage (preregistered secondary analysis, estimates with CIs, no hypothesis test).** For each arm vs. BASE: total detections, estimated real detections, estimated invented detections, estimated recall.

**Exploratory:** robustness on T-corrupt; detections on Perception Lab billboard-truck renders.

## 4. Data

### 4.1 BDD100K

- Official BDD100K images (1280×720) and current official detection labels (version recorded in the manifest), 10 classes: car, truck, bus, person, rider, bike, motor, traffic light, traffic sign, train. Attributes used: `timeofday` ∈ {daytime, night, dawn/dusk, undefined}, `weather` ∈ {clear, partly cloudy, overcast, rainy, snowy, foggy, undefined}.
- License: BDD100K terms permit educational, research, and not-for-profit use. The terms are checked on the official site at G2 and recorded; no images or labels are committed to this repository.
- Downloaded by the user (registration required) onto the GX10.

### 4.2 Splits

| Split | Source | Filter | Use |
|---|---|---|---|
| train | BDD train | daytime ∧ clear, minus the validation holdout | training |
| id-val | BDD train | 10% of daytime ∧ clear, seeded random | pilot tuning and monitoring only |
| T-clean | BDD val | daytime ∧ clear | RQ4; base for T-texture and T-corrupt |
| T-shift: night | BDD val | night (any weather) | RQ2, RQ3 |
| T-shift: rain / snow / fog | BDD val | daytime ∧ rainy / snowy / foggy | RQ2, RQ3 |
| T-texture | T-clean | stylized with held-out style images | RQ1 |
| T-corrupt | T-clean | `imagecorruptions` (all corruption types, severities 1–5) | exploratory |

- T-shift (pooled) = union of the four shift subsets. Weather subsets are restricted to daytime to separate weather from darkness.
- Dawn/dusk, partly cloudy, overcast, and undefined attributes are excluded from hypothesis sets; dawn/dusk and overcast are reported descriptively.
- Subsets with few images (fog expected) are reported with CIs and carry no separate hypothesis.
- Manifests (`manifests/<split>.csv`: image name, SHA-256 of the image file) are committed. Tests assert no image appears in more than one of train, id-val, and any test split.

### 4.3 Stylization

- AdaIN style transfer (Huang & Belongie, 2017) with published pretrained weights, full stylization (α = 1.0), as used to build Stylized-ImageNet.
- Style images: Painter by Numbers (used by the original papers) or the Describable Textures Dataset, whichever passes the license check at G2. The style pool is split 80/20 into **training styles** and **test styles** with a fixed seed; T-texture uses test styles only.
- Training: 2 stylized copies per training image, each with a randomly drawn training style (seeded), precomputed on the GX10. In STY and BOTH, each sample is drawn from {original, copy 1, copy 2} with P(original) = 0.5, P(each copy) = 0.25.
- Bounding boxes are unchanged (stylization preserves geometry and image size; a test checks image size).
- **Known confound:** STY/BOTH have seen stylization artifacts, which favors them on T-texture. T-shift (natural shifts) is therefore the primary test; the paper states this.

### 4.4 User dashcam footage

- Video only, no audio (California requires all-party consent for recording conversations).
- **Unfiltered sampling rule (frozen in the preregistration):** every drive recorded during a declared collection period; one frame every N seconds, with N chosen so the total is ≥ 2,000 frames; no manual selection. Only unreadable frames (e.g., all-black, corrupted) are dropped, and every drop is logged. Each drive is tagged with time of day and weather.
- Raw footage and frames stay private on the GX10. Anything published (audit examples, figures) has faces and license plates blurred.
- The audit and labeling are performed by the researchers; no outside participants are involved.

## 5. Detector and training

### 5.1 Model

torchvision Faster R-CNN with ResNet-50 + FPN (Ren et al., 2015; Lin et al., 2017). Reasons: the Stylized-ImageNet weights are a standard torchvision ResNet-50; comparable to Michaelis et al.; BSD license.

### 5.2 Backbone initializations

- **ImageNet:** torchvision `ResNet50_Weights.IMAGENET1K_V1` (trained with the classic recipe, matching the Stylized-ImageNet training recipe more closely than V2's heavy-augmentation recipe).
- **Stylized-ImageNet:** Geirhos et al.'s ResNet-50 trained on Stylized-ImageNet only (`texture-vs-shape` repository), the most shape-biased variant released. Checksum recorded.
- Detection layers (FPN, RPN, ROI heads) are freshly initialized in every arm. No COCO-pretrained detector weights are used, because they derive from an ImageNet backbone and would contaminate the shape-biased arms.

### 5.3 Recipe (identical across arms)

- Fixed in this spec: input at native 1280×720 with scale jitter and horizontal flip (p = 0.5); frozen batch norm in the backbone; torchvision default trainable backbone layers (3); bf16 mixed precision; identical number of iterations; evaluate the **final checkpoint** (no checkpoint selection).
- Fixed by the pilot (G3), then frozen in `configs/recipe.toml`: learning rate, schedule length, batch size, scale-jitter range.
- Seeds: 3 per arm (0, 1, 2). A seed controls head initialization, data order, augmentation, and stylized-copy choice; the precomputed stylized pool is shared.

### 5.4 Pilot (G3)

BASE arm only, seed 0, on train with id-val for monitoring. Chooses the recipe items above and measures throughput. Estimated 4–5 GPU-hours per full run (≈ 2–3 GX10-days for 12 runs) until measured. If the measured budget is too large, the schedule or input scale is reduced **identically for all arms** before freezing.

### 5.5 Toolchain

PyTorch + torchvision on the GX10 (Arm64 + NVIDIA GB10). G1 smoke test: both backbones load into the detector and a 50-iteration training run completes. If standard wheels fail, NVIDIA's PyTorch containers are the fallback.

## 6. Footage audit protocol (RQ5)

- Models: seed-0 run of each arm. Score threshold per arm = the full-BDD-val equal-recall threshold (§7.3), fixed before any footage is processed.
- **Counts:** detections per frame and per class, per arm.
- **Blind precision audit:** sample 150 detections per arm uniformly at random from all thresholded detections (600 items). Present them in shuffled order in a static HTML audit page showing the frame with the box and predicted class; the arm is hidden, and the sample-to-arm key is stored separately. Labels: *real object, correct class* / *real object, wrong class* / *nothing there (invented)* / *unsure*. Precision = correct / (items − unsure), Wilson 95% CI; unsure counts reported.
- **Recall check:** 100 frames sampled at random from the footage manifest are fully labeled (all objects of the 9 evaluated classes) with a local tool (CVAT or Label Studio) **without any model pre-annotation**. Each arm's recall and false detections per frame are computed directly on these frames.
- **Headline table (per arm vs. BASE):** total detections; estimated real detections = detections × precision; estimated invented detections = detections × invented share; recall on the labeled frames. All with CIs.
- Optional: a second auditor on a 20% overlap; agreement reported (Cohen's κ).

## 7. Analysis plan

Frozen in the preregistration.

### 7.1 Metrics

- COCO-style mAP@[0.5:0.95] and AP50, plus per-class AP. Primary mAP averages the 9 classes excluding `train` (too rare for stable AP); `train` AP is reported per class only.
- Per-image matching is computed once per run; bootstrap AP is computed from these matches with a vectorized implementation validated against `pycocotools` on full splits (agreement within 0.1 mAP points).

### 7.2 Comparisons and uncertainty

- Each hypothesis compares one arm against BASE on one metric. Arm metric = mean over its 3 seeds.
- Paired bootstrap over test images (B = 2,000; the same resampled image set for all runs in a replicate); 95% percentile CIs for arm − BASE differences; one-sided bootstrap p-values in the predicted direction.
- Holm correction across the confirmatory family {H1b-STY, H1b-SIN, H2, H3, H4-STY, H4-SIN}, α = 0.05. A hypothesis is supported if its Holm-adjusted p < 0.05.
- Seed spread (min/max and SD per arm) reported beside every estimate. BOTH and additivity are exploratory, with CIs and no tests.

### 7.3 Equal-recall operating points and false-detection types

- **T-shift operating point (H4):** R\* = BASE seed-0 recall on T-shift (pooled) at score ≥ 0.5, IoU ≥ 0.5, across the 9 classes. For every run, its threshold is set so its T-shift recall equals R\*; false detections per image are counted at that threshold. If a run's maximum attainable recall is below R\*, its false detections are reported at its maximum recall and the run is flagged; flagged runs are listed in the paper.
- **Full-val operating point (footage):** the same construction on all of BDD val, giving each arm's footage threshold.
- **False-detection types** for each unmatched detection, in order: *duplicate* (its same-class ground truth is already matched by a higher-scoring detection), *wrong class* (IoU ≥ 0.5 with a ground truth of another class), *poor localization* (0.1 ≤ IoU < 0.5 with a same-class ground truth), *invented* (IoU < 0.1 with every ground truth).
- Precision-recall curves per arm and subset are reported.

### 7.4 Reporting

Absolute counts beside percentages; every subset reported, including ones where shape bias does not help; regressions shown, not only gains.

## 8. Software architecture

Python, managed with `uv`; package `snz` under `src/`; `pytest`, `ruff`. Runs on the GX10 (all real work) and on any CPU machine (tests with tiny synthetic fixtures).

```text
src/snz/
  data/bdd.py         BDD label loading, attribute filtering, split manifests, overlap checks
  data/footage.py     fixed-interval frame sampling, drop log, footage manifest
  stylize/            AdaIN wrapper, style-pool split, precomputed stylized copies, T-texture
  corrupt.py          T-corrupt via imagecorruptions
  train/model.py      Faster R-CNN R50-FPN with backbone ∈ {in1k_v1, sin}
  train/dataset.py    BDD detection dataset, stylized-copy mixing
  train/train.py      training loop from configs/recipe.toml, bf16, checkpoints, logs
  eval/predict.py     run a checkpoint on a split → detections file
  eval/coco_eval.py   pycocotools mAP
  eval/matching.py    per-image matches; vectorized AP for bootstrap
  eval/operating.py   equal-recall thresholds, false-detection types
  eval/bootstrap.py   paired bootstrap, Holm correction
  eval/report.py      tables and figures
  audit/sample.py     blind sampling of footage detections, separate key
  audit/app/          static HTML audit page → CSV
  audit/score.py      precision (Wilson CI), estimated counts, recall on labeled frames
  cli.py
configs/   recipe.toml, arms.toml, splits.toml
manifests/ split manifests (image names + hashes)
prereg/    prereg.md
results/   metric summaries, bootstrap outputs, figures (small files only)
tests/
```

Unit boundaries: `data` and `stylize` produce manifests and image files; `train` consumes manifests and configs and produces checkpoints; `eval` consumes checkpoints, manifests, and labels and produces result summaries; `audit` consumes footage detections and audit CSVs. Each is testable alone.

**What goes in git:** code, configs, manifests, preregistration, result summaries, figures. **What stays on the GX10:** images, BDD labels, stylized copies, checkpoints, raw detections, footage, audit answer key until unblinding. Result summaries are small enough to commit; large raw outputs, if needed for release, go to a user-uploaded GitHub Release.

## 9. Testing

Run on CPU here and in GitHub Actions (`pytest`, `ruff`):

1. BDD attribute filtering correct on a synthetic label file; splits deterministic; no image in two splits.
2. Style pool split disjoint; stylization (with a stub style network) preserves image size; boxes unchanged.
3. Stylized-copy mixing hits the configured probabilities and is deterministic per seed.
4. Both backbone variants build the same architecture with the same trainable parameters.
5. A 2-iteration training run on 4 tiny synthetic images completes on CPU.
6. mAP on hand-built detections matches hand-computed values; vectorized AP matches `pycocotools`.
7. Equal-recall threshold and false-detection types correct on constructed cases.
8. Bootstrap reproducible with a fixed seed and recovers a planted difference; Holm correction correct.
9. Audit sheet contains no arm identity; scoring and Wilson CIs correct.
10. Footage sampling rule produces the expected frame indices and logs drops.

## 10. Roles

- **Claude (this repository; container has no GPU and cannot download datasets or weights):** code, tests, configs, preregistration draft, analysis code, figures, paper drafts, review of committed results.
- **User (GX10):** environment smoke test; downloads (BDD100K, Stylized-ImageNet ResNet-50, AdaIN weights, style images); running data build, stylization, pilot, 12 training runs, and evaluation; committing result summaries; recording footage; labeling 100 frames; auditing 600 detections.

## 11. Gates

| Gate | Condition |
|---|---|
| G0 | This spec approved |
| G1 | GX10 smoke test: both backbones load; 50-iteration training run completes |
| G2 | Licenses checked; data built; manifests committed; stylized sets built |
| G3 | Pilot complete; recipe frozen; training time measured |
| G4 | Preregistration committed and tagged `prereg-v1` (includes footage sampling rule and audit protocol) |
| G5 | 12 training runs complete |
| G6 | Evaluation on all BDD test splits; analysis; figures |
| G7 | Footage collected, recall labels done, audit done, unblinded, analyzed |
| G8 | Paper draft |

No test split is evaluated before G4. Footage may be recorded at any time, but no model runs on it before G4.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Arm64 + GB10 toolchain issues | G1 smoke test first; NVIDIA PyTorch containers |
| Training slower than estimated | Pilot measures throughput; reduce schedule/input scale equally for all arms before freezing |
| Shape-biased backbone hurts clean accuracy substantially | Reported as RQ4; it is a finding, not a failure |
| Fine-tuning washes out backbone shape bias | Measured via H1b/H3; reported either way |
| T-texture favors stylized arms | T-shift is primary; confound stated |
| Small shift subsets (fog) | Reported with CIs only |
| BDD or style-image license restrictions | Checked at G2; nothing redistributed |
| Footage privacy | Raw data private; blur before publishing; no audio |
| Audit bias | Blinded, shuffled, key held separately; optional second auditor |
| Premise not supported (no texture reliance or no gains) | Null results are reported; the paper becomes a negative-result study |

## 13. References

- Geirhos, R., et al. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. ICLR. arXiv:1811.12231.
- Michaelis, C., et al. (2019). Benchmarking robustness in object detection: Autonomous driving when winter is coming. arXiv:1907.07484.
- Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization. ICCV. arXiv:1703.06868.
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. NeurIPS. arXiv:1506.01497.
- Lin, T.-Y., et al. (2017). Feature pyramid networks for object detection. CVPR. arXiv:1612.03144.
- Yu, F., et al. (2020). BDD100K: A diverse driving dataset for heterogeneous multitask learning. CVPR. arXiv:1805.04687.
- Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. ICLR. arXiv:1903.12261.
- Cimpoi, M., et al. (2014). Describing textures in the wild. CVPR.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2).
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158).
