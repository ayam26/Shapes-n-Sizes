# Shapes-n-Sizes

**Do road-object detectors rely on texture instead of shape — and does training them to be shape-biased make them find more real objects and invent fewer fake ones?**

Vision models used for driving sometimes "hallucinate": they miss real objects or report objects that are not there. One proposed cause is **texture bias** — ImageNet-trained networks classify mostly by surface texture rather than object shape ([Geirhos et al., 2019](https://arxiv.org/abs/1811.12231)). This project trains a road-object detector on our **ASUS Ascent GX10** to be shape-biased and tests, on real dashcam images and our own unfiltered dashcam footage, whether that helps.

- **Status (2026-10-11):** G1–G4 done — GX10 smoke test ([`results/g1_smoke.json`](results/g1_smoke.json)); data built from BDD100K images + **2018 labels** (official servers offline; see [`prereg/prereg.md`](prereg/prereg.md)); recipe frozen by the pilot (lr 0.02, 18k iterations; [`results/pilot/`](results/pilot/)); preregistration frozen at tag `prereg-v1`. **G5 running:** 12 training runs (~2 GX10-days). Next: G6 evaluation (`scripts/run_g6.sh`), G7 your dashcam footage, G8 paper ([`paper/paper.md`](paper/paper.md)). Runbook: [`docs/runbook.md`](docs/runbook.md). Handoff: [`docs/handoff.md`](docs/handoff.md). Plans: [`docs/superpowers/plans/`](docs/superpowers/plans/).
- **Focus shift to GNNs (2026-10-11, deadline 13:00 PDT):** CNN training stopped after BASE seed 0 (STY-s0 also finished and is kept as a baseline). BASE-s0 is stage 1 of a **texture-blind GNN shape verifier** (SAM 2 outlines → shape graph + size cues → GNN re-scores each detection). Status: outlines and second-stage training done (fold-B AUROC: BASE 0.908, GNN 0.910, RGB 0.918); preregistration frozen at tag `prereg-v2`; test re-scoring and analysis running → [`results/gnn/report.md`](results/gnn/report.md). Design: [`docs/superpowers/specs/2026-10-11-gnn-shape-verifier-design.md`](docs/superpowers/specs/2026-10-11-gnn-shape-verifier-design.md) · Plan: [`docs/superpowers/plans/2026-10-11-gnn-shape-verifier.md`](docs/superpowers/plans/2026-10-11-gnn-shape-verifier.md) · Preregistration: [`prereg/prereg-v2.md`](prereg/prereg-v2.md) · Commands: [`docs/runbook.md`](docs/runbook.md).
- **Deliverable:** an arXiv / workshop-style paper with code, preregistration, split manifests, and result summaries.
- **Honesty rules:** every result is reported, including nulls and regressions; analysis is frozen before test data is evaluated; no claim of improved vehicle safety.

---

## Contents

1. [The plan in one picture](#1-the-plan-in-one-picture)
2. [Research questions](#2-research-questions)
3. [How we train](#3-how-we-train)
4. [How we test](#4-how-we-test)
5. [Data and test material](#5-data-and-test-material)
6. [Hardware and models on the GX10](#6-hardware-and-models-on-the-gx10)
7. [Roadmap and who does what](#7-roadmap-and-who-does-what)
8. [Repository layout](#8-repository-layout)
9. [Licenses, privacy, provenance](#9-licenses-privacy-provenance)
10. [References](#10-references)

---

## 1. The plan in one picture

```text
BDD100K (clear daytime only) ──► train 4 detector variants on the GX10 ──► test on:
                                   (2 backbones × 2 kinds of training images)   • clear daytime          (cost)
                                   3 seeds each = 12 runs                       • night/rain/snow/fog    (main test)
                                                                                • texture-scrambled      (texture reliance)
                                                                                • nuImages / others      (other cities & cameras)
                                                                                • our unfiltered dashcam (real world, blind audit)
```

The **"before"** is the same detector trained the normal way. The **"after"** variants are trained to rely on shape.

## 2. Research questions

| | Question | How it is answered |
|---|---|---|
| **RQ1** | Are standard dashcam detectors texture-biased? | Accuracy on texture-scrambled copies of the test images vs. the originals ("retention"). Do the shape-biased variants retain more? |
| **RQ2** *(main)* | Does shape-biased training help when appearance changes but shapes don't? | Train on clear daytime only; test on night, rain, snow, fog. |
| **RQ3** | Does it hallucinate less? | False detections per image **at equal recall**, split into *invented objects*, *wrong class*, *poor box*, *duplicate*. |
| **RQ4** | What does it cost on normal daytime images? | Clean daytime accuracy difference. |
| **RQ5** | Does it hold on real, unfiltered footage — more *real* objects, fewer *fake* ones? | Blind audit of our own dashcam footage (§4). |

Preregistered hypotheses (frozen in `prereg/` before any test evaluation): shape-biased variants retain more accuracy on texture-scrambled images (H1b), score higher on night/rain/snow/fog (H2, H3), and produce fewer false detections at equal recall (H4). Holm-corrected, paired bootstrap over test images, 3 seeds per variant.

## 3. How we train

**Detector:** torchvision Faster R-CNN with a ResNet-50 + FPN backbone (BSD license; the shape-biased weights are a standard torchvision ResNet-50).

**Four variants (2 × 2):**

| | Normal training images | Half the images texture-scrambled |
|---|---|---|
| **Standard ImageNet backbone** | **BASE — the "before"** | STY |
| **Shape-biased backbone** (trained on Stylized-ImageNet) | SIN | BOTH |

- **Texture scrambling** = AdaIN style transfer: each training image's textures are replaced with a random painting/texture style while object outlines stay intact, so the model can't rely on texture. In STY/BOTH each image is shown original 50% of the time and stylized 50% ([Michaelis et al., 2019](https://arxiv.org/abs/1907.07484) found this mix keeps clean accuracy while adding robustness).
- **Fair comparison:** identical recipe for all variants (image size, augmentation, schedule, learning rate, steps); final checkpoint evaluated (no picking); no COCO-pretrained detector weights (they'd contaminate the shape-biased variants); 3 seeds each → 12 runs.
- **Pilot first:** a short BASE-only run sets the learning rate/schedule and measures speed. Estimated 4–5 GPU-hours per run (≈ 2–3 GX10-days total) until measured.

## 4. How we test

**Labeled real images (BDD100K val, 10k images):**

| Test set | What | Answers |
|---|---|---|
| T-clean | daytime + clear | RQ4 |
| T-shift | night / rain / snow / fog | RQ2, RQ3 (main) |
| T-texture | T-clean with textures scrambled using **styles never seen in training** | RQ1 |
| T-corrupt | T-clean with standard corruptions (blur, noise, weather) | exploratory |
| Other datasets | nuImages etc. — different cities and cameras | generalization (planned addition) |

**Our unfiltered dashcam footage — "did it find more objects, and are they real?"**

1. Record drives (video only, no audio). Take one frame every N seconds from every drive — no hand-picking.
2. Run all four variants; count detections per frame and per class.
3. **Blind precision audit:** 150 random detections per variant (600 total), shuffled, variant hidden. Mark each: *real object, right class* / *real, wrong class* / *nothing there (invented)* / *unsure*.
4. **Recall check:** hand-label every object in 100 random frames (no model help) → missed-object rate.
5. **Headline table per variant vs. BASE:** total detections, **estimated real detections**, **estimated invented detections**, recall.

"More objects" only counts as better if they are real — that is what steps 3–4 establish.

## 5. Data and test material

**Nothing listed here is stored in this repository.** Most datasets are non-commercial research licenses and many forbid redistribution. Register with your own account where required and download **directly onto the GX10**. Record what you downloaded (version, checksum) under `manifests/`.

Compiled 2026-10-11. **Verified:** ✅ checked (link/file tested or official docs read) · 🔶 secondary sources, confirm on the official page · ❓ license not found, check before use.

### 5.1 What our study uses (start here)

| Item | Where | Size | License | Verified |
|---|---|---|---|---|
| **BDD100K** — "100K Images" + "Detection 2020 Labels" (10 classes; weather & time-of-day tags) | [Download docs](https://doc.bdd100k.com/download.html), hosted at `https://dl.cv.ethz.ch/bdd100k/data/`; older portal `bdd-data.berkeley.edu` | images 5.3 GB (md5 `5a0359c86a0b8713adab1eee9a3041cb`); det_20 labels 53 MB (md5 `b86a3e1b7edbcad421b7dad2b3987c94`) | Education, research, not-for-profit | 🔶 md5s/host from official docs; re-read terms at download |
| **Shape-biased backbone** — ResNet-50 trained on Stylized-ImageNet (`resnet50_trained_on_SIN`) | URL in [`rgeirhos/texture-vs-shape`](https://github.com/rgeirhos/texture-vs-shape) → `models/load_pretrained_models.py` (Bitbucket) | 205 MB | See repo | ✅ downloaded; sha256 below |
| **Standard backbone** — torchvision `ResNet50_Weights.IMAGENET1K_V1` | Downloaded automatically by torchvision (`resnet50-0676ba61.pth`) | 98 MB | BSD-3 | ✅ hash prefix checked by torchvision |
| **AdaIN style-transfer weights** — `decoder.pth`, `vgg_normalised.pth` (British spelling) | [`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) release `v0.0.0` | 14 MB + 80 MB | MIT (repo archived, still downloadable) | ✅ downloaded; sha256 below |
| **Style images** — Describable Textures Dataset (5,640 images) | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/) (`dtd-r1.0.1.tar.gz`) | ~600 MB | "For research purposes" | 🔶 |
| **Corruptions** | `pip install imagecorruptions` ([Michaelis et al.](https://github.com/bethgelab/robust-detection-benchmark)) | — | Apache-2.0 (package) | 🔶 |

Checksums of files downloaded and verified this session (compare on the GX10 with `sha256sum`):

```text
c8e5653e47bbf3f6efad5da531f3acea9963acb93362b6530cf39b2ba4750da3  resnet50_train_60_epochs-c8e5653e.pth.tar
379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472  decoder.pth
804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a  vgg_normalised.pth
```

Shape-biased weights URL: `https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar`

### 5.2 Driving datasets with bounding boxes

| Dataset | What | License | Verified | Use for us |
|---|---|---|---|---|
| [BDD100K](https://doc.bdd100k.com/download.html) | 100k frames from US dashcam videos; 10 classes; weather/time-of-day/scene tags | Research / non-profit | 🔶 | **Core** training + labeled test |
| [nuImages](https://www.nuscenes.org/nuimages) | 93k images, 2D boxes and masks for ~800k objects; rain, snow, night (Boston, Singapore) | CC BY-NC-SA 4.0; account required | 🔶 | **Best second test set**; no traffic lights/signs, so evaluate shared classes |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) | 7.5k training images, 2D/3D boxes (Karlsruhe) | CC BY-NC-SA 3.0 | 🔶 | Small classic cross-dataset test |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 5k finely annotated urban images (masks → boxes); Foggy Cityscapes derivative | Custom non-commercial; registration | 🔶 | Cross-dataset test; synthetic fog |
| [Waymo Open Dataset](https://waymo.com/open/) | Large multi-camera dataset with 2D camera boxes | [Non-commercial license](https://waymo.com/open/terms/) (Mar 2025); registration; redistribution only to registered users | ✅ terms page | Large-scale test if needed |
| [Argoverse 2](https://www.argoverse.org/) | Multi-city sensor data, 3D annotations | CC BY-NC-SA 4.0 | 🔶 | Optional |
| [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) | 25k street-level images worldwide, many cameras | Research license (CC BY-NC-SA) | 🔶 | Diverse cameras/countries |
| [D²-City](https://paperswithcode.com/paper/d2-city-a-large-scale-dashcam-video-dataset) | 10k+ dashcam clips (China); boxes + tracks on 1k videos, keyframe boxes on the rest | ❓ | ❓ | Dashcam video with labels |
| [SHIFT](https://www.vis.xyz/shift) (synthetic) | 2.5M simulator frames; continuous changes in rain, fog, cloud, time of day | ❓ | 🔶 | Controlled shift curves |

### 5.3 Bad weather and night

| Dataset | What | License | Verified |
|---|---|---|---|
| BDD100K subsets | Night / rainy / snowy / foggy via attributes | Research / non-profit | 🔶 |
| [ACDC](https://arxiv.org/abs/2104.13395) | 4,006 images across fog, night, rain, snow, each with a clear-weather counterpart; panoptic labels (boxes derivable) | ❓ (site `acdc.vision.ee.ethz.ch`) | ❓ |
| [DAWN](https://arxiv.org/abs/2008.05402) | ~1,000 real images in fog, snow, rain, sandstorm; vehicle/person/cyclist boxes | CC BY 4.0 per arXiv; web-sourced images, per-image rights vary | 🔶 |
| nuImages | Rain / snow / night samples | CC BY-NC-SA 4.0 | 🔶 |
| [Cityscapes-C / COCO-C](https://github.com/bethgelab/robust-detection-benchmark) | Corrupted versions of standard sets | See repo | ✅ repo exists |

### 5.4 Corner cases and anomalies (hallucination-prone scenes)

| Dataset | What | License | Verified |
|---|---|---|---|
| [CODA](https://coda-dataset.github.io/) | 1,500 real driving scenes, ~4 object-level corner cases each (30+ categories); standard detectors reach ≤ 12.8% mAR | ❓ | 🔶 |
| [SegmentMeIfYouCan](https://segmentmeifyoucan.com/) | RoadAnomaly21 (real, web-sourced) and RoadObstacle21 (recorded): unusual objects on the road | ❓ | 🔶 |
| Lost and Found, Fishyscapes | Small road obstacles (Lost and Found); synthetic anomalies (Fishyscapes) | ❓ | 🔶 |

These measure the opposite failure — *missing* unusual objects — and show whether shape-biased training trades one error for another.

### 5.5 Raw / unfiltered dashcam footage

| Source | What | License | Verified | Caveat |
|---|---|---|---|---|
| **Our own dashcam** | The study's unfiltered test | Ours | — | Video only, no audio; blur faces/plates before publishing |
| BDD100K videos | 40-second source videos behind the labeled frames | Research / non-profit | 🔶 | Same source as training data — weaker test |
| [Nexar dashcam dataset](https://arxiv.org/abs/2503.03848) | 1,500 ~40 s clips: collisions, near-collisions, normal driving; anonymized; on Hugging Face | Conflicting: MIT (press) vs. "open, restrictions on unethical use" (paper) | 🔶 | Event-heavy, not unfiltered |
| [DoTA](https://arxiv.org/abs/2004.03044) | 4,677 dashcam accident videos (YouTube-sourced) with anomaly windows and tracks | ❓ | 🔶 | Accident-heavy |
| DADA-2000 | 2,000 accident clips with driver-attention maps | Academic use (per survey) | 🔶 | Accident-heavy |

Accident datasets are useful stress tests but over-represent rare events.

### 5.6 Texture-vs-shape probes

| Material | What | License | Verified |
|---|---|---|---|
| Geirhos cue-conflict stimuli | Shape of one class + texture of another; the standard shape-bias test. Also in TensorFlow Datasets (`geirhos_conflict_stimuli`) | CC BY 4.0 (per a third-party paper) | 🔶 |
| [model-vs-human](https://github.com/bethgelab/model-vs-human) | 17 out-of-distribution datasets (cue-conflict, silhouettes, edges, stylized…) with human data | Check repo | 🔶 |
| [Stylized-ImageNet code](https://github.com/rgeirhos/Stylized-ImageNet) | Recipe for stylized images with AdaIN | See repo | 🔶 |
| `perception-lab/` renders | Our matched real-vs-picture truck scenes | Self-created | ✅ |

### 5.7 Testing the GX10's vision-language models (Nemotron etc.)

| Benchmark | What it tests | License | Verified |
|---|---|---|---|
| [POPE](https://github.com/RUCAIBox/POPE) | Object hallucination: "Is there a ⟨object⟩?" yes/no on COCO, with random/popular/adversarial negatives | MIT (secondary) | 🔶 |
| HallusionBench | Visual illusions and language-prior hallucinations | ❓ | ❓ |
| AMBER | Existence/attribute/relation hallucinations | ❓ | ❓ |
| [DriveBench](https://drive-bench.github.io/) (ICCV 2025) | Driving VLM reliability: 19,200 frames, 20,498 QA pairs, clean vs. corrupted vs. text-only; VLMs often answer from text priors. Data: Hugging Face `drive-bench/arena` | ❓ | 🔶 |
| [CODA-LM](https://coda-dataset.github.io/coda-lm/) | Corner-case driving scenes for VLMs: 9,768 scenarios | ❓ | 🔶 |
| [DriveLM](https://github.com/OpenDriveLab/DriveLM) (ECCV 2024) | Driving graph-VQA on nuScenes and CARLA | Language data CC BY-NC-SA 4.0; code Apache-2.0; nuScenes images under nuScenes terms | 🔶 |

Suggested first VLM tests: (1) POPE subset as a general hallucination baseline; (2) **Driving-POPE** — POPE-style yes/no questions we generate from BDD100K/nuImages labels ("Is there a truck?"), giving hallucination rates on real dashcam frames; (3) DriveBench clean vs. corrupted; (4) CODA-LM; (5) cue-conflict shape bias, linking the VLM side to the detector study.

## 6. Hardware and models on the GX10

From [`perception-lab/artifacts/environment.md`](perception-lab/artifacts/environment.md): Ubuntu 24.04 aarch64, 20 Arm cores, NVIDIA GB10 (CUDA 13.0), 121 GiB unified memory, 916 GB disk (≈613 GB free). Ollama models:

| Model | Size / quant | Vision |
|---|---|---|
| `nemotron3:33b` | 33B, Q4_K_M | ✅ |
| `qwen3.6:35b` | 36B, Q4_K_M | ✅ |
| `gemma4:26b` | 26B, Q4_K_M | ✅ |
| `nemotron-3-super` | 124B, Q4_K_M | ❌ text only |

Pilot evidence from the hackathon build in `perception-lab/`: on its held-out set, `nemotron3:33b` labeled all 6 real trucks as flat pictures (6/18 correct), `qwen3.6` scored 13/18, `gemma4` 17/18 ([report](perception-lab/artifacts/report.md)).

Detector training uses PyTorch + torchvision on the GB10 (first step: a smoke test; NVIDIA PyTorch containers as fallback).

## 7. Roadmap and who does what

| Gate | Milestone | Who |
|---|---|---|
| G0 | Design spec approved | team |
| G1 | GX10 smoke test: both backbones load; 50-iteration training run | GX10 operator |
| G2 | Licenses checked; BDD100K, weights, styles downloaded; manifests committed; stylized sets built | GX10 operator, code by Claude |
| G3 | Pilot run; training recipe frozen; speed measured | GX10 operator |
| G4 | Preregistration committed and tagged `prereg-v1` (incl. footage sampling rule and audit protocol) | team, drafted by Claude |
| G5 | 12 training runs | GX10 operator |
| G6 | Evaluation on all test sets; analysis; figures | GX10 operator runs, Claude analyzes |
| G7 | Dashcam footage recorded; 100 frames labeled; 600-detection blind audit; unblinded | team |
| G8 | Paper draft | team + Claude |

No test set is evaluated before G4. Footage can be recorded any time, but no model runs on it before G4.

**Claude (cloud session)** writes the code, tests, configs, preregistration draft, analysis, and paper drafts. Its container has no GPU and cannot reach Hugging Face or dataset hosts, so **all real runs happen on the GX10**.

## 8. Repository layout

```text
README.md                    this file — the full plan
docs/superpowers/specs/      design spec (2026-10-11-shape-biased-detector-design.md)
docs/superpowers/plans/      implementation plans
docs/runbook.md              GX10 commands for gates G1–G3
src/snz/                     data (splits, manifests), stylize (AdaIN), train, eval; `snz` CLI
configs/                     splits, stylize, recipe (frozen at G3), arms
scripts/                     GX10 setup and download scripts
manifests/                   split manifests (image names + SHA-256), written at G2
results/                     small result summaries (g1_smoke.json, pilot metrics, ...)
tests/                       pytest suite (CPU, synthetic fixtures)
perception-lab/              hackathon VLM testbed and results (pilot evidence)
# planned (Plans 2–3): src/snz/eval analysis (bootstrap, operating points), src/snz/audit, prereg/
```

## 9. Licenses, privacy, provenance

- Most datasets are non-commercial research licenses: fine for a paper, not a product. Never commit images, labels, or derived image sets.
- Read any ❓ license on its official page before use and update this README.
- Save `sha256sum` output and dataset versions to `manifests/` after every download.
- Our own footage stays private; blur faces and license plates in anything published; no audio (California requires all-party consent for recorded conversations).

## 10. References

- Geirhos et al. (2019). ImageNet-trained CNNs are biased towards texture. ICLR. [arXiv:1811.12231](https://arxiv.org/abs/1811.12231) · [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape)
- Michaelis et al. (2019). Benchmarking robustness in object detection: Autonomous driving when winter is coming. [arXiv:1907.07484](https://arxiv.org/abs/1907.07484) · [robust-detection-benchmark](https://github.com/bethgelab/robust-detection-benchmark)
- Huang & Belongie (2017). AdaIN style transfer. ICCV. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868) · [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
- Ren et al. (2015). Faster R-CNN. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497) · Lin et al. (2017). FPN. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- Yu et al. (2020). BDD100K. CVPR. [arXiv:1805.04687](https://arxiv.org/abs/1805.04687) · [download docs](https://doc.bdd100k.com/download.html)
- Li et al. (2023). POPE: Evaluating object hallucination in large vision-language models. [arXiv:2305.10355](https://arxiv.org/abs/2305.10355)
- Xie et al. (2025). Are VLMs ready for autonomous driving? (DriveBench). ICCV. [arXiv:2501.04003](https://arxiv.org/abs/2501.04003)
- CODA [arXiv:2203.07724](https://arxiv.org/abs/2203.07724) · CODA-LM [arXiv:2404.10595](https://arxiv.org/abs/2404.10595) · ACDC [arXiv:2104.13395](https://arxiv.org/abs/2104.13395) · DAWN [arXiv:2008.05402](https://arxiv.org/abs/2008.05402) · SHIFT [arXiv:2206.08367](https://arxiv.org/abs/2206.08367) · Nexar [arXiv:2503.03848](https://arxiv.org/abs/2503.03848)
