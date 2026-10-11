# Datasets and Test Material

A teammate's guide to open data for this project: dashcam and driving datasets for training and testing object detectors, weather/night shift sets, corner-case and anomaly sets, raw dashcam footage, texture-vs-shape probes, and hallucination benchmarks for the vision-language models (Nemotron etc.) installed on the GX10.

**Nothing listed here is stored in this repository.** Most of these datasets are licensed for non-commercial research only and many forbid redistribution. Each person registers with their own account where required and downloads directly onto the GX10. Record what you downloaded (version, checksum) under `manifests/`.

Compiled 2026-10-11. **Verification column:** ✅ checked this session (link or file tested, or official docs read) · 🔶 from secondary sources, confirm on the official page · ❓ license not found, check before use.

## 1. What our study uses (start here)

The design is in [`docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md`](docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md).

| Item | Where | Size | License | Verified |
|---|---|---|---|---|
| **BDD100K** — "100K Images" + "Detection 2020 Labels" (10 classes; weather & time-of-day tags) | [Download docs](https://doc.bdd100k.com/download.html), hosted at `https://dl.cv.ethz.ch/bdd100k/data/`; older portal `bdd-data.berkeley.edu` | images 5.3 GB (md5 `5a0359c86a0b8713adab1eee9a3041cb`); det_20 labels 53 MB (md5 `b86a3e1b7edbcad421b7dad2b3987c94`) | Education, research, not-for-profit | 🔶 md5s and host from the official docs; terms to re-read at download |
| **Shape-biased backbone** — ResNet-50 trained on Stylized-ImageNet (`resnet50_trained_on_SIN`) | URL in [`rgeirhos/texture-vs-shape`](https://github.com/rgeirhos/texture-vs-shape) → `models/load_pretrained_models.py` (Bitbucket) | 205 MB | See repo | ✅ downloaded; sha256 below |
| **Standard backbone** — torchvision `ResNet50_Weights.IMAGENET1K_V1` | Downloaded automatically by torchvision (`resnet50-0676ba61.pth`) | 98 MB | BSD-3 | ✅ known file; hash prefix `0676ba61` checked by torchvision |
| **AdaIN style-transfer weights** — `decoder.pth`, `vgg_normalised.pth` (British spelling) | [`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) release `v0.0.0` | 14 MB + 80 MB | MIT (repo archived, still downloadable) | ✅ downloaded; sha256 below |
| **Style images** — Describable Textures Dataset (5,640 images) | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/) (`dtd-r1.0.1.tar.gz`) | ~600 MB | "For research purposes" | 🔶 |
| **Corruptions** for robustness tests | `pip install imagecorruptions` (from [Michaelis et al.](https://github.com/bethgelab/robust-detection-benchmark)) | — | Apache-2.0 (package) | 🔶 |

**Checksums of files downloaded this session** (compare after downloading on the GX10 with `sha256sum`):

```text
c8e5653e47bbf3f6efad5da531f3acea9963acb93362b6530cf39b2ba4750da3  resnet50_train_60_epochs-c8e5653e.pth.tar
379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472  decoder.pth
804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a  vgg_normalised.pth
```

Shape-biased weights URL (from the repo's `load_pretrained_models.py`):
`https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar`

## 2. Driving datasets with bounding boxes (train / test detectors)

| Dataset | What | License | Verified | Use for us |
|---|---|---|---|---|
| [BDD100K](https://doc.bdd100k.com/download.html) | 100k frames from US dashcam videos; 10 classes; weather/time-of-day/scene tags | Research / non-profit | 🔶 | **Core** training + labeled test |
| [nuImages](https://www.nuscenes.org/nuimages) | 93k images, 2D boxes and masks for ~800k objects; includes rain, snow, night (Boston, Singapore) | CC BY-NC-SA 4.0; account required | 🔶 | **Best second test set** (different cities and cameras); no traffic lights/signs, so evaluate shared classes |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) | 7.5k training images with 2D/3D boxes (Karlsruhe) | CC BY-NC-SA 3.0 | 🔶 | Small classic cross-dataset test |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 5k finely annotated urban images (instance masks → boxes); Foggy Cityscapes derivative | Custom, non-commercial; registration | 🔶 | Cross-dataset test; synthetic fog |
| [Waymo Open Dataset](https://waymo.com/open/) | Large multi-camera dataset with 2D camera boxes | [Non-commercial license](https://waymo.com/open/terms/) (March 2025); registration; redistribution only to registered users | ✅ terms page | Large-scale test if needed |
| [Argoverse 2](https://www.argoverse.org/) | Multi-city sensor data with 3D annotations | CC BY-NC-SA 4.0 | 🔶 | Optional |
| [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) | 25k street-level images worldwide, many cameras | Research license (CC BY-NC-SA) | 🔶 | Diverse cameras/countries |
| [D²-City](https://paperswithcode.com/paper/d2-city-a-large-scale-dashcam-video-dataset) | 10k+ dashcam clips from China; boxes + tracks on 1k videos, keyframe boxes on the rest | ❓ | ❓ | Dashcam video with labels |
| [SHIFT](https://www.vis.xyz/shift) (synthetic) | 2.5M frames from a simulator; continuous changes in rain, fog, cloud, time of day | ❓ | 🔶 | Controlled shift curves (synthetic) |

## 3. Bad weather and night (appearance-shift tests)

| Dataset | What | License | Verified |
|---|---|---|---|
| BDD100K subsets | Night / rainy / snowy / foggy splits via attributes | Research / non-profit | 🔶 |
| [ACDC](https://arxiv.org/abs/2104.13395) | 4,006 images evenly across fog, night, rain, snow, each with a clear-weather counterpart; pixel-level panoptic labels (boxes derivable) | ❓ (project site `acdc.vision.ee.ethz.ch`) | ❓ |
| [DAWN](https://arxiv.org/abs/2008.05402) | ~1,000 real images in fog, snow, rain, sandstorm; boxes for vehicles, people, cyclists | CC BY 4.0 per arXiv metadata; images web-sourced, so per-image rights vary | 🔶 |
| nuImages | Rain / snow / night samples | CC BY-NC-SA 4.0 | 🔶 |
| [Cityscapes-C / COCO-C](https://github.com/bethgelab/robust-detection-benchmark) | Corrupted versions (noise, blur, weather, digital) of standard sets | See repo | ✅ repo exists |

## 4. Corner cases, anomalies, hallucination-prone scenes (detectors)

| Dataset | What | License | Verified |
|---|---|---|---|
| [CODA](https://coda-dataset.github.io/) | 1,500 real driving scenes with ~4 object-level corner cases each (30+ categories); standard detectors reach ≤ 12.8% mAR | ❓ | 🔶 |
| [SegmentMeIfYouCan](https://segmentmeifyoucan.com/) | RoadAnomaly21 (real, web-sourced) and RoadObstacle21 (recorded) — unusual objects on the road | ❓ | 🔶 |
| Lost and Found, Fishyscapes | Small obstacles on the road (Lost and Found); synthetic anomalies (Fishyscapes) | ❓ | 🔶 |

These measure the opposite failure to texture-driven false positives — missing unusual objects — and are useful for showing whether shape-biased training trades one error for another.

## 5. Raw / unfiltered dashcam footage

| Source | What | License | Verified | Caveat |
|---|---|---|---|---|
| **Our own dashcam** | The study's unfiltered test (sampling rule in the spec) | Ours | — | Video only, no audio; blur faces/plates before publishing anything |
| BDD100K videos | The 40-second source videos behind the labeled frames | Research / non-profit | 🔶 | Same source as training data — weaker test than our own footage |
| [Nexar dashcam dataset](https://arxiv.org/abs/2503.03848) | 1,500 ~40 s clips: collisions, near-collisions, normal driving; anonymized; on Hugging Face | Conflicting: MIT (press release) vs. "open license with restrictions on unethical use" (paper) | 🔶 | Event-heavy, not unfiltered |
| [DoTA](https://arxiv.org/abs/2004.03044) | 4,677 dashcam accident videos (from YouTube) with anomaly windows and tracks | ❓ | 🔶 | YouTube-sourced; accident-heavy |
| DADA-2000 | 2,000 accident clips with driver-attention maps | Academic use (per survey table) | 🔶 | Accident-heavy |

Accident datasets are useful for stress tests but are not "unfiltered": they over-represent rare events.

## 6. Texture-vs-shape probes

| Material | What | License | Verified |
|---|---|---|---|
| Geirhos cue-conflict stimuli | Images with the shape of one class and the texture of another; the standard shape-bias test. Also in TensorFlow Datasets (`geirhos_conflict_stimuli`) | CC BY 4.0 (per a third-party paper) | 🔶 |
| [model-vs-human](https://github.com/bethgelab/model-vs-human) | Toolbox with 17 out-of-distribution datasets (cue-conflict, silhouettes, edges, stylized, …) plus human data | Check repo | 🔶 |
| [Stylized-ImageNet code](https://github.com/rgeirhos/Stylized-ImageNet) | Recipe for making stylized images with AdaIN | See repo | 🔶 |
| `perception-lab/` renders | Our own matched real-vs-picture truck scenes | Self-created | ✅ |

## 7. Testing the GX10's vision-language models (Nemotron etc.)

**Installed on the GX10** (Ollama, from [`perception-lab/artifacts/environment.md`](perception-lab/artifacts/environment.md)):

| Model | Size / quant | Vision |
|---|---|---|
| `nemotron3:33b` | 33B, Q4_K_M | ✅ |
| `qwen3.6:35b` | 36B, Q4_K_M | ✅ |
| `gemma4:26b` | 26B, Q4_K_M | ✅ |
| `nemotron-3-super` | 124B, Q4_K_M | ❌ text only — cannot see images |

Pilot result to keep in mind: on the Perception Lab held-out set, `nemotron3:33b` labeled all 6 real trucks as flat pictures (6/18 correct), `qwen3.6` scored 13/18, `gemma4` 17/18 ([report](perception-lab/artifacts/report.md)).

| Benchmark | What it tests | License | Verified |
|---|---|---|---|
| [POPE](https://github.com/RUCAIBox/POPE) | Object hallucination: "Is there a ⟨object⟩ in the image?" yes/no on COCO images, with random/popular/adversarial negatives | MIT (per secondary source) | 🔶 |
| HallusionBench | Visual illusions and language-prior hallucinations | ❓ | ❓ |
| AMBER | Hallucination benchmark (existence, attributes, relations) | ❓ | ❓ |
| [DriveBench](https://drive-bench.github.io/) (ICCV 2025) | Driving VLM reliability: 19,200 frames, 20,498 QA pairs, clean vs. corrupted vs. text-only inputs; finds VLMs often answer from text priors instead of the image. Data on Hugging Face (`drive-bench/arena`) | ❓ | 🔶 |
| [CODA-LM](https://coda-dataset.github.io/coda-lm/) | Corner-case driving scenes for VLMs: 9,768 scenarios, 21,537 corner-case annotations | ❓ | 🔶 |
| [DriveLM](https://github.com/OpenDriveLab/DriveLM) (ECCV 2024) | Driving graph-VQA on nuScenes and CARLA | Language data CC BY-NC-SA 4.0; code Apache-2.0; nuScenes images under nuScenes terms | 🔶 |
| Geirhos cue-conflict (above) | Shape bias of a VLM: ask "what object is this?" on shape/texture conflicts | CC BY 4.0 (secondary) | 🔶 |

**Suggested first tests for Nemotron and the other two VLMs**

1. **POPE subset** — general object-hallucination baseline, comparable to published numbers.
2. **Driving-POPE (we build it):** POPE-style yes/no questions generated from BDD100K / nuImages labels ("Is there a truck in this image?"), with negatives that are random, frequent, or co-occurring classes. Measures hallucination on real dashcam frames with ground truth from labels.
3. **DriveBench clean vs. corrupted** — does the model admit when the image is degraded, or answer from priors?
4. **CODA-LM** — unusual road objects.
5. **Cue-conflict** — is the VLM shape- or texture-biased? Directly connects the VLM side to the detector study.

## 8. Practical notes

- **Download on the GX10, not on laptops.** It has 916 GB disk (≈613 GB free at the Perception Lab start). Claude's cloud container cannot reach Hugging Face, ETH, or PyTorch download hosts; the GX10 can reach PyPI and should reach the rest.
- **Licenses:** most entries are non-commercial research licenses — fine for a research paper, not for a product. Don't commit images, labels, or derived image sets to this repo.
- **Provenance:** after each download, save `sha256sum` output and the dataset version to `manifests/` so results are reproducible.
- **Privacy:** our own footage stays private; blur faces and license plates in anything published; no audio.
- **Before using any ❓ entry**, read its license on the official page and note it here.

## Sources

- BDD100K: [download docs](https://doc.bdd100k.com/download.html) · [SysCV dataset preparation](https://github.com/SysCV/bdd100k-models/blob/main/doc/PREPARE_DATASET.md)
- Shape bias: [Geirhos et al. 2019](https://arxiv.org/abs/1811.12231) · [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) · [Michaelis et al. 2019](https://arxiv.org/abs/1907.07484) · [robust-detection-benchmark](https://github.com/bethgelab/robust-detection-benchmark)
- AdaIN: [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) · DTD: [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [TFDS](https://www.tensorflow.org/datasets/catalog/dtd)
- nuImages: [Dataset Ninja](https://datasetninja.com/nu-images) · [nuscenes-devkit](https://pypi.org/project/nuscenes-devkit/1.1.3) · Waymo: [terms](https://waymo.com/open/terms/)
- Weather: [ACDC](https://arxiv.org/abs/2104.13395) · [DAWN](https://arxiv.org/abs/2008.05402), [Mendeley](https://data.mendeley.com/datasets/766ygrbt8y/3) · [SHIFT](https://arxiv.org/abs/2206.08367)
- Corner cases: [CODA](https://arxiv.org/abs/2203.07724) · [CODA-LM](https://arxiv.org/abs/2404.10595) · [SegmentMeIfYouCan](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/d67d8ab4f4c10bf22aa353e27879133c-Paper-round2.pdf)
- Dashcam video: [Nexar](https://arxiv.org/abs/2503.03848), [Nexar blog](https://data.getnexar.com/blog/nexars-open-dataset-and-crash-prediction-challenge-pushing-the-boundaries-of-ai-driven-road-safety) · [D²-City](https://paperswithcode.com/paper/d2-city-a-large-scale-dashcam-video-dataset) · [accident dataset survey](https://arxiv.org/abs/2308.15985)
- VLM benchmarks: [POPE](https://github.com/RUCAIBox/POPE) · [DriveBench](https://arxiv.org/abs/2501.04003) · [DriveLM](https://github.com/OpenDriveLab/DriveLM)
- Shape probes: [model-vs-human](https://github.com/bethgelab/model-vs-human) · [TFDS cue-conflict stimuli](https://www.tensorflow.org/datasets/catalog/geirhos_conflict_stimuli)
