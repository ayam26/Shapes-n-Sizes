# Shapes-n-Sizes — Presentation, Demo and Q&A Guide

Working document for the final presentation. Sections marked **⟨fill⟩** are completed from
`results/gnn/report.md` (and `results/report.md` for the CNN baselines) when the GX10 runs finish.
Rule for everything below: say only what the results show; nulls and regressions get presented,
not hidden.

---

## 1. One-line pitch

> **Camera detectors in cars "hallucinate" because they lean on texture. We added a second
> opinion that is texture-blind by construction — a graph neural network that only sees an
> object's outline and size — and measured whether it removes fake detections and lets the
> detector find more real objects at night and in bad weather, all running locally on an ASUS
> GX10.**

---

## 2. The problem

### What goes wrong

Object detectors in driving systems sometimes report things that are not there (a shadow, a
reflection, a road marking, a patch of truck-colored texture becomes a "car") or miss things
that are (a pedestrian at night). These are often called **hallucinations**. In a detector we
measure them precisely as **false detections**: invented objects (nothing there), wrong class
(a truck called a car), poorly placed boxes, and duplicates.

### Why it happens: texture bias

Geirhos et al. (2019) showed that standard ImageNet-trained networks decide mostly by
**surface texture**, not **shape**: show one a cat-shaped image with elephant-skin texture and it
says "elephant"; people say "cat". Detectors in cars are built on the same kind of backbone.

Driving makes this worse, because the conditions that change texture the most —
**night, rain, snow, fog, glare** — are exactly where errors matter. Darkness and weather change
colors, contrast and surface detail a lot, but **an object's outline and size barely change**:
a car at night is still car-shaped and car-sized for where it sits in the road.

### Why existing fixes are incomplete

The known fix (Michaelis et al., 2019) is to train on style-transferred images so the network
learns to rely less on texture. That helps, but the network **still sees texture** and can
still be fooled by it; you are discouraging a habit, not removing the input.

### What we do not claim

This is a research prototype. We do not claim a safer car. We measure one failure mode under
controlled, preregistered conditions.

---

## 3. Our approach — what is unique

**Idea: separate "where is the object and what shape is it?" from "what is it?", and make the
"what is it?" step physically unable to use texture.**

```text
dashcam frame ─► Faster R-CNN detector proposes boxes (incl. unsure ones)
             ─► SAM 2 cuts each object's outline (class-agnostic: finds a boundary, does not name it)
             ─► 64 points along the outline become a graph  +  size cues (no colors anywhere)
             ─► graph neural network: "car / pedestrian / truck / … / not an object"
             ─► combine with the detector's confidence ─► keep or drop the detection
```

What makes it different:

1. **Texture-blind by construction.** The GNN's input is only geometry (point positions,
   turning angles, curvature) and size. Not a single pixel color reaches it. A unit test checks
   that recoloring an image leaves the GNN's input bit-identical.
2. **Shapes *and* sizes.** The size cues use road geometry: an object lower in the frame is
   closer and must look bigger. A "car" four pixels tall at the bottom of the frame is
   implausible — the GNN can know that without ever seeing the pixels. (Hence the name.)
3. **A plug-in second opinion.** It re-scores any detector's output; the detector is not
   retrained. The GNN is tiny (≈ 0.5 M parameters) and trains in minutes.
4. **"More real objects", measured properly.** Because the GNN removes fakes, the detector can
   run at a lower threshold and still have the same number of false alarms — so the fair
   question is *recall at equal false-alarm rate*, not raw detection counts.
5. **A control that sees texture.** We train a same-sized image-crop CNN through the identical
   pipeline. If the GNN beats it under night/weather, the gain comes from *ignoring* texture,
   not just from having a second classifier.
6. **Honest-by-design evaluation.** Hypotheses are frozen in a preregistration (git tag
   `prereg-v2`) before the GNN ever sees a test image, and the code refuses to score test data
   until the tag exists. Training uses only clear daytime images; night, rain, snow and fog are
   never seen in training.
7. **Fully local on the GX10.** Detector, SAM 2, GNN training and evaluation, and the
   vision-language models from our pilot all run on the box — no cloud, which matters for
   private dashcam data.

### The experiment

| Arm | What re-scores the detections |
|---|---|
| BASE | nothing (the detector alone) — the "before" |
| **GNN** | outline graph + size cues — our method |
| RGB | image-crop CNN (sees texture) — control |
| GNN-SHAPE | outline only, no size cues — ablation |
| GNN-COLOR | outline + size + colors at the outline — ablation |

All arms re-score the **same** detections, so differences come only from the second stage.

Hypotheses (night/rain/snow/fog, BDD100K validation images, 4,752 images):

- **G1** — GNN makes fewer false detections per image than BASE at the same recall.
- **G2** — GNN finds more real objects than BASE at the same false-alarm rate.
- **G3** — GNN makes fewer false detections than the texture-seeing RGB check.

Paired bootstrap over images, Holm correction across the three tests.

### Where it came from (the journey, 30 seconds)

1. **Perception Lab pilot (VLMs on the GX10):** local vision-language models confused real
   trucks with pictures of trucks — `nemotron3:33b` labeled all 6 real trucks in the held-out
   set as flat pictures (6/18 correct); `qwen3.6` 13/18; `gemma4` 17/18.
2. **CNN study:** a preregistered 2×2 of the established fixes (shape-biased backbone ×
   stylized training) on BDD100K — the BASE detector from it is our stage 1.
3. **GNN verifier:** instead of discouraging texture use, remove texture from the decision.

---

## 4. Results ⟨fill⟩

### Already in hand

- GX10 runs the full stack locally (GB10 GPU, CUDA 13.0, PyTorch 2.14; `results/g1_smoke.json`).
- Detector pilot on clear-daytime validation images (id-val): mAP 24.5 (lr 0.01) vs. **25.9**
  (lr 0.02), AP50 50.3 vs. 51.8 — recipe frozen at lr 0.02.
- Perception Lab VLM pilot (above).

### Headline table (from `results/gnn/report.md`)

| | BASE | GNN | RGB | GNN-SHAPE | GNN-COLOR |
|---|---|---|---|---|---|
| False detections / image on T-shift at equal recall (G1, G3) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| …of which **invented** objects / image | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| Recall on T-shift at equal false-alarm rate (G2) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| mAP T-shift / T-clean / T-texture | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| Outline success rate (SAM 2) night / clear | ⟨⟩ | | | | |

Confirmatory: G1 ⟨supported / not⟩ (Δ ⟨⟩, 95% CI ⟨⟩, Holm p ⟨⟩) · G2 ⟨…⟩ · G3 ⟨…⟩.

### How to present each possible outcome

| Outcome | What to say |
|---|---|
| G1 + G2 supported, G3 supported | "Removing texture from the decision cut false detections by ⟨x⟩% at the same recall and found ⟨y⟩% more real objects at the same false-alarm rate — and beat a texture-seeing check, so it's the shape-only input that helps." |
| G1/G2 supported, G3 not | "A second stage helps, but we can't yet say shape-only beats a texture-seeing one." |
| Nothing supported | "With one seed and ~2k training detections, a texture-blind verifier did not measurably help. Here is where it failed (outline success at night, per-class changes) — and the pipeline is ready to test with more data." |
| GNN hurts | Show it. Point to the false-detection types and per-class AP that changed, and the outline failure rate. |

Never present selected examples without saying they are selected and showing the full table.

---

## 5. Demo

### 5.1 Three-minute demo script

| Time | Show | Say |
|---|---|---|
| 0:00–0:25 | A night BDD image with BASE boxes, several fakes highlighted | "The detector sees a car in a shadow and a truck in a reflection. That's texture." |
| 0:25–1:00 | The same image: each box's outline graph next to it, GNN verdict bars | "Our GNN only ever sees these 64 dots and the box size — never the pixels." |
| 1:00–1:30 | **Recolor test, live**: hue-rotate / invert the image on the GX10; detector output changes, GNN input numbers identical | "Change every color: the detector's opinion moves, the GNN literally cannot notice." |
| 1:30–2:10 | Side-by-side BASE vs BASE+GNN at equal false-alarm rate; counts of real / invented | "Same false alarms, ⟨y⟩% more real objects." (or the honest null) |
| 2:10–2:40 | Results table + one failure case (a real object the GNN wrongly dropped; a billboard truck it can't catch) | "Where it fails, and why." |
| 2:40–3:00 | GX10 resource panel: detector + SAM 2 + GNN running locally, memory and ms per frame | "All of it on one desk-side box, no cloud." |

### 5.2 Demo ideas on the GX10 (pick 2–3)

1. **Hallucination killer (static + live).** Precompute 8–10 night/rain images: BASE boxes vs
   BASE+GNN boxes, removed boxes in red with their outline graphs. Then run one fresh image
   live through detector → SAM 2 → GNN on the GX10.
2. **Color-blindness proof (live, strongest visual).** Apply hue rotation, color inversion, or a
   stylized texture to an image on the spot. Show the detector's boxes changing while the GNN's
   input feature vector (printed) and verdict stay the same.
3. **"What the GNN sees."** Plot the 64-point graphs of a real car, a pedestrian, and a fake
   detection (shadow patch) side by side with the GNN's class probabilities. Shapes alone
   are often obvious to the audience too.
4. **Size sanity slider.** Drag a car-shaped outline up and down the frame; the GNN's "car"
   probability changes because the size no longer fits the position (shows the size cues; compare
   GNN vs GNN-SHAPE).
5. **Dashcam playback.** Play a comma2k19 clip (MIT license) with BASE vs BASE+GNN overlays at
   the same false-alarm rate and a running count. Label it **descriptive** — the footage isn't
   hand-audited.
6. **Local-AI panel.** `nvidia-smi` / `ollama ps` style panel: memory and per-frame latency for
   detector, SAM 2 and GNN; optionally ask Nemotron "is there a truck?" on the same frame to
   contrast a VLM's answer with the GNN's.
7. **Honest limit, physically.** Toy car vs. a printed photo of the toy car on a phone camera:
   both pass the shape check — the GNN can't tell a picture from the real thing (that's the
   Perception Lab problem, a different failure mode).

### 5.3 What to build for the demo (≈ 45 min on a laptop, run on the GX10)

- `scripts/demo_gnn.py IMAGE [--recolor hue|invert|stylize] [--out DIR]`: runs BASE-s0 → SAM 2 →
  GNN on one image, writes a side-by-side PNG (BASE vs BASE+GNN at the T-shift equal-false-alarm
  threshold), per-box outline-graph thumbnails, GNN probability bars, and a JSON of timings
  (detector ms, SAM 2 ms, GNN ms per box) and the GNN input features.
- A tiny HTML page (or slides) showing the precomputed examples; pick examples by a fixed rule
  (e.g., the 5 T-night images with the most removed false detections **and** the 3 with the most
  wrongly removed true positives) and say it's selected.
- Record a fallback screen capture in case the live run stalls.

---

## 6. Questions to be ready for

**About the idea**

- **Why a GNN and not just another CNN?** A CNN on the image sees texture no matter what we
  tell it. A GNN on outline points only receives geometry, so texture-blindness is guaranteed by
  the input, not hoped for from training. Graphs also fit outlines naturally: each point's
  meaning depends on its neighbors along the contour (corners, curves, symmetry).
- **Isn't SAM 2 itself a texture-using network?** Yes — it uses pixels to find the boundary.
  But it doesn't decide *what* the object is; the class decision is texture-blind. If SAM's
  outlines degrade at night, that shows up in the outline success rate we report per condition.
- **Why not train end-to-end?** Time, and clarity: a separate verifier lets us attribute the
  effect. End-to-end is follow-up work.
- **What do the size cues add?** That's the GNN vs GNN-SHAPE comparison ⟨fill⟩.
- **Can the GNN find objects the detector missed?** No — it only re-scores. The recall gain
  comes from being able to lower the detector's threshold without adding false alarms (G2).
- **What about pictures of objects — billboards, truck-side graphics?** A picture of a truck has
  a truck's outline, so it passes. That's a different failure (depiction confusion), the one our
  Perception Lab pilot studied. Depth, motion, or LiDAR would be needed.

**About the evaluation**

- **How do you define a hallucination for a detector?** False detections at IoU 0.5, split into
  invented / wrong class / poor box / duplicate / near another class, counted per image.
- **Why "at equal recall"?** A model that detects less automatically invents less. Fixing recall
  makes the comparison fair; likewise G2 fixes the false-alarm rate.
- **Why train on clear daytime only?** So night, rain, snow and fog are truly unseen — any gain
  there is generalization, which is what texture-blindness predicts.
- **How do you know it isn't luck?** Paired bootstrap over 4,752 images (2,000 replicates),
  thresholds re-estimated in every replicate, Holm correction, and hypotheses frozen in git before
  testing. Limitation: one training seed.
- **Isn't the training set small?** The GNN trains on detections from 624 clear-day images
  (fold A), fusion on the other 624 — ⟨n⟩ detections. Small but realistic: the detector's real
  mistakes. More data via cross-fitting is follow-up.
- **Why 2018 BDD labels?** The official BDD servers were offline; the maintainer's mirror had
  the 2018 labels. Documented as a deviation.
- **Fog has 5 test images?** Yes — reported descriptively, not tested.
- **Did you test on real unfiltered dashcam footage?** We have comma2k19 footage sampled (2,270
  frames, no hand-picking), but the blind human audit was cut for time; any footage counts we show
  are descriptive. The audit protocol is ready.

**About the system**

- **Is it real-time?** ⟨fill: detector ms + SAM 2 ms + GNN ms per frame on the GX10⟩. SAM 2 is
  the bottleneck; a lighter outline model would be the path to real time.
- **What does the GX10 add?** 121 GiB of unified CPU/GPU memory and the GB10 GPU let the detector, SAM 2,
  GNN training, and our local VLMs share one box; nothing leaves it — important for dashcam
  privacy.
- **Would you put this in a car?** No claim. It's a research prototype measuring one failure
  mode; production would need latency, validation, and certification work.
- **How does it compare to stylized training (the known fix)?** ⟨fill from CNN study arms that
  finished; otherwise cite Michaelis et al. 2019 and say the 2×2 was stopped for the GNN pivot⟩.
- **What happened to the CNN study?** Its BASE detector became stage 1; the remaining arms were
  stopped when we refocused on GNNs. ⟨List any finished arms.⟩

**About ethics and data**

- **Privacy?** Footage stays on the GX10; faces and plates blurred in anything shown; no audio.
- **Licenses?** BDD100K and most datasets are research-only; nothing redistributed. comma2k19 is
  MIT.

---

## 7. Slide outline (8 slides)

1. Title + one-line pitch
2. The problem: hallucinations, texture vs shape (Geirhos cat/elephant), night/weather examples
3. Why existing fixes fall short (stylized training still sees texture)
4. Our approach: pipeline diagram — texture-blind GNN verifier with size cues
5. How we test fairly: arms, equal-recall / equal-false-alarm, clear-day training, preregistration
6. Results ⟨fill⟩: headline table + one chart (false detections by type, BASE vs GNN vs RGB)
7. Demo (live) + limits (billboards, SAM at night, one seed)
8. What's next: cross-fitting, more seeds, footage audit, outline quality vs BDD masks, LiDAR
