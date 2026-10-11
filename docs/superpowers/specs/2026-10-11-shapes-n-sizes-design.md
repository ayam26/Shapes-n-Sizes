# Shapes-n-Sizes — Research Design Spec

- **Date:** 2026-10-11
- **Status:** design approved in conversation; awaiting review of this written spec
- **Supersedes for implementation purposes:** `perception-lab-design.md` (kept as background; see §2)
- **Deliverable:** an arXiv / workshop-style paper with released dataset, code, and preregistration

## 1. Summary

Vision-language models (VLMs) are increasingly used to interpret camera images, including road scenes. A known failure family is **flat-depiction confusion**: treating a picture of an object (a billboard, a printed cutout, graphics on a truck) as the object itself. This project asks two questions, in two phases:

1. **Phase 1 — Diagnose.** Which visual depth cues do current open VLMs actually use to tell a real three-dimensional object from a flat picture of it? We render controlled synthetic scenes in which a flat picture is pixel-matched to the real object from the camera's viewpoint, then re-introduce one cue at a time (viewing angle, cast shadows, framing/support, shading consistency) and measure where each model's judgment changes.
2. **Phase 2 — Train.** Does LoRA fine-tuning a small open VLM on synthetic examples (with *different* objects than the benchmark) teach the distinction in a way that generalizes — to held-out objects, a held-out environment, and real photographs — or does it learn shortcuts?

All model inference runs on the user's **ASUS Ascent GX10**; stimulus generation, the runner software, and analysis are built in this repository. Ground truth never reaches a model. Every result — pass, fail, abstention, error — is kept. A null result is a valid result.

This spec covers both phases so the study is designed as one story (Phase 2's need for held-out objects shapes Phase 1's benchmark). Implementation is planned per phase: the first implementation plan covers Phase 1 (gates G1–G6, §13); Phase 2 gets its own plan at G7.

## 2. Relation to `perception-lab-design.md`

The original document specified a 10-hour hackathon build. This project keeps its question and its integrity rules and changes its form.

| Kept | Changed |
|---|---|
| Physical-vs-flat-depiction question | 10-hour clock dropped; staged gates instead (§13) |
| Ground truth kept out of model inputs; opaque filenames | Interactive three-column demo app dropped (non-goal) |
| Strict JSON parsing, `invalid_output` / `timeout` never correct, raw responses saved | Renderer: Blender (`bpy`) + Cycles instead of Three.js — physically based shadows are the cues under study |
| Full provenance (hashes, prompt hashes, model config, timing) | Corpus: 36 captures → factorial design, 380 images per scene seed (§5) |
| Development vs held-out (here "confirmatory") separation, frozen before held-out runs | Primary question: cue reliance (psychophysics-style) rather than prompt intervention; prompt intervention becomes secondary RQ5 |
| Local inference on the GX10; no hidden cloud fallback | Labels generalized for non-vehicle targets: `real_object`, `flat_picture`, `neither`, `uncertain` |
| No claims of safety, vulnerability, or significance beyond evidence | Training added as Phase 2 (the original listed training as a non-goal) |
| | Plain geometric shapes added as a no-semantics control |

## 3. Goals, non-goals, success criteria

**Goals**

- A frozen, versioned synthetic benchmark of matched physical/depiction stimuli with full provenance.
- A preregistered Phase 1 study across a size ladder of open VLMs.
- A preregistered Phase 2 fine-tuning study with generalization and shortcut tests.
- A small human-comparison study establishing which cues are visible in the stimuli.
- A paper draft, with every reported number recomputable from committed raw records (`snz verify`).

**Non-goals**

- Interactive demo application, driving controller, collision prediction, or any safety claim.
- Scraping dashcam footage or downloading driving datasets.
- Cloud-hosted model inference (hosted models may be future work; not in this study).
- Training a model from scratch; Phase 2 is parameter-efficient fine-tuning only.

**Success criteria**

- Phase 1: confirmatory stimuli frozen and hashed before any model sees them; preregistration tagged in git before confirmatory runs; results for ≥ 4 models; all hypotheses reported regardless of outcome.
- Phase 2: preregistration tagged before training; base vs fine-tuned vs control-fine-tuned compared on all generalization sets; shortcut probes and retention reported.
- An honest null or negative result meets these criteria.

## 4. Phase 1 research questions and hypotheses

Frozen in `prereg/phase1.md` (git tag `prereg-phase1-v1`) before confirmatory runs.

**RQ1 — Behavior under perfect illusion.** In the *perfect-illusion cell* (angle 0°, cutout board, overcast, consistent shading) a depiction is nearly pixel-identical to the physical object, so `uncertain` is the normatively correct answer.

- **H1:** Models over-answer `real_object` on depictions in the perfect-illusion cell: among responses that are either `real_object` or `uncertain`, `real_object` is the majority (one-sided).

**RQ2 (primary) — Cue reliance.** Which cues increase correct identification of depictions?

- **H2a — Viewing angle:** P(`flat_picture` | depiction) increases with the target's angle away from the matched view (0–60°).
- **H2b — Shadow:** sunny lighting (hard shadows whose shape reveals the flat board) increases P(`flat_picture` | depiction) relative to overcast.
- **H2c — Framing/support:** a framed billboard on posts increases P(`flat_picture` | depiction) relative to a frameless cutout. **Shortcut check:** a physical object with an empty frame standing beside it receives more `flat_picture` responses than the same object without the frame.
- **H2d — Shading consistency:** a depiction whose baked-in shading is inconsistent with the scene lighting receives more `flat_picture` responses than one whose shading is consistent.

**RQ3 — Object familiarity.**

- **H3 (two-sided):** the effect of viewing angle differs between vehicles and plain geometric shapes (angle × category interaction). Our expectation, stated but not tested directionally, is a stronger "vehicles on roads are real" prior that raises false `real_object` rates for vehicle depictions.

**Secondary / exploratory** (labeled as such in the paper)

- **RQ4:** Does cue sensitivity (the 50%-angle threshold, §7.3) vary with model size?
- **RQ5:** Does a structure-focused prompt (P1) change outcomes relative to the baseline prompt (P0) on byte-identical images?
- Distance (randomized covariate) effects; object-recognition accuracy from the `object` field.

## 5. Stimulus design

### 5.1 Scene

- A straight road on a ground plane, sky, sun and ambient light, and simple procedural background clutter (poles, boxes, low vegetation shapes). All geometry and textures are procedural or self-created.
- **Sensor camera:** 1.5 m above the road surface, aimed at the target centre, which is always at the image centre. Fixed horizontal field of view, chosen during renderer validation so the target spans 15–50% of the image width across the distance range.
- **Target marker:** fixed corner brackets drawn post-render (Pillow) at fixed pixel coordinates around the image centre. The marker is pixel-identical across all images and so cannot carry class information.
- **Output:** 512×512 PNG, Blender Cycles, fixed sample count and fixed sampler seed, OpenImageDenoise on CPU, all PNG text/metadata chunks stripped. Blender version pinned (`bpy==5.2.2`).

### 5.2 Targets and conditions

- **Benchmark objects (6):** vehicles — box truck, sedan, van; shapes — box, cylinder, sphere. Realistic dimensions (e.g., truck ≈ 7 m long). Each object has a fixed base yaw (vehicles in a natural three-quarter view). These six objects are **never** used for Phase 2 training.
- **Physical:** the 3D object itself.
- **Depiction:** a picture of the object on a 2 cm-thick board. The picture texture is a render of the physical object **from the sensor camera's exact pose at angle 0°**, so at 0° the depiction reproduces the physical object's appearance.
  - *Cutout* style: board cut to the object's silhouette (alpha), no frame — a printed standee.
  - *Framed* style: rectangular billboard with frame and posts; object on a plain light-grey background.
  - Board material is diffuse (a print), with one exposure-calibration factor per lighting condition so that at 0° the target region's mean luminance matches the physical render. Factors are recorded.
- **Blank board (truth `neither`):** a board with no picture. Styles: *framed* (as above) or *panel* (rectangular, unframed). Blank boards are never silhouette-shaped (a blank silhouette would itself be an ambiguous depiction).

### 5.3 Factors

Crossed (systematically varied):

| Factor | Levels | Applies to |
|---|---|---|
| Angle (yaw of the target about its vertical axis, relative to the matched view; always counter-clockwise seen from above) | 0, 15, 30, 45, 60° | all |
| Lighting | sunny (hard shadows) / overcast (diffuse) | all |
| Board style | cutout / framed (depictions); panel / framed (blank) | depictions, blank boards |
| Context | none / empty frame standing beside the object | physical |
| Shading consistency | consistent / inconsistent | depictions |

- The picture texture is always rendered at 0°; at other angles the board (and the physical object) rotate, so the depiction becomes foreshortened and its edge thickness becomes visible.
- *Consistent* shading: the texture is rendered under the scene's own lighting. *Inconsistent*: the texture is rendered under a directional sun whose azimuth is mirrored across the camera's vertical viewing plane (sunny scenes), or under a directional sun where the scene is overcast.
- *Context — empty frame:* the same frame-and-posts structure as the framed billboard, with no board, standing beside the physical object without occluding it; side (left/right) randomized per image.
- Rotation-invariant shapes (sphere, vertical cylinder) look identical at all angles when physical; for them the angle cue lives entirely in the depiction. This is expected and reported.

Randomized per image (recorded as covariates, not crossed): target distance uniform 15–35 m; sun azimuth/elevation jitter within the lighting condition; background clutter layout and ground texture variation. All are drawn from a RNG seeded by (scene seed, cell index).

### 5.4 Counts and splits

Per scene seed: depictions 6 × 5 × 2 × 2 × 2 = **240**; physical 6 × 5 × 2 × 2 = **120**; blank 5 × 2 × 2 = **20**; total **380**.

| Split | Seeds | Images | Use |
|---|---|---|---|
| development | 1 (seed 1000) | 380 | Pipeline validation, response-format compliance. Never used to tune prompts for accuracy. |
| confirmatory | 4 provisional (2001–2004), final count from §7.5, max 8 | 1,520 (provisional) | Preregistered hypothesis tests. Rendered and hashed but shown to no model until `prereg-phase1-v1` is tagged. |

Depictions outnumber physical objects by design; all analyses are conditional on truth class, never pooled accuracy.

### 5.5 Stimulus validation (before freezing)

- Every image: target visible (object-index mask pixel count above threshold), marker present and pixel-identical, PNG carries no text chunks, scene spec re-renders to the same hash on the same machine.
- **Illusion quality:** SSIM between physical and depiction renders in the perfect-illusion cell, computed within the target region, reported per object. Calibration is fixed if SSIM is poor; the threshold is set during renderer validation and recorded in the preregistration.
- Measured render time per image and PNG size per image, recorded in `artifacts/render-benchmark.md`.
- A rejected render is recorded with a reason, never silently dropped.

## 6. Model protocol

### 6.1 Models and runtime

- A size ladder in at least two families (e.g., Qwen-VL roughly 3B–72B; Gemma 3 4B/12B/27B), plus one or two further families, run on the GX10.
- Runtime: any OpenAI-compatible chat-completions server with image input. vLLM is preferred because it can return per-token log-probabilities; Ollama is acceptable.
- Exact model identifiers, weight revisions, quantization, runtime and version are fixed in `experiments/model-lock.json` only after a smoke test on the GX10 (one real image request per model, response parsed successfully). The lock file is part of the preregistration. A text-only completion does not count as a smoke test.

### 6.2 Task, labels, output

- Labels: `real_object` (a real three-dimensional object standing in the scene), `flat_picture` (a flat picture or printed image of an object, e.g., on a sign or board), `neither` (no object or picture of an object present, e.g., a blank surface), `uncertain`.
- Correct answers: physical → `real_object`; depiction → `flat_picture`; blank → `neither`. `uncertain` is never correct and is reported separately as abstention.
- Output: a single JSON object `{"label": ..., "object": ..., "evidence": ...}`. The label comes first so the immediate judgment is captured and its token log-probability is cleanly readable.

### 6.3 Prompts

Draft text below; exact text is frozen in `experiments/prompts.json` (with SHA-256) before the preregistration tag. Development runs may change wording only to fix format compliance, never to raise accuracy.

**P0 (baseline):**

> Look at the target marked by the corner brackets in the center of this image and decide what it is. Allowed labels: "real_object" — a real three-dimensional object standing in the scene; "flat_picture" — a flat picture or printed image of an object, for example on a sign or board; "neither" — no object or picture of an object is present, for example a blank surface; "uncertain" — the image does not provide enough visible evidence to decide. Use only visible evidence. Respond with only a JSON object of the form {"label": "<allowed label>", "object": "<what the target is or depicts, in a few words>", "evidence": "<short description of the visible evidence>"}.

**P1 (structure-focused):** P0 plus the fixed sentence:

> When deciding, consider the target's visible thickness, its perspective, how it is supported, and whether its shading and shadows are consistent with the lighting in the scene.

### 6.4 Decoding and execution

- Temperature 0, fixed seed where supported, `max_tokens` 200, one sample per (image, prompt, model).
- Image sent as the original PNG bytes (base64 data URL). The runtime's preprocessing (resize, patching) is recorded in the lock file where known.
- Request order shuffled per (model, prompt) with a fixed seed.
- Repeatability: a seeded random 5% of requests is re-run after the main pass and stored with `rerun: true`; the disagreement rate is reported; reruns are excluded from primary analysis.
- Warm-up requests are excluded from latency statistics.

### 6.5 Parsing

- Accept a response that is exactly one JSON object, optionally wrapped in a single fenced code block. Nothing else.
- `label` must be one of the four allowed strings exactly (after trimming surrounding whitespace). `object` and `evidence` are optional strings. Extra keys are ignored.
- Anything else → status `invalid_output` (with a machine-readable reason). Request exceeding the timeout → `timeout`. No semantic repair, no second model.
- Transport errors (connection refused/reset, HTTP 5xx) retried once. Invalid outputs are never retried.
- Raw response text is always stored.

### 6.6 Ground-truth separation

- Image files are named by opaque IDs; the stimulus manifest given to the runner contains only `image_id`, `sha256`, `path`.
- Scene specifications and truth labels live in `data/truth/`, read only by analysis code. The runner package does not import the truth module, and its tests pass with `data/truth/` absent.
- A test asserts that no request payload contains truth labels, factor names, or descriptive strings from the scene spec.

### 6.7 Result record

One JSON line per request in `data/results/<split>/<model_id>__<prompt_id>.jsonl` (append-only):

`run_id, image_id, image_sha256, prompt_id, prompt_sha256, model_id, model_lock_sha256, runtime, decoding (temperature, seed, max_tokens), cache_key, started_at, elapsed_ms, status (ok | invalid_output | timeout | transport_error), invalid_reason, raw_response, parsed (label, object, evidence), label_logprobs (if available), rerun, provenance (live | fixture)`.

`cache_key = sha256(image_sha256, prompt_sha256, model_lock_sha256, decoding, preprocessing_version)`. The runner skips requests whose cache key already has a terminal record, so restarts never duplicate work and any prompt/model/decoding change invalidates the cache.

## 7. Phase 1 analysis plan

Frozen in the preregistration. Run separately per model; each model is a replication unit.

### 7.1 Measures

- **Response rates** per truth class and cell: P(`real_object`), P(`flat_picture`), P(`neither`), P(`uncertain`), among valid responses.
- **Discrimination:** d′ = z(H) − z(F), with H = P(`flat_picture` | depiction) and F = P(`flat_picture` | physical) within matched angle × lighting cells, log-linear correction for rates of 0 or 1. Separates sensitivity from a bias toward answering "picture".
- **Coverage:** 1 − P(`uncertain`), reported beside every accuracy figure. Abstention is never scored as an error in the d′ analysis.
- **Operational:** counts of `invalid_output`, `timeout`, `transport_error`; latency p50/p95 with n.
- Absolute counts are shown beside every percentage.

### 7.2 Hypothesis tests

- **H1:** exact one-sided binomial test of `real_object` vs `uncertain` counts among depiction responses in the perfect-illusion cell (confirmatory seeds pooled).
- **H2a–d, H3:** logistic regression on depiction trials of `said flat_picture` ~ angle (continuous, scaled) + lighting + board style + shading consistency + category + angle × category + distance (covariate). Standard errors cluster-robust by (scene seed × object). H2a–d and H3 are tests of the corresponding coefficients (H3 = interaction).
- **Shortcut check:** logistic regression on physical trials of `said flat_picture` ~ context + angle + lighting + category + distance; test the context coefficient.
- **Multiplicity:** Holm correction across {H1, H2a, H2b, H2c, H2d, H3, shortcut} within each model, α = 0.05.
- **Across models:** report, per hypothesis, how many models show a significant effect in the predicted direction, with each model's estimate and CI. No pooled cross-model p-value.

### 7.3 Psychometric thresholds

Per model, fit a logistic psychometric function of P(`flat_picture` | depiction) vs angle (lapse rate bounded), report the 50%-angle θ50 with 95% bootstrap CIs (resampling seeds and objects). When the curve never crosses 50% in 0–60°, report "> 60°" rather than extrapolating. RQ4 compares θ50 against parameter count descriptively. Where log-probabilities are available, the fits are repeated on P(`flat_picture`) as a secondary analysis.

### 7.4 Missing data

Invalid outputs and timeouts are excluded from the regressions and reported in full. Sensitivity analysis: all models refit with invalid/timeout counted as incorrect (`said flat_picture` = 0 on depiction trials).

### 7.5 Power analysis (gate G2)

Simulation-based, before the preregistration is frozen. Simulate responses from the planned regression with assumed effects: odds ratio 2 for each binary cue; detection rising from 20% to 80% across 0–60°; a perfect-illusion `real_object` rate of 70%. For seed counts n = 2…8, simulate 500 datasets each, fit the planned models, and compute Holm-adjusted rejection rates. Choose the smallest n with ≥ 80% power for H1 and H2a–d. If H3 or the shortcut check stays below 80% at n = 8, it is moved to exploratory in the preregistration before confirmatory runs. The simulation code, seed, and output are committed.

### 7.6 Human comparison study

- 10–20 adult volunteers judge ~120 confirmatory images, stratified across angle × lighting × board style × category × truth class, using the same four labels, via a static HTML form. Responses saved as CSV; no personal data beyond an anonymous participant code.
- Purpose: establish whether each cue is visible in our renders (a model failing on an invisible cue is uninformative) and provide a human reference curve.
- Requires an IRB determination (UCSC; the exempt category typically covers this task, but the IRB decides). No participant data is collected before that determination. Fallback, if the determination is not obtained in time: authors act as observers, reported as such.

## 8. Phase 2 — Training and generalization

Phase 2 starts only after Phase 1 confirmatory analysis. Its own preregistration (`prereg/phase2.md`, tag `prereg-phase2-v1`) is frozen before any training run.

### 8.1 Model selection rule

Fine-tune the smallest Phase 1 model with (a) d′ below 1.0 averaged over the angle 0–15° cells and (b) working LoRA support on the GX10. A second, larger model is optional. **Ceiling rule:** if no model meets (a), Phase 2 restricts training and evaluation to the hardest cells (lowest pooled d′) and to models below 10B parameters, and says so.

### 8.2 Training setup

- Hugging Face Transformers + PEFT (LoRA), bf16, on the GX10 (Arm + CUDA). First step: a tiny smoke fine-tune to confirm the toolchain (NVIDIA's DGX Spark playbooks cover the same platform).
- **Training objects (disjoint from the benchmark):** vehicles — pickup, bus, hatchback; shapes — cone, pyramid, torus.
- Fresh scene seeds, wider randomization ranges, balanced `real_object` / `flat_picture` / `neither` targets.
- Training target: `{"label": ..., "object": ...}`. `uncertain` is not trained; whether fine-tuning removes appropriate abstention is measured and reported as a risk.
- Training set size from a learning-curve pilot (1k / 3k / 6k images, scored on a Phase 2 development split built from training objects) and measured render time; ~6k expected. Transfer of training images to the GX10 is decided in the Phase 2 implementation plan.

### 8.3 Conditions

- **Base:** the selected model, zero-shot, prompt P0.
- **Fine-tuned:** LoRA on real/picture/neither labels plus object names.
- **Control fine-tune:** identical images and procedure, trained to output object names only. Separates adaptation to our renders from learning the distinction.

### 8.4 Evaluation sets (easiest to hardest generalization)

1. **Held-out objects:** the Phase 1 confirmatory benchmark.
2. **Held-out environment:** a newly rendered urban street scene (buildings, sidewalks) never seen in training, same factor design.
3. **Real photographs:** ~100–200 photos taken by the user: toy vehicles and simple objects, and printed photos of them mounted on card (cutout and framed), at several angles, indoor and outdoor light. Labels known by construction; self-created, license recorded. This is the paper's only real-world evidence.
4. **Shortcut probes:** cue-conflict images — e.g., a cutout with a painted-on shadow, a physical object inside a frame.
5. **Retention:** a fixed subset of POPE (object-hallucination benchmark), run on the GX10 before and after fine-tuning.

### 8.5 Hypotheses

- **H4:** fine-tuned > base in d′ on held-out objects.
- **H5:** the gain transfers to the held-out environment and to real photos (smaller effect expected).
- **H6 (two-sided):** fine-tuning changes the shortcut error rate on cue-conflict probes.
- **Retention:** change in POPE accuracy, descriptive.
- Paired per-image comparisons (McNemar) between base, fine-tuned, and control; Holm correction within the Phase 2 family.

Claims are limited to measured improvements on these sets. No claim of improved vehicle safety.

## 9. Software architecture

Python 3.13 (required by `bpy` 5.2.2), managed with `uv`; package `snz` under `src/`; `pytest`, `ruff`.

| Extra | Contents | Runs on |
|---|---|---|
| core | spec, design, parse, runner (`httpx`), Pillow | everywhere incl. GX10 |
| `[render]` | `bpy==5.2.2` | x86-64 Linux (this container) or macOS |
| `[analysis]` | pandas, numpy, scipy, statsmodels, scikit-image, matplotlib | analysis machine |
| `[train]` | torch, transformers, peft | GX10, Phase 2 |

```text
src/snz/
  spec.py          SceneSpec dataclasses; validation; canonical JSON; hashing
  design.py        experiment config → list of SceneSpecs; split assignment; balance checks
  render/          bpy-only
    objects.py     procedural builders for benchmark and training objects
    depiction.py   matched-view texture render; cutout/framed/panel boards; calibration
    scene.py       road, sky, lights, clutter, camera from a SceneSpec
    capture.py     render → PNG; strip metadata; draw marker; mask check; SHA-256
  runner/
    client.py      OpenAI-compatible client; base64 image; timeout; transport retry
    prompts.py     load frozen prompts; hash
    run.py         resumable batch runner; cache key; JSONL append; fixture provider
  parse.py         strict response parsing
  analysis/
    metrics.py     rates, d′, coverage, operational summaries
    models.py      regressions, psychometric fits, bootstrap
    power.py       simulation-based power analysis
    report.py      paper tables and figures
  cli.py           snz design | render | validate | run | analyze | verify
experiments/       phase1.toml, prompts.json, model-lock.json
data/stimuli/<split>/   <image_id>.png, manifest.jsonl (image_id, sha256, path)
data/truth/<split>.jsonl          scene specs + truth (analysis only)
data/results/<split>/<model_id>__<prompt_id>.jsonl
prereg/            phase1.md, phase2.md
human-study/       static form, stratified image list, response CSVs
artifacts/         render-benchmark.md, environment notes, figures
tests/
```

Unit boundaries: `spec`/`design` know nothing about Blender; `render` consumes SceneSpecs and produces files plus hashes; `runner` consumes only the stimulus manifest, prompts, and the model lock; `analysis` joins results with truth. Each is testable alone.

## 10. Data management and provenance

- Stimulus PNGs and manifests are committed to git: development split first; confirmatory split pushed in chunks. If the total exceeds ~1 GB, the frozen splits move to a GitHub Release tarball uploaded by the user (repo hashes remain authoritative).
- Datasets are frozen as files, not as re-renders: cross-platform Blender output need not be byte-identical, so hashes in the manifests define the dataset.
- Results JSONL files are append-only and committed by the user after GX10 runs.
- Fixture (stand-in) records carry `provenance: fixture`; analysis refuses them unless explicitly run with `--allow-fixture` (tests only).
- No secrets in the repo: the model lock stores the *name* of an API-key environment variable if one is needed, never its value.

## 11. Testing and verification

**Unit tests (no Blender):**

1. SceneSpec canonical JSON and hash are stable; invalid specs rejected.
2. Design generator yields exactly 380 specs per seed with correct per-cell counts and split assignment.
3. Parser: all four labels; unknown label; malformed JSON; fenced JSON; multiple objects; empty output; extra keys.
4. Metrics on hand-built mixed outcomes: denominators, abstentions, operational failures, d′ with 0/1 correction.
5. Runner against a fake OpenAI-compatible server (`httpx.MockTransport`): resume without duplicates; cache invalidates on prompt/model/decoding change; transport errors retried once; invalid outputs not retried; timeouts recorded.
6. Truth-leak test: payloads contain no truth labels, factor names, or scene-spec strings; runner works with `data/truth/` removed.
7. Power simulation recovers a planted effect.

**Render tests (marked `render`, require `bpy`):** tiny scene renders; same spec → same hash twice on one machine; target mask non-empty; marker pixels identical across images; no PNG text chunks; matched-pair SSIM in the perfect-illusion cell above the recorded threshold.

**`snz verify`:** recomputes every stimulus hash against its manifest and every reported number from raw JSONL plus truth.

**CI:** GitHub Actions runs unit tests and `ruff` on push. Render tests run locally (the `bpy` wheel is ~400 MB).

## 12. Roles

- **Claude (this repo, x86-64 container without GPU or Hugging Face access):** code, stimulus rendering, power analysis, analysis, figures, paper drafts.
- **User (GX10):** model serving and smoke tests, batch runs (`snz run`), committing result files, Phase 2 training runs, real photographs, IRB determination, human-study recruitment.

## 13. Gates

| Gate | Condition to pass |
|---|---|
| G0 | This spec reviewed and approved |
| G1 | Renderer validated: stimulus checks pass, SSIM threshold set, render time and file size measured |
| G2 | Power analysis run; confirmatory seed count fixed |
| G3 | GX10 smoke test per model; `model-lock.json` committed |
| G4 | Development runs complete; prompts frozen; `prereg/phase1.md` committed and tagged `prereg-phase1-v1` |
| G5 | Confirmatory stimuli rendered, hashed, committed; then confirmatory runs complete |
| G6 | Phase 1 analysis, `snz verify` clean, figures |
| G7 | Phase 2 preregistration tagged; then training and evaluation |
| G8 | Paper draft |

Confirmatory stimuli may be rendered before G4 but are not sent to any model until G4 passes. Any change to a confirmatory image after a model has seen it creates a new exploratory case.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CPU rendering too slow | Measure at G1; reduce sample count with denoising; render in background chunks |
| Illusion imperfect at 0° (luminance/shading mismatch) | Per-lighting calibration; SSIM check gates G1 |
| Models fail JSON format | Development split exists to catch this; format-only prompt fixes before freezing |
| Runtime lacks log-probabilities | Log-probability analyses are secondary; primary analyses use labels only |
| H3 / shortcut check underpowered | Power analysis demotes them to exploratory before freezing |
| Repository grows too large | Measure PNG size at G1; Release-tarball fallback (§10) |
| Arm/CUDA toolchain problems for Phase 2 | Smoke fine-tune first; NVIDIA platform containers/playbooks |
| IRB delay | Authors-as-observers fallback, reported as such |
| Phase 1 ceiling (models already succeed) | Phase 2 ceiling rule (§8.1); a ceiling result is itself reported |
| Fine-tuning learns render-specific shortcuts | Held-out environment, real photos, cue-conflict probes, control fine-tune |

## 15. References

- Geirhos, R., et al. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. ICLR. arXiv:1811.12231.
- Eykholt, K., et al. (2018). Robust physical-world attacks on deep learning visual classification. CVPR.
- Li, Y., et al. (2023). Evaluating object hallucination in large vision-language models (POPE). EMNLP. arXiv:2305.10355.
- Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.
- Green, D. M., & Swets, J. A. (1966). Signal Detection Theory and Psychophysics. Wiley.
- Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I. Fitting, sampling, and goodness of fit. Perception & Psychophysics, 63(8).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2).

A fuller related-work survey (VLM spatial reasoning, depth cue literature, depiction perception) is part of the paper-writing stage, not a prerequisite for building.
