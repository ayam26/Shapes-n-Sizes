# Shape-Biased Detector — Training Pipeline Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `snz` package from data splits through training and id-val evaluation, so gates G1 (GX10 smoke test), G2 (data build) and G3 (pilot) can run on the GX10.

**Architecture:** A `uv`-managed Python package `snz` under `src/`, with one module per responsibility: `data` (BDD labels → split manifests), `stylize` (AdaIN → stylized copies), `train` (Faster R-CNN R50-FPN, dataset, iteration-based training loop with resume), and `eval` (predict + `pycocotools` mAP, with a gate that refuses test splits before the `prereg-v1` tag). Each stage communicates through files (manifests, image folders, checkpoints, JSONL detections), so stages run and are tested on their own. Tests run on CPU with tiny synthetic fixtures; all real work runs on the GX10.

**Tech Stack:** Python 3.12, uv, PyTorch ≥ 2.9 + torchvision ≥ 0.24 (CUDA 13.0 wheels on the GX10, CPU elsewhere), Pillow, NumPy, pycocotools, pytest, ruff.

**Spec:** [`docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md`](../specs/2026-10-11-shape-biased-detector-design.md). Read it first; this plan argues from it.

**Follow-on plans (not in this plan):**
- **Plan 2 — Analysis:** `eval/matching.py`, `eval/operating.py`, `eval/bootstrap.py`, `eval/report.py`, `corrupt.py`, `prereg/prereg.md` draft (gates G4, G6).
- **Plan 3 — Footage & audit:** `data/footage.py`, `audit/` (gate G7).

## Global Constraints

- Detector: torchvision Faster R-CNN, ResNet-50 + FPN; backbone ∈ {`in1k_v1` = torchvision `ResNet50_Weights.IMAGENET1K_V1`, `sin` = Geirhos `resnet50_train_60_epochs-c8e5653e.pth.tar`}.
- Detection layers (FPN, RPN, ROI heads) freshly initialized in every arm; **no COCO-pretrained detector weights**.
- Frozen batch norm in the backbone; torchvision default trainable backbone layers (3); bf16 mixed precision on the GX10; identical iterations for all arms; evaluate the **final checkpoint** (no checkpoint selection).
- Input at native 1280×720 with scale jitter and horizontal flip (p = 0.5).
- Arms: BASE (in1k_v1, normal data), STY (in1k_v1, half-stylized), SIN (sin, normal), BOTH (sin, half-stylized). Seeds 0, 1, 2.
- Half-stylized: each sample drawn from {original, copy 1, copy 2} with P(original) = 0.5, P(each copy) = 0.25; 2 stylized copies per training image, precomputed; AdaIN α = 1.0.
- Style pool split 80/20 into training and test styles with a fixed seed; T-texture uses test styles only.
- 10 BDD classes; primary mAP averages the 9 classes excluding `train`.
- Splits: train / id-val (10% of daytime ∧ clear BDD train, seeded) / T-clean (val daytime ∧ clear) / T-shift night (val night, any weather) / rain, snow, fog (val daytime ∧ rainy/snowy/foggy). Dawn/dusk, partly cloudy, overcast, undefined excluded.
- Manifests `manifests/<split>.csv`: image name, SHA-256. No image in more than one of train, id-val, and any test split.
- **No test split is evaluated before the `prereg-v1` git tag (G4).**
- Never commit images, labels, stylized images, checkpoints, or weights. Commit code, configs, manifests, small result summaries.
- Recipe items fixed by the pilot (learning rate, schedule length, batch size, scale-jitter range) live in `configs/recipe.toml` and are frozen at G3.
- Known weight checksums (sha256):
  - `resnet50_train_60_epochs-c8e5653e.pth.tar` `c8e5653e47bbf3f6efad5da531f3acea9963acb93362b6530cf39b2ba4750da3`
  - `decoder.pth` `379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472`
  - `vgg_normalised.pth` `804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a`

## Review Focus

1. **The same image under two names in two splits** (BDD has near-duplicates): a name-only check would pass silently. Expected: duplicates found by SHA-256; train/id-val copies of test images, and id-val copies of train images, are dropped and logged; any other overlap stops the build. → Task 3.
2. **Messy BDD labels**: frames whose `labels` is `null`, unknown categories (`other vehicle`, `trailer`, `other person`), zero-width boxes, and missing attributes. Expected: empty frames kept, unknown/degenerate boxes dropped and counted, missing attributes read as `undefined`, never a crash mid-training. → Tasks 2 and 7.
3. **A multi-day training run dies partway** (power loss, OOM, SSH drop). Expected: rerunning the same command resumes from `last.pt` and ends with the same weights as an uninterrupted run; a diverging loss stops the run with a clear error instead of writing a NaN checkpoint. → Task 8.
4. **The stylized-copy build is interrupted, or run again on a subset**. Expected: the same image always gets the same style; finished files are skipped; no half-written JPEGs; at training time a missing stylized copy is an error, not a silent fallback to the original (which would quietly turn STY into BASE). → Tasks 5 and 7.
5. **A truncated or wrong weights file** (an HTML error page saved as `.pth.tar`, or the wrong SIN variant). Expected: a checksum error naming the file before any loading, never a silently random backbone. → Task 6.

---

## File Structure

```text
pyproject.toml                 uv project, deps, torch index routing, pytest/ruff config
uv.lock                        generated by `uv lock`
.github/workflows/ci.yml       ruff + pytest on CPU
.gitignore                     + data/, runs/, *.pt, *.pth, *.tar
configs/splits.toml            seed, id-val fraction
configs/stylize.toml           style seed, test fraction, copies, alpha, style size
configs/recipe.toml            training recipe (pilot items marked; `frozen` flag)
configs/arms.toml              BASE / STY / SIN / BOTH
src/snz/__init__.py
src/snz/cli.py                 `snz` command: build-splits, stylize, train, evaluate, smoke
src/snz/hashing.py             sha256_file, verify_sha256, KNOWN_SHA256
src/snz/data/__init__.py
src/snz/data/bdd.py            Frame/Box, load_frames, build_splits, select_frames
src/snz/data/manifest.py       write/read manifests, duplicate dropping, disjointness check
src/snz/data/build.py          build_and_write_splits (orchestrates the two above)
src/snz/stylize/__init__.py
src/snz/stylize/adain.py       AdaIN nets, adain(), stylize_tensor(), Stylizer
src/snz/stylize/pool.py        list_styles, split_style_pool
src/snz/stylize/build.py       style_for, stylized_name, build_stylized, run_stylize, StylizeConfig
src/snz/train/__init__.py
src/snz/train/model.py         load_backbone_state, build_model
src/snz/train/dataset.py       choose_variant, frame_target, BDDDetection, collate
src/snz/train/config.py        Recipe, Arm, load_recipe, load_arms
src/snz/train/train.py         SeededStream, lr_factor, train_run
src/snz/train/smoke.py         run_smoke (gate G1)
src/snz/eval/__init__.py
src/snz/eval/guard.py          ensure_split_allowed (G4 gate)
src/snz/eval/predict.py        predict, write/read detections
src/snz/eval/coco_eval.py      evaluate (pycocotools mAP, AP50, per-class AP)
scripts/gx10_setup.sh          uv + CUDA PyTorch on the GX10
scripts/fetch_weights.sh       SIN + AdaIN + IN1K weights, checksum-verified
scripts/fetch_styles.sh        Describable Textures Dataset
scripts/fetch_bdd.sh           BDD100K images + det_20 labels (run after accepting terms)
docs/runbook.md                G1–G3 commands for the GX10 operator
results/g1_smoke.json          G1 output (committed)
tests/helpers.py               synthetic images / BDD label files
tests/test_cli.py  tests/test_bdd.py  tests/test_manifest.py  tests/test_adain.py
tests/test_stylize_build.py  tests/test_model.py  tests/test_dataset.py
tests/test_train.py  tests/test_eval.py  tests/test_smoke.py
```

Data layout on the GX10 (outside git), root `$SNZ_DATA` (default `~/snz-data`):

```text
weights/        resnet50_train_60_epochs-c8e5653e.pth.tar, decoder.pth, vgg_normalised.pth
bdd100k/images/100k/{train,val}/*.jpg
bdd100k/labels/det_20/det_{train,val}.json
dtd/images/<category>/*.jpg
stylized/train/  stylized/t-texture/
runs/<ARM>-s<seed>/  runs/pilot/<name>/
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/snz/__init__.py`, `src/snz/cli.py`, `tests/test_cli.py`, `tests/helpers.py`, `.github/workflows/ci.yml`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `snz.cli.main(argv: list[str] | None = None) -> int`; `snz.cli.build_parser() -> argparse.ArgumentParser`. Later tasks add subcommands inside `build_parser` with `p = commands.add_parser(...)` and `p.set_defaults(func=<handler(args) -> int>)`.
- Produces (tests): `tests/helpers.py` with `write_image(path, size=(64, 48), color=(120, 80, 40)) -> Path`, `frame_json(name, weather="clear", timeofday="daytime", labels=()) -> dict`, `write_labels(path, frames) -> Path`.

- [ ] **Step 1: Install uv on the Mac if missing**

Run: `command -v uv || brew install uv`
Expected: a path to `uv`, or a successful install.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "snz"
version = "0.1.0"
description = "Shape-biased road-object detector study (Shapes-n-Sizes)"
requires-python = ">=3.12"
dependencies = [
  "numpy>=1.26",
  "pillow>=10",
  "pycocotools>=2.0.8",
  "torch>=2.9",
  "torchvision>=0.24",
]

[project.scripts]
snz = "snz.cli:main"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/snz"]

# GX10 (Linux aarch64) gets CUDA 13.0 wheels; CI (Linux x86_64) gets small CPU wheels;
# macOS uses PyPI.
[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
  { index = "pytorch-cpu", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
]
torchvision = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
  { index = "pytorch-cpu", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 3: Write the failing CLI test**

`tests/test_cli.py`:

```python
import pytest

from snz.cli import main


def test_cli_requires_a_command():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_cli_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz'` (uv creates `.venv` and `uv.lock` on first run).

- [ ] **Step 5: Write the package and CLI**

`src/snz/__init__.py`:

```python
"""Shapes-n-Sizes: shape-biased road-object detector study."""
```

`src/snz/cli.py`:

```python
"""Command-line entry point: `snz <command>`."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snz", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)  # noqa: F841 (tasks add here)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
```

`tests/helpers.py`:

```python
"""Synthetic fixtures: tiny images and BDD-format label files."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def write_image(path: Path, size=(64, 48), color=(120, 80, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def frame_json(name, weather="clear", timeofday="daytime", labels=()):
    """labels: iterable of (category, x1, y1, x2, y2)."""
    return {
        "name": name,
        "attributes": {"weather": weather, "timeofday": timeofday, "scene": "city street"},
        "labels": [
            {"id": str(i), "category": c, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
            for i, (c, x1, y1, x2, y2) in enumerate(labels)
        ],
    }


def write_labels(path: Path, frames) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(frames)))
    return path
```

- [ ] **Step 6: Run tests and lint**

Run: `uv run pytest tests/test_cli.py -v && uv run ruff check src tests`
Expected: 2 passed; `All checks passed!`

- [ ] **Step 7: CI workflow and ignores**

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync
      - run: uv run ruff check src tests
      - run: uv run pytest
```

Append to `.gitignore`:

```text
# snz: data, weights, and run outputs never go in git
/data/
/runs/
*.pt
*.pth
*.pth.tar
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/snz tests .github .gitignore
git commit -m "snz: project scaffold (uv, CLI, CI)"
```

---

### Task 2: BDD labels and study splits

**Files:**
- Create: `src/snz/data/__init__.py` (empty), `src/snz/data/bdd.py`
- Test: `tests/test_bdd.py`

**Interfaces:**
- Produces:
  - `CLASSES: tuple[str, ...]` (10, det_20 names), `EVAL_CLASSES` (9, no `train`), `TEST_SPLITS = ("t-clean", "t-night", "t-rain", "t-snow", "t-fog")`, `TRAIN_SPLITS = ("train", "id-val")`
  - `Box(category, x1, y1, x2, y2)`, `Frame(name, weather, timeofday, boxes: tuple[Box, ...])` (frozen dataclasses)
  - `LoadStats(frames, boxes, dropped_unknown: dict[str, int], dropped_degenerate)`
  - `load_frames(path) -> tuple[list[Frame], LoadStats]`
  - `build_splits(train_frames, val_frames, seed: int, idval_fraction: float) -> dict[str, list[str]]` (keys: `train`, `id-val`, then `TEST_SPLITS`; each sorted)
  - `select_frames(frames, names) -> list[Frame]` (manifest order; `KeyError` if any name is missing)

- [ ] **Step 1: Write the failing tests**

`tests/test_bdd.py`:

```python
import json

import pytest
from helpers import frame_json, write_labels

from snz.data.bdd import CLASSES, EVAL_CLASSES, Frame, build_splits, load_frames, select_frames


def test_classes():
    assert len(CLASSES) == 10
    assert EVAL_CLASSES == tuple(c for c in CLASSES if c != "train")


def test_load_frames_normalizes_and_drops(tmp_path):
    frames = [
        frame_json("a.jpg", labels=[
            ("car", 0, 0, 10, 10),
            ("person", 1, 1, 5, 9),        # 2018 alias -> pedestrian
            ("bike", 1, 1, 5, 9),          # alias -> bicycle
            ("motor", 1, 1, 5, 9),         # alias -> motorcycle
            ("other vehicle", 0, 0, 5, 5), # unknown -> dropped, counted
            ("trailer", 0, 0, 5, 5),
            ("car", 5, 5, 5.5, 20),        # width < 1 px -> dropped
        ]),
        {"name": "b.jpg", "attributes": {"weather": "rainy", "timeofday": "night"}, "labels": None},
        {"name": "c.jpg"},                 # no attributes, no labels
    ]
    loaded, stats = load_frames(write_labels(tmp_path / "l.json", frames))
    a, b, c = loaded
    assert [x.category for x in a.boxes] == ["car", "pedestrian", "bicycle", "motorcycle"]
    assert b.boxes == () and (b.weather, b.timeofday) == ("rainy", "night")
    assert (c.weather, c.timeofday) == ("undefined", "undefined")
    assert stats.frames == 3 and stats.boxes == 4
    assert stats.dropped_unknown == {"other vehicle": 1, "trailer": 1}
    assert stats.dropped_degenerate == 1


def test_load_frames_ignores_non_box_labels(tmp_path):
    item = frame_json("a.jpg", labels=[("car", 0, 0, 10, 10)])
    item["labels"].append({"id": "9", "category": "lane", "poly2d": []})
    loaded, _ = load_frames(write_labels(tmp_path / "l.json", [item]))
    assert len(loaded[0].boxes) == 1


def test_load_frames_rejects_duplicate_names(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps([frame_json("a.jpg"), frame_json("a.jpg")]))
    with pytest.raises(ValueError, match="a.jpg"):
        load_frames(path)


def _f(name, weather, tod):
    return Frame(name, weather, tod, ())


def test_build_splits_membership():
    train = [_f(f"tr{i}.jpg", "clear", "daytime") for i in range(20)] + [
        _f("tr_night.jpg", "clear", "night"),
        _f("tr_rain.jpg", "rainy", "daytime"),
    ]
    val = [
        _f("v_clean.jpg", "clear", "daytime"),
        _f("v_night_clear.jpg", "clear", "night"),
        _f("v_night_rain.jpg", "rainy", "night"),
        _f("v_rain.jpg", "rainy", "daytime"),
        _f("v_snow.jpg", "snowy", "daytime"),
        _f("v_fog.jpg", "foggy", "daytime"),
        _f("v_dusk.jpg", "clear", "dawn/dusk"),
        _f("v_overcast.jpg", "overcast", "daytime"),
        _f("v_pcloudy.jpg", "partly cloudy", "daytime"),
        _f("v_undef.jpg", "undefined", "daytime"),
    ]
    s = build_splits(train, val, seed=0, idval_fraction=0.1)
    assert list(s) == ["train", "id-val", "t-clean", "t-night", "t-rain", "t-snow", "t-fog"]
    assert len(s["id-val"]) == 2 and len(s["train"]) == 18
    assert not set(s["train"]) & set(s["id-val"])
    assert "tr_night.jpg" not in s["train"] + s["id-val"]
    assert s["t-clean"] == ["v_clean.jpg"]
    assert s["t-night"] == ["v_night_clear.jpg", "v_night_rain.jpg"]
    assert (s["t-rain"], s["t-snow"], s["t-fog"]) == (["v_rain.jpg"], ["v_snow.jpg"], ["v_fog.jpg"])
    used = {n for split in s.values() for n in split}
    for excluded in ("v_dusk.jpg", "v_overcast.jpg", "v_pcloudy.jpg", "v_undef.jpg"):
        assert excluded not in used


def test_build_splits_deterministic_per_seed():
    train = [_f(f"tr{i:03d}.jpg", "clear", "daytime") for i in range(200)]
    a = build_splits(train, [], seed=0, idval_fraction=0.1)
    b = build_splits(list(reversed(train)), [], seed=0, idval_fraction=0.1)
    c = build_splits(train, [], seed=1, idval_fraction=0.1)
    assert a == b
    assert a["id-val"] != c["id-val"]


def test_select_frames_keeps_manifest_order_and_reports_missing():
    frames = [_f("a.jpg", "clear", "daytime"), _f("b.jpg", "clear", "daytime")]
    assert [f.name for f in select_frames(frames, ["b.jpg", "a.jpg"])] == ["b.jpg", "a.jpg"]
    with pytest.raises(KeyError, match="1 manifest name"):
        select_frames(frames, ["a.jpg", "zzz.jpg"])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bdd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.data'`.

- [ ] **Step 3: Implement `src/snz/data/bdd.py`**

```python
"""BDD100K detection labels: loading, attribute filters, and study splits (spec §4.1-4.2)."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# det_20 category names. Older (2018) labels use person/bike/motor; ALIASES maps them.
CLASSES = (
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
)
EVAL_CLASSES = tuple(c for c in CLASSES if c != "train")  # `train` too rare for stable AP
ALIASES = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}
MIN_BOX_SIDE = 1.0  # px; torchvision rejects zero-area boxes

TRAIN_SPLITS = ("train", "id-val")
TEST_SPLITS = ("t-clean", "t-night", "t-rain", "t-snow", "t-fog")


@dataclass(frozen=True)
class Box:
    category: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Frame:
    name: str
    weather: str
    timeofday: str
    boxes: tuple[Box, ...]


@dataclass
class LoadStats:
    frames: int = 0
    boxes: int = 0
    dropped_unknown: dict[str, int] = field(default_factory=dict)
    dropped_degenerate: int = 0


def load_frames(path: Path | str) -> tuple[list[Frame], LoadStats]:
    with open(path) as f:
        raw = json.load(f)
    stats = LoadStats()
    frames: list[Frame] = []
    seen: set[str] = set()
    for item in raw:
        name = item["name"]
        if name in seen:
            raise ValueError(f"duplicate frame name in {path}: {name}")
        seen.add(name)
        attrs = item.get("attributes") or {}
        boxes = []
        for label in item.get("labels") or []:
            b = label.get("box2d")
            if b is None:
                continue  # non-box annotation
            category = ALIASES.get(label["category"], label["category"])
            if category not in CLASSES:
                stats.dropped_unknown[category] = stats.dropped_unknown.get(category, 0) + 1
                continue
            box = Box(category, float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
            if box.x2 - box.x1 < MIN_BOX_SIDE or box.y2 - box.y1 < MIN_BOX_SIDE:
                stats.dropped_degenerate += 1
                continue
            boxes.append(box)
        frames.append(Frame(
            name,
            attrs.get("weather", "undefined"),
            attrs.get("timeofday", "undefined"),
            tuple(boxes),
        ))
        stats.boxes += len(boxes)
    stats.frames = len(frames)
    return frames, stats


def is_clear_day(f: Frame) -> bool:
    return f.timeofday == "daytime" and f.weather == "clear"


def _day_with(weather: str) -> Callable[[Frame], bool]:
    return lambda f: f.timeofday == "daytime" and f.weather == weather


# Weather subsets are daytime-only, to separate weather from darkness (spec §4.2).
VAL_RULES: dict[str, Callable[[Frame], bool]] = {
    "t-clean": is_clear_day,
    "t-night": lambda f: f.timeofday == "night",
    "t-rain": _day_with("rainy"),
    "t-snow": _day_with("snowy"),
    "t-fog": _day_with("foggy"),
}


def build_splits(
    train_frames: Iterable[Frame],
    val_frames: Iterable[Frame],
    seed: int,
    idval_fraction: float,
) -> dict[str, list[str]]:
    pool = sorted(f.name for f in train_frames if is_clear_day(f))
    idval = set(random.Random(seed).sample(pool, round(len(pool) * idval_fraction)))
    splits = {"train": [n for n in pool if n not in idval], "id-val": sorted(idval)}
    val_frames = list(val_frames)
    for split, rule in VAL_RULES.items():
        splits[split] = sorted(f.name for f in val_frames if rule(f))
    return splits


def select_frames(frames: Iterable[Frame], names: Sequence[str]) -> list[Frame]:
    by_name = {f.name: f for f in frames}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"{len(missing)} manifest name(s) not in labels, e.g. {missing[:3]}")
    return [by_name[n] for n in names]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bdd.py -v && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/snz/data tests/test_bdd.py
git commit -m "snz: BDD label loading and study splits"
```

---

### Task 3: Hashing, manifests, and the split build

**Files:**
- Create: `src/snz/hashing.py`, `src/snz/data/manifest.py`, `src/snz/data/build.py`, `configs/splits.toml`
- Modify: `src/snz/cli.py` (add `build-splits`)
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `load_frames`, `build_splits`, `TRAIN_SPLITS`, `TEST_SPLITS` (Task 2).
- Produces:
  - `snz.hashing`: `sha256_file(path) -> str`, `verify_sha256(path, expected) -> None` (raises `ChecksumError`), `KNOWN_SHA256: dict[str, str]`, `class ChecksumError(ValueError)`
  - `snz.data.manifest`: `Row = tuple[str, str]` (name, sha256); `hash_rows(names, image_dir) -> list[Row]`; `write_manifest(path, rows)`; `read_manifest(path) -> list[Row]`; `drop_hash_duplicates(rows, forbidden_hashes) -> tuple[list[Row], list[Row]]` (kept, dropped); `check_disjoint(manifests: dict[str, list[Row]]) -> None` (raises `SplitOverlapError`)
  - `snz.data.build.build_and_write_splits(train_labels, val_labels, train_images, val_images, out_dir, seed, idval_fraction) -> dict` (summary also written to `out_dir/split_counts.json`)

- [ ] **Step 1: Write the failing tests**

`tests/test_manifest.py`:

```python
import hashlib
import json

import pytest
from helpers import frame_json, write_image, write_labels

from snz.cli import main
from snz.data.build import build_and_write_splits
from snz.data.manifest import (
    SplitOverlapError,
    check_disjoint,
    drop_hash_duplicates,
    hash_rows,
    read_manifest,
    write_manifest,
)
from snz.hashing import ChecksumError, sha256_file, verify_sha256


def test_sha256_and_verify(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == hashlib.sha256(b"hello").hexdigest()
    verify_sha256(p, hashlib.sha256(b"hello").hexdigest())
    with pytest.raises(ChecksumError, match="x.bin"):
        verify_sha256(p, "0" * 64)


def test_manifest_roundtrip_and_missing_image(tmp_path):
    write_image(tmp_path / "img" / "a.jpg")
    rows = hash_rows(["a.jpg"], tmp_path / "img")
    write_manifest(tmp_path / "m.csv", rows)
    assert read_manifest(tmp_path / "m.csv") == rows
    with pytest.raises(FileNotFoundError, match="b.jpg"):
        hash_rows(["b.jpg"], tmp_path / "img")


def test_check_disjoint_by_name_and_by_hash():
    check_disjoint({"train": [("a", "h1")], "t-clean": [("b", "h2")]})
    with pytest.raises(SplitOverlapError, match="name"):
        check_disjoint({"train": [("a", "h1")], "t-clean": [("a", "h2")]})
    with pytest.raises(SplitOverlapError, match="hash"):
        check_disjoint({"train": [("a", "h1")], "t-clean": [("b", "h1")]})


def test_drop_hash_duplicates():
    kept, dropped = drop_hash_duplicates([("a", "h1"), ("b", "h2")], {"h2"})
    assert kept == [("a", "h1")] and dropped == [("b", "h2")]


def test_idval_copies_of_train_images_are_dropped(tmp_path):
    train = [frame_json(f"tr{i}.jpg") for i in range(10)]
    for i in range(10):  # every image identical: id-val copies must all be dropped
        write_image(tmp_path / "train" / f"tr{i}.jpg", color=(50, 50, 50))
    write_image(tmp_path / "val" / "v.jpg", color=(0, 0, 200))
    out = tmp_path / "m"
    summary = build_and_write_splits(
        write_labels(tmp_path / "t.json", train), write_labels(tmp_path / "v.json",
                                                               [frame_json("v.jpg")]),
        tmp_path / "train", tmp_path / "val", out, seed=0, idval_fraction=0.2)
    assert summary["counts"]["id-val"] == 0 and summary["dropped_duplicates"] == 2


def _fixture(tmp_path):
    train = [frame_json(f"tr{i}.jpg", labels=[("car", 0, 0, 10, 10)]) for i in range(10)]
    val = [
        frame_json("v_clean.jpg"),
        frame_json("v_night.jpg", timeofday="night"),
        frame_json("v_rain.jpg", weather="rainy"),
    ]
    for i in range(10):  # colors far apart so JPEG bytes differ
        write_image(tmp_path / "train" / f"tr{i}.jpg", color=(20 * i, 0, 0))
    write_image(tmp_path / "val" / "v_clean.jpg", color=(0, 0, 200))
    write_image(tmp_path / "val" / "v_night.jpg", color=(0, 200, 0))
    # same pixels as tr3.jpg under another name: a cross-split duplicate
    write_image(tmp_path / "val" / "v_rain.jpg", color=(60, 0, 0))
    return (
        write_labels(tmp_path / "det_train.json", train),
        write_labels(tmp_path / "det_val.json", val),
    )


def test_build_and_write_splits_drops_cross_split_duplicates(tmp_path):
    tl, vl = _fixture(tmp_path)
    out = tmp_path / "manifests"
    summary = build_and_write_splits(
        tl, vl, tmp_path / "train", tmp_path / "val", out, seed=0, idval_fraction=0.2
    )
    train = [n for n, _ in read_manifest(out / "train.csv")]
    idval = [n for n, _ in read_manifest(out / "id-val.csv")]
    assert "tr3.jpg" not in train + idval
    assert [n for n, _ in read_manifest(out / "t-rain.csv")] == ["v_rain.jpg"]
    dropped = (out / "dropped_duplicates.csv").read_text()
    assert "tr3.jpg" in dropped and "v_rain.jpg" in dropped
    assert summary["counts"]["t-clean"] == 1
    assert json.loads((out / "split_counts.json").read_text()) == summary


def test_cli_build_splits(tmp_path):
    tl, vl = _fixture(tmp_path)
    cfg = tmp_path / "splits.toml"
    cfg.write_text("seed = 0\nidval_fraction = 0.2\n")
    code = main([
        "build-splits", "--train-labels", str(tl), "--val-labels", str(vl),
        "--train-images", str(tmp_path / "train"), "--val-images", str(tmp_path / "val"),
        "--out", str(tmp_path / "m"), "--config", str(cfg),
    ])
    assert code == 0
    assert (tmp_path / "m" / "t-night.csv").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.hashing'`.

- [ ] **Step 3: Implement `src/snz/hashing.py`**

```python
"""File checksums and the known-good weight hashes (README §5.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

KNOWN_SHA256 = {
    "resnet50_train_60_epochs-c8e5653e.pth.tar":
        "c8e5653e47bbf3f6efad5da531f3acea9963acb93362b6530cf39b2ba4750da3",
    "decoder.pth": "379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472",
    "vgg_normalised.pth": "804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a",
}


class ChecksumError(ValueError):
    pass


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: Path | str, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumError(f"{path}: expected sha256 {expected}, got {actual}")
```

- [ ] **Step 4: Implement `src/snz/data/manifest.py`**

```python
"""Split manifests (image name + SHA-256) and the no-image-in-two-splits check (spec §4.2)."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path

from snz.hashing import sha256_file

Row = tuple[str, str]  # (image name, sha256)


class SplitOverlapError(ValueError):
    pass


def hash_rows(names: Sequence[str], image_dir: Path) -> list[Row]:
    rows = []
    for name in names:
        path = Path(image_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"image listed in labels but missing on disk: {path}")
        rows.append((name, sha256_file(path)))
    return rows


def write_manifest(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "sha256"])
        w.writerows(rows)


def read_manifest(path: Path) -> list[Row]:
    with open(path, newline="") as f:
        return [(r["name"], r["sha256"]) for r in csv.DictReader(f)]


def drop_hash_duplicates(
    rows: Sequence[Row], forbidden_hashes: set[str]
) -> tuple[list[Row], list[Row]]:
    kept = [r for r in rows if r[1] not in forbidden_hashes]
    dropped = [r for r in rows if r[1] in forbidden_hashes]
    return kept, dropped


def check_disjoint(manifests: dict[str, Sequence[Row]]) -> None:
    problems = []
    for (a, rows_a), (b, rows_b) in combinations(manifests.items(), 2):
        for kind, idx in (("name", 0), ("hash", 1)):
            shared = {r[idx] for r in rows_a} & {r[idx] for r in rows_b}
            if shared:
                problems.append(f"{a} / {b}: {len(shared)} shared {kind}(s), e.g. {sorted(shared)[:3]}")
    if problems:
        raise SplitOverlapError("splits overlap:\n" + "\n".join(problems))
```

- [ ] **Step 5: Implement `src/snz/data/build.py`**

```python
"""Build all split manifests from BDD labels + images (gate G2)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from snz.data.bdd import TEST_SPLITS, TRAIN_SPLITS, LoadStats, build_splits, load_frames
from snz.data.manifest import check_disjoint, drop_hash_duplicates, hash_rows, write_manifest


def _stats(s: LoadStats) -> dict:
    return {"frames": s.frames, "boxes": s.boxes, "dropped_unknown": s.dropped_unknown,
            "dropped_degenerate": s.dropped_degenerate}


def build_and_write_splits(
    train_labels: Path,
    val_labels: Path,
    train_images: Path,
    val_images: Path,
    out_dir: Path,
    seed: int,
    idval_fraction: float,
) -> dict:
    out_dir = Path(out_dir)
    train_frames, train_stats = load_frames(train_labels)
    val_frames, val_stats = load_frames(val_labels)
    names = build_splits(train_frames, val_frames, seed, idval_fraction)

    rows = {s: hash_rows(names[s], val_images) for s in TEST_SPLITS}
    forbidden = {h for split_rows in rows.values() for _, h in split_rows}
    dropped_log = []
    # train first: its hashes are then also forbidden in id-val
    for s in TRAIN_SPLITS:
        kept, dropped = drop_hash_duplicates(hash_rows(names[s], train_images), forbidden)
        rows[s] = kept
        dropped_log += [(s, name, h) for name, h in dropped]
        forbidden |= {h for _, h in kept}
    rows = {s: rows[s] for s in (*TRAIN_SPLITS, *TEST_SPLITS)}
    check_disjoint(rows)

    for split, split_rows in rows.items():
        write_manifest(out_dir / f"{split}.csv", split_rows)
    by_hash = {h: (s, n) for s in (*TEST_SPLITS, "train") for n, h in rows[s]}
    with open(out_dir / "dropped_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "name", "sha256", "duplicate_of_split", "duplicate_of_name"])
        w.writerows((s, n, h, *by_hash[h]) for s, n, h in dropped_log)

    summary = {
        "seed": seed,
        "idval_fraction": idval_fraction,
        "counts": {s: len(r) for s, r in rows.items()},
        "dropped_duplicates": len(dropped_log),
        "labels": {"train": _stats(train_stats), "val": _stats(val_stats)},
    }
    (out_dir / "split_counts.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
```

`configs/splits.toml`:

```toml
# Split construction (spec §4.2). Changing these changes every manifest.
seed = 0
idval_fraction = 0.1
```

- [ ] **Step 6: Add the `build-splits` command to `src/snz/cli.py`**

Add at top: `import json`, `import tomllib`, `from pathlib import Path`. Inside `build_parser`, before `return parser` (and drop the `# noqa` on `commands`):

```python
    p = commands.add_parser("build-splits", help="write split manifests from BDD labels (G2)")
    p.add_argument("--train-labels", type=Path, required=True)
    p.add_argument("--val-labels", type=Path, required=True)
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--val-images", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("manifests"))
    p.add_argument("--config", type=Path, default=Path("configs/splits.toml"))
    p.set_defaults(func=_cmd_build_splits)
```

And module-level:

```python
def _cmd_build_splits(args) -> int:
    from snz.data.build import build_and_write_splits

    cfg = tomllib.loads(args.config.read_text())
    summary = build_and_write_splits(
        args.train_labels, args.val_labels, args.train_images, args.val_images,
        args.out, seed=cfg["seed"], idval_fraction=cfg["idval_fraction"],
    )
    print(json.dumps(summary["counts"], indent=2))
    return 0
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_manifest.py tests/test_cli.py -v && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/snz configs/splits.toml tests/test_manifest.py
git commit -m "snz: split manifests with hash-based overlap check; build-splits command"
```

---

### Task 4: AdaIN style transfer and the style pool

**Files:**
- Create: `src/snz/stylize/__init__.py` (empty), `src/snz/stylize/adain.py`, `src/snz/stylize/pool.py`
- Test: `tests/test_adain.py`

**Interfaces:**
- Consumes: `verify_sha256`, `KNOWN_SHA256` (Task 3).
- Produces:
  - `make_vgg() -> nn.Sequential` (full VGG-19 layout matching `vgg_normalised.pth`), `make_decoder() -> nn.Sequential`, `ENCODER_LAYERS = 31`
  - `adain(content_feat, style_feat, eps=1e-5) -> Tensor`
  - `stylize_tensor(content, style, encoder, decoder, alpha=1.0) -> Tensor` (same H×W as `content`, clamped to [0, 1])
  - `class Stylizer(encoder, decoder, device="cpu", style_size=512, alpha=1.0)`, callable `(content: PIL.Image, style: PIL.Image) -> PIL.Image` (same size as content); `Stylizer.from_weights(vgg_path, decoder_path, device, style_size=512, alpha=1.0)` verifies checksums
  - `list_styles(root) -> list[str]` (sorted relative POSIX paths of .jpg/.jpeg/.png), `split_style_pool(paths, seed, test_fraction=0.2) -> tuple[list[str], list[str]]` (train, test; disjoint, sorted)

- [ ] **Step 1: Write the failing tests**

`tests/test_adain.py`:

```python
import torch
from helpers import write_image
from PIL import Image
from torch import nn

from snz.stylize.adain import (
    ENCODER_LAYERS,
    Stylizer,
    adain,
    make_decoder,
    make_vgg,
    stylize_tensor,
)
from snz.stylize.pool import list_styles, split_style_pool


def test_adain_matches_style_statistics():
    g = torch.Generator().manual_seed(0)
    content = torch.randn(1, 4, 16, 16, generator=g) * 3 + 1
    style = torch.randn(1, 4, 16, 16, generator=g) * 0.5 - 2
    out = adain(content, style)
    flat_out, flat_style = out.reshape(4, -1), style.reshape(4, -1)
    torch.testing.assert_close(flat_out.mean(1), flat_style.mean(1), atol=1e-4, rtol=0)
    torch.testing.assert_close(flat_out.std(1), flat_style.std(1), atol=1e-3, rtol=0)


def test_real_architecture_shapes():
    encoder = make_vgg()[:ENCODER_LAYERS]
    decoder = make_decoder()
    x = torch.rand(1, 3, 64, 64)
    feat = encoder(x)
    assert feat.shape == (1, 512, 8, 8)
    assert decoder(feat).shape == (1, 3, 64, 64)


def _stub():
    encoder = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8, padding=1))
    decoder = nn.Sequential(nn.Upsample(scale_factor=8), nn.Conv2d(4, 3, 1))
    return encoder, decoder


def test_stylize_tensor_preserves_odd_sizes():
    encoder, decoder = _stub()
    out = stylize_tensor(torch.rand(1, 3, 23, 37), torch.rand(1, 3, 50, 50), encoder, decoder)
    assert out.shape == (1, 3, 23, 37)
    assert out.min() >= 0 and out.max() <= 1


def test_stylizer_returns_same_size_rgb_image():
    encoder, decoder = _stub()
    stylizer = Stylizer(encoder, decoder, device="cpu", style_size=32)
    result = stylizer(Image.new("RGB", (37, 23)), Image.new("RGB", (80, 60), (10, 200, 30)))
    assert result.size == (37, 23) and result.mode == "RGB"


def test_style_pool_split(tmp_path):
    for cat in ("banded", "dotted"):
        for i in range(10):
            write_image(tmp_path / cat / f"{cat}_{i:02d}.jpg")
    (tmp_path / "readme.txt").write_text("not an image")
    styles = list_styles(tmp_path)
    assert len(styles) == 20 and styles[0] == "banded/banded_00.jpg"
    train, test = split_style_pool(styles, seed=0, test_fraction=0.2)
    assert len(test) == 4 and len(train) == 16
    assert not set(train) & set(test)
    assert split_style_pool(list(reversed(styles)), seed=0) == (train, test)
    assert split_style_pool(styles, seed=1) != (train, test)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_adain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.stylize'`.

- [ ] **Step 3: Implement `src/snz/stylize/adain.py`**

```python
"""AdaIN style transfer (Huang & Belongie, 2017).

Network layout matches naoto0804/pytorch-AdaIN, so its released `vgg_normalised.pth`
and `decoder.pth` load directly. Inputs are RGB tensors in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor, nn

from snz.hashing import KNOWN_SHA256, verify_sha256

ENCODER_LAYERS = 31  # vgg children up to relu4_1


def _conv(cin: int, cout: int) -> list[nn.Module]:
    return [nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(cin, cout, (3, 3)), nn.ReLU()]


def _pool() -> nn.Module:
    return nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)


def make_vgg() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        *_conv(3, 64), *_conv(64, 64), _pool(),
        *_conv(64, 128), *_conv(128, 128), _pool(),
        *_conv(128, 256), *_conv(256, 256), *_conv(256, 256), *_conv(256, 256), _pool(),
        *_conv(256, 512),  # relu4_1 = child 30
        *_conv(512, 512), *_conv(512, 512), *_conv(512, 512), _pool(),
        *_conv(512, 512), *_conv(512, 512), *_conv(512, 512), *_conv(512, 512),
    )


def make_decoder() -> nn.Sequential:
    def up() -> nn.Module:
        return nn.Upsample(scale_factor=2, mode="nearest")

    return nn.Sequential(
        *_conv(512, 256), up(),
        *_conv(256, 256), *_conv(256, 256), *_conv(256, 256), *_conv(256, 128), up(),
        *_conv(128, 128), *_conv(128, 64), up(),
        *_conv(64, 64),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
    )


def _mean_std(feat: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    n, c = feat.shape[:2]
    flat = feat.reshape(n, c, -1)
    mean = flat.mean(dim=2).reshape(n, c, 1, 1)
    std = (flat.var(dim=2) + eps).sqrt().reshape(n, c, 1, 1)
    return mean, std


def adain(content_feat: Tensor, style_feat: Tensor, eps: float = 1e-5) -> Tensor:
    c_mean, c_std = _mean_std(content_feat, eps)
    s_mean, s_std = _mean_std(style_feat, eps)
    return (content_feat - c_mean) / c_std * s_std + s_mean


@torch.no_grad()
def stylize_tensor(
    content: Tensor, style: Tensor, encoder: nn.Module, decoder: nn.Module, alpha: float = 1.0
) -> Tensor:
    f_c = encoder(content)
    t = adain(f_c, encoder(style))
    t = alpha * t + (1 - alpha) * f_c
    out = decoder(t)
    if out.shape[-2:] != content.shape[-2:]:
        out = F.interpolate(out, size=content.shape[-2:], mode="bilinear", align_corners=False)
    return out.clamp(0, 1)


class Stylizer:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: str = "cpu",
        style_size: int = 512,
        alpha: float = 1.0,
    ):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.device = device
        self.style_size = style_size
        self.alpha = alpha

    @classmethod
    def from_weights(
        cls, vgg_path: Path, decoder_path: Path, device: str = "cuda",
        style_size: int = 512, alpha: float = 1.0,
    ) -> Stylizer:
        verify_sha256(vgg_path, KNOWN_SHA256["vgg_normalised.pth"])
        verify_sha256(decoder_path, KNOWN_SHA256["decoder.pth"])
        vgg = make_vgg()
        vgg.load_state_dict(torch.load(vgg_path, map_location="cpu", weights_only=True))
        decoder = make_decoder()
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
        return cls(vgg[:ENCODER_LAYERS], decoder, device, style_size, alpha)

    def __call__(self, content: Image.Image, style: Image.Image) -> Image.Image:
        c = TF.to_tensor(content.convert("RGB")).unsqueeze(0).to(self.device)
        s = TF.resize(style.convert("RGB"), self.style_size)
        s = TF.to_tensor(s).unsqueeze(0).to(self.device)
        out = stylize_tensor(c, s, self.encoder, self.decoder, self.alpha)
        return TF.to_pil_image(out[0].float().cpu())
```

- [ ] **Step 4: Implement `src/snz/stylize/pool.py`**

```python
"""Style image pool and its fixed train/test split (spec §4.3)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

STYLE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_styles(root: Path) -> list[str]:
    root = Path(root)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in STYLE_SUFFIXES
    )


def split_style_pool(
    paths: Sequence[str], seed: int, test_fraction: float = 0.2
) -> tuple[list[str], list[str]]:
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    test, train = sorted(shuffled[:n_test]), sorted(shuffled[n_test:])
    if not train or not test:
        raise ValueError(f"style pool of {len(paths)} images is too small to split")
    return train, test
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_adain.py -v && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/snz/stylize tests/test_adain.py
git commit -m "snz: AdaIN stylizer and style-pool split"
```

---

### Task 5: Precomputed stylized copies and T-texture

**Files:**
- Create: `src/snz/stylize/build.py`, `configs/stylize.toml`
- Modify: `src/snz/cli.py` (add `stylize`)
- Test: `tests/test_stylize_build.py`

**Interfaces:**
- Consumes: `Stylizer`, `list_styles`, `split_style_pool` (Task 4); `read_manifest`, `hash_rows`, `write_manifest` (Task 3).
- Produces:
  - `StylizeConfig(seed, test_fraction, train_copies, alpha, style_size)`; `load_stylize_config(path) -> StylizeConfig`
  - `style_for(name, copy, styles, seed) -> str` (deterministic, independent of order and subset)
  - `stylized_name(name, copy) -> str` → `"<stem>_s<copy>.jpg"`
  - `build_stylized(names, image_dir, style_root, styles, out_dir, copies, seed, stylizer) -> list[dict]` (rows: name, copy, style, file); skips existing outputs; atomic writes
  - `run_stylize(set_name: "train" | "t-texture", manifests_dir, image_dir, style_root, out_dir, cfg, stylizer) -> list[dict]`; writes `manifests/styles_train.csv`, `manifests/styles_test.csv`, `manifests/stylized_<set>.csv`. `train` reads `manifests/train.csv`, copies 1..`train_copies`, training styles. `t-texture` reads `manifests/t-clean.csv`, copy 1, test styles.

- [ ] **Step 1: Write the failing tests**

`tests/test_stylize_build.py`:

```python
import csv

import pytest
from helpers import write_image
from PIL import Image

from snz.data.manifest import write_manifest
from snz.stylize.build import (
    StylizeConfig,
    build_stylized,
    run_stylize,
    style_for,
    stylized_name,
)

STYLES = [f"s{i}.jpg" for i in range(10)]


class CountingStylizer:
    """Returns the style image resized to the content size, and counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, content, style):
        self.calls += 1
        return style.resize(content.size)


def test_style_for_is_order_and_subset_independent():
    a = {(n, k): style_for(n, k, STYLES, seed=0) for n in ("x.jpg", "y.jpg") for k in (1, 2)}
    assert style_for("y.jpg", 2, STYLES, seed=0) == a[("y.jpg", 2)]
    assert len({style_for(f"{i}.jpg", 1, STYLES, 0) for i in range(100)}) > 5
    assert any(style_for(f"{i}.jpg", 1, STYLES, 0) != style_for(f"{i}.jpg", 1, STYLES, 1)
               for i in range(20))


def test_stylized_name():
    assert stylized_name("abc-123.jpg", 2) == "abc-123_s2.jpg"


def _setup(tmp_path):
    for n in ("a.jpg", "b.jpg"):
        write_image(tmp_path / "img" / n, size=(40, 30))
    for i, s in enumerate(STYLES):
        write_image(tmp_path / "styles" / s, size=(20, 20), color=(i * 20, 0, 0))


def test_build_stylized_writes_resumes_and_keeps_size(tmp_path):
    _setup(tmp_path)
    st = CountingStylizer()
    kw = dict(image_dir=tmp_path / "img", style_root=tmp_path / "styles", styles=STYLES,
              out_dir=tmp_path / "out", copies=(1, 2), seed=0)
    rows = build_stylized(["a.jpg", "b.jpg"], stylizer=st, **kw)
    assert st.calls == 4 and len(rows) == 4
    assert Image.open(tmp_path / "out" / "a_s1.jpg").size == (40, 30)
    assert not list((tmp_path / "out").glob("*.tmp"))
    again = build_stylized(["b.jpg", "a.jpg"], stylizer=st, **kw)
    assert st.calls == 4  # everything existed; nothing recomputed
    key = lambda r: (r["name"], r["copy"])  # noqa: E731
    assert sorted(again, key=key) == sorted(rows, key=key)


def test_build_stylized_rejects_size_change(tmp_path):
    _setup(tmp_path)
    with pytest.raises(ValueError, match="size"):
        build_stylized(["a.jpg"], tmp_path / "img", tmp_path / "styles", STYLES,
                       tmp_path / "out", (1,), 0, lambda c, s: s)


def test_run_stylize_uses_disjoint_pools(tmp_path):
    _setup(tmp_path)
    m = tmp_path / "manifests"
    write_manifest(m / "train.csv", [("a.jpg", "h")])
    write_manifest(m / "t-clean.csv", [("b.jpg", "h")])
    cfg = StylizeConfig(seed=0, test_fraction=0.2, train_copies=2, alpha=1.0, style_size=16)
    tr = run_stylize("train", m, tmp_path / "img", tmp_path / "styles", tmp_path / "o1", cfg,
                     CountingStylizer())
    te = run_stylize("t-texture", m, tmp_path / "img", tmp_path / "styles", tmp_path / "o2",
                     cfg, CountingStylizer())
    with open(m / "styles_test.csv") as f:
        test_styles = {r["name"] for r in csv.DictReader(f)}
    assert {r["copy"] for r in tr} == {1, 2} and {r["copy"] for r in te} == {1}
    assert all(r["style"] not in test_styles for r in tr)
    assert all(r["style"] in test_styles for r in te)
    assert (m / "stylized_train.csv").exists() and (m / "stylized_t-texture.csv").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_stylize_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.stylize.build'`.

- [ ] **Step 3: Implement `src/snz/stylize/build.py`**

```python
"""Precompute stylized training copies and the T-texture test set (spec §4.3).

Style choice is a pure function of (seed, image name, copy), so an interrupted or
partial build resumes to exactly the same result.
"""

from __future__ import annotations

import csv
import hashlib
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from snz.data.manifest import hash_rows, read_manifest, write_manifest
from snz.stylize.pool import list_styles, split_style_pool

StylizeFn = Callable[[Image.Image, Image.Image], Image.Image]


@dataclass(frozen=True)
class StylizeConfig:
    seed: int
    test_fraction: float
    train_copies: int
    alpha: float
    style_size: int


def load_stylize_config(path: Path) -> StylizeConfig:
    return StylizeConfig(**tomllib.loads(Path(path).read_text()))


def style_for(name: str, copy: int, styles: Sequence[str], seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{name}:{copy}".encode()).digest()
    return styles[int.from_bytes(digest[:8], "big") % len(styles)]


def stylized_name(name: str, copy: int) -> str:
    return f"{Path(name).stem}_s{copy}.jpg"


def build_stylized(
    names: Sequence[str],
    image_dir: Path,
    style_root: Path,
    styles: Sequence[str],
    out_dir: Path,
    copies: Sequence[int],
    seed: int,
    stylizer: StylizeFn,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        for copy in copies:
            style = style_for(name, copy, styles, seed)
            out = out_dir / stylized_name(name, copy)
            if not out.exists():
                content = Image.open(Path(image_dir) / name).convert("RGB")
                result = stylizer(content, Image.open(Path(style_root) / style).convert("RGB"))
                if result.size != content.size:
                    raise ValueError(
                        f"stylizer changed size of {name}: {content.size} -> {result.size}")
                tmp = out.with_name(out.name + ".tmp")
                result.save(tmp, format="JPEG", quality=95)
                tmp.replace(out)
            rows.append({"name": name, "copy": copy, "style": style, "file": out.name})
    return rows


def run_stylize(
    set_name: str,
    manifests_dir: Path,
    image_dir: Path,
    style_root: Path,
    out_dir: Path,
    cfg: StylizeConfig,
    stylizer: StylizeFn,
) -> list[dict]:
    manifests_dir = Path(manifests_dir)
    train_styles, test_styles = split_style_pool(
        list_styles(style_root), cfg.seed, cfg.test_fraction)
    write_manifest(manifests_dir / "styles_train.csv", hash_rows(train_styles, style_root))
    write_manifest(manifests_dir / "styles_test.csv", hash_rows(test_styles, style_root))
    if set_name == "train":
        source, copies, styles = "train.csv", range(1, cfg.train_copies + 1), train_styles
    elif set_name == "t-texture":
        source, copies, styles = "t-clean.csv", (1,), test_styles
    else:
        raise ValueError(f"unknown stylized set {set_name!r}; expected 'train' or 't-texture'")
    names = [n for n, _ in read_manifest(manifests_dir / source)]
    rows = build_stylized(names, image_dir, style_root, styles, out_dir, tuple(copies), cfg.seed,
                          stylizer)
    with open(manifests_dir / f"stylized_{set_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "copy", "style", "file"])
        w.writeheader()
        w.writerows(rows)
    return rows
```

`configs/stylize.toml`:

```toml
# Stylization (spec §4.3). Style images: Describable Textures Dataset (README §5.1).
seed = 0
test_fraction = 0.2   # held-out styles for T-texture
train_copies = 2      # stylized copies per training image
alpha = 1.0           # full stylization, as in Stylized-ImageNet
style_size = 512      # style image shorter side, px
```

- [ ] **Step 4: Add the `stylize` command to `src/snz/cli.py`**

Inside `build_parser`:

```python
    p = commands.add_parser("stylize", help="precompute stylized copies (G2)")
    p.add_argument("--set", choices=["train", "t-texture"], required=True)
    p.add_argument("--manifests", type=Path, default=Path("manifests"))
    p.add_argument("--images", type=Path, required=True, help="BDD train (for train) or val dir")
    p.add_argument("--styles-root", type=Path, required=True, help="e.g. dtd/images")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vgg", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/stylize.toml"))
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_stylize)
```

Module level:

```python
def _cmd_stylize(args) -> int:
    from snz.stylize.adain import Stylizer
    from snz.stylize.build import load_stylize_config, run_stylize

    cfg = load_stylize_config(args.config)
    stylizer = Stylizer.from_weights(args.vgg, args.decoder, args.device, cfg.style_size,
                                     cfg.alpha)
    rows = run_stylize(args.set, args.manifests, args.images, args.styles_root, args.out, cfg,
                       stylizer)
    print(f"{len(rows)} stylized images in {args.out}")
    return 0
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_stylize_build.py -v && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/snz configs/stylize.toml tests/test_stylize_build.py
git commit -m "snz: resumable stylized-copy and T-texture build; stylize command"
```

---

### Task 6: Detector model with verified backbones

**Files:**
- Create: `src/snz/train/__init__.py` (empty), `src/snz/train/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `CLASSES` (Task 2); `verify_sha256`, `KNOWN_SHA256`, `ChecksumError` (Task 3).
- Produces:
  - `NUM_CLASSES = 11`, `BACKBONES = ("in1k_v1", "sin")`, `SIN_FILENAME`
  - `load_backbone_state(kind, sin_path=None, expected_sha256=None) -> dict[str, Tensor]` (`module.` prefix stripped)
  - `build_model(backbone_state: Mapping[str, Tensor] | None, *, trainable_layers=3, min_size=720, max_size=1280, **kwargs) -> FasterRCNN` (`None` = random backbone, for loading a full checkpoint afterwards)

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:

```python
import pytest
import torch
from torch import nn
from torchvision.models import resnet50

from snz.hashing import ChecksumError, sha256_file
from snz.train.model import NUM_CLASSES, build_model, load_backbone_state


def _state(seed):
    torch.manual_seed(seed)
    return resnet50(weights=None).state_dict()


def _trainable(model):
    return [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]


def test_both_backbones_build_identical_architectures():
    a, b = build_model(_state(0)), build_model(_state(1))
    assert _trainable(a) == _trainable(b)
    assert a.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES == 11


def test_backbone_weights_loaded_and_frozen_as_specified():
    state = _state(0)
    model = build_model(state)
    body = model.backbone.body
    torch.testing.assert_close(body.conv1.weight, state["conv1.weight"])
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert not body.conv1.weight.requires_grad
    assert not any(p.requires_grad for p in body.layer1.parameters())
    for layer in (body.layer2, body.layer3, body.layer4):
        assert all(p.requires_grad for p in layer.parameters())


def test_sin_checkpoint_loading(tmp_path):
    state = _state(0)
    path = tmp_path / "sin.pth.tar"
    torch.save({"epoch": 60, "arch": "resnet50",
                "state_dict": {f"module.{k}": v for k, v in state.items()}}, path)
    loaded = load_backbone_state("sin", path, expected_sha256=sha256_file(path))
    assert set(loaded) == set(state)
    with pytest.raises(ChecksumError):
        load_backbone_state("sin", path, expected_sha256="0" * 64)


def test_bad_inputs():
    with pytest.raises(ValueError, match="sin_path"):
        load_backbone_state("sin")
    with pytest.raises(ValueError, match="unknown backbone"):
        load_backbone_state("vit")


def test_forward_on_cpu():
    model = build_model(_state(0), min_size=48, max_size=64).eval()
    with torch.no_grad():
        out = model([torch.rand(3, 48, 64)])
    assert set(out[0]) == {"boxes", "labels", "scores"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.train'`.

- [ ] **Step 3: Implement `src/snz/train/model.py`**

```python
"""Faster R-CNN ResNet-50 + FPN with a chosen backbone initialization (spec §5.1-5.2).

The detector is assembled by hand rather than with fasterrcnn_resnet50_fpn(weights=None),
because that path switches to trainable BatchNorm and all-layers-trainable when no
pretrained weights are passed. Here the backbone always has frozen BN and 3 trainable
stages, and FPN/RPN/ROI heads are freshly initialized (no COCO weights).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.misc import FrozenBatchNorm2d

from snz.data.bdd import CLASSES
from snz.hashing import KNOWN_SHA256, verify_sha256

NUM_CLASSES = len(CLASSES) + 1  # + background
BACKBONES = ("in1k_v1", "sin")
SIN_FILENAME = "resnet50_train_60_epochs-c8e5653e.pth.tar"


def load_backbone_state(
    kind: str, sin_path: Path | None = None, expected_sha256: str | None = None
) -> dict[str, Tensor]:
    if kind == "in1k_v1":
        return ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
    if kind == "sin":
        if sin_path is None:
            raise ValueError("backbone 'sin' needs sin_path (Stylized-ImageNet ResNet-50)")
        verify_sha256(sin_path, expected_sha256 or KNOWN_SHA256[SIN_FILENAME])
        ckpt = torch.load(sin_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        return {k.removeprefix("module."): v for k, v in state.items()}
    raise ValueError(f"unknown backbone {kind!r}; expected one of {BACKBONES}")


def build_model(
    backbone_state: Mapping[str, Tensor] | None,
    *,
    trainable_layers: int = 3,
    min_size: int | tuple[int, ...] = 720,
    max_size: int = 1280,
    **kwargs,
) -> FasterRCNN:
    body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    if backbone_state is not None:
        body.load_state_dict(backbone_state)
    backbone = _resnet_fpn_extractor(body, trainable_layers)
    return FasterRCNN(backbone, num_classes=NUM_CLASSES, min_size=min_size, max_size=max_size,
                      **kwargs)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_model.py -v && uv run ruff check src tests`
Expected: all pass. If `torch.load(..., weights_only=True)` later rejects the real SIN checkpoint at G1, switch that one call to `weights_only=False` (the file is checksum-verified first) and say so in the commit.

- [ ] **Step 5: Commit**

```bash
git add src/snz/train tests/test_model.py
git commit -m "snz: Faster R-CNN R50-FPN with checksum-verified in1k/SIN backbones"
```

---

### Task 7: Training dataset with stylized-copy mixing

**Files:**
- Create: `src/snz/train/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `Frame`, `Box`, `CLASSES` (Task 2); `stylized_name` (Task 5).
- Produces:
  - `LABEL_INDEX: dict[str, int]` (class → 1..10)
  - `choose_variant(name, seed, epoch, p_original=0.5, n_copies=2) -> int` (0 = original, k = copy k)
  - `frame_target(frame) -> dict[str, Tensor]` (`boxes` float32 [N,4] xyxy, `labels` int64 [N])
  - `BDDDetection(frames, image_dir, *, stylized_dir=None, p_original=0.5, n_copies=2, seed=0, train=True, hflip_p=0.5)`; `__getitem__(key: int | tuple[int, int])` where a tuple is `(index, epoch)`; returns `(image_tensor[3,H,W] float in [0,1], target)`
  - `collate(batch) -> tuple`

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset.py`:

```python
from collections import Counter

import pytest
import torch
from helpers import write_image

from snz.data.bdd import Box, Frame
from snz.train.dataset import LABEL_INDEX, BDDDetection, choose_variant, frame_target


def test_choose_variant_probabilities_and_determinism():
    names = [f"img{i}.jpg" for i in range(20000)]
    counts = Counter(choose_variant(n, seed=0, epoch=0) for n in names)
    assert abs(counts[0] / 20000 - 0.5) < 0.015
    assert abs(counts[1] / 20000 - 0.25) < 0.015
    assert abs(counts[2] / 20000 - 0.25) < 0.015
    assert choose_variant("a.jpg", 0, 3) == choose_variant("a.jpg", 0, 3)
    assert any(choose_variant(n, 0, 0) != choose_variant(n, 0, 1) for n in names[:50])
    assert any(choose_variant(n, 0, 0) != choose_variant(n, 1, 0) for n in names[:50])


def test_frame_target_including_empty():
    f = Frame("a.jpg", "clear", "daytime",
              (Box("car", 1, 2, 30, 40), Box("traffic sign", 0, 0, 5, 5)))
    t = frame_target(f)
    assert t["boxes"].tolist() == [[1, 2, 30, 40], [0, 0, 5, 5]]
    assert t["labels"].tolist() == [LABEL_INDEX["car"], LABEL_INDEX["traffic sign"]]
    empty = frame_target(Frame("b.jpg", "clear", "daytime", ()))
    assert empty["boxes"].shape == (0, 4) and empty["labels"].dtype == torch.int64


def _frames(tmp_path, n=40):
    frames = []
    for i in range(n):
        write_image(tmp_path / "img" / f"f{i}.jpg", size=(64, 48), color=(0, 0, 0))
        write_image(tmp_path / "sty" / f"f{i}_s1.jpg", size=(64, 48), color=(255, 255, 255))
        write_image(tmp_path / "sty" / f"f{i}_s2.jpg", size=(64, 48), color=(255, 255, 255))
        frames.append(Frame(f"f{i}.jpg", "clear", "daytime", (Box("car", 10, 5, 20, 15),)))
    return frames


def test_dataset_reads_stylized_copies_when_chosen(tmp_path):
    frames = _frames(tmp_path)
    ds = BDDDetection(frames, tmp_path / "img", stylized_dir=tmp_path / "sty", seed=0,
                      hflip_p=0.0)
    for i, f in enumerate(frames):
        image, _ = ds[(i, 0)]
        is_stylized = bool(image.mean() > 0.5)
        assert is_stylized == (choose_variant(f.name, 0, 0) != 0)
    plain = BDDDetection(frames, tmp_path / "img", seed=0, hflip_p=0.0)
    assert all(plain[(i, 0)][0].mean() < 0.5 for i in range(len(frames)))


def test_missing_stylized_copy_is_an_error(tmp_path):
    frames = _frames(tmp_path, n=40)
    for p in (tmp_path / "sty").glob("*_s2.jpg"):
        p.unlink()
    ds = BDDDetection(frames, tmp_path / "img", stylized_dir=tmp_path / "sty", seed=0)
    with pytest.raises(FileNotFoundError, match="stylized"):
        for i in range(len(frames)):
            ds[(i, 0)]


def test_hflip_mirrors_boxes(tmp_path):
    frames = _frames(tmp_path, n=1)
    ds = BDDDetection(frames, tmp_path / "img", seed=0, hflip_p=1.0)
    _, target = ds[0]
    assert target["boxes"].tolist() == [[44, 5, 54, 15]]  # width 64: x1' = 64-20, x2' = 64-10
    eval_ds = BDDDetection(frames, tmp_path / "img", seed=0, train=False, hflip_p=1.0)
    assert eval_ds[0][1]["boxes"].tolist() == [[10, 5, 20, 15]]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.train.dataset'`.

- [ ] **Step 3: Implement `src/snz/train/dataset.py`**

```python
"""BDD detection dataset with original/stylized mixing (spec §4.3, §5.3).

Every random choice (which copy, whether to flip) is a hash of (seed, epoch, name),
so data is identical across worker counts and across a resumed run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from snz.data.bdd import CLASSES, Frame
from snz.stylize.build import stylized_name

LABEL_INDEX = {c: i + 1 for i, c in enumerate(CLASSES)}


def _uniform(*parts) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_variant(
    name: str, seed: int, epoch: int, p_original: float = 0.5, n_copies: int = 2
) -> int:
    u = _uniform(seed, epoch, name, "variant")
    if u < p_original:
        return 0
    return 1 + min(int((u - p_original) / (1 - p_original) * n_copies), n_copies - 1)


def frame_target(frame: Frame) -> dict[str, Tensor]:
    boxes = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in frame.boxes], dtype=torch.float32)
    labels = torch.tensor([LABEL_INDEX[b.category] for b in frame.boxes], dtype=torch.int64)
    return {"boxes": boxes.reshape(-1, 4), "labels": labels}


class BDDDetection(Dataset):
    def __init__(
        self,
        frames: Sequence[Frame],
        image_dir: Path,
        *,
        stylized_dir: Path | None = None,
        p_original: float = 0.5,
        n_copies: int = 2,
        seed: int = 0,
        train: bool = True,
        hflip_p: float = 0.5,
    ):
        self.frames = list(frames)
        self.image_dir = Path(image_dir)
        self.stylized_dir = Path(stylized_dir) if stylized_dir is not None else None
        self.p_original = p_original
        self.n_copies = n_copies
        self.seed = seed
        self.train = train
        self.hflip_p = hflip_p

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, key: int | tuple[int, int]) -> tuple[Tensor, dict[str, Tensor]]:
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        frame = self.frames[index]
        variant = 0
        if self.train and self.stylized_dir is not None:
            variant = choose_variant(frame.name, self.seed, epoch, self.p_original, self.n_copies)
        if variant == 0:
            path = self.image_dir / frame.name
        else:
            path = self.stylized_dir / stylized_name(frame.name, variant)
            if not path.exists():
                raise FileNotFoundError(f"missing stylized copy {path} (run `snz stylize`)")
        image = TF.to_tensor(Image.open(path).convert("RGB"))
        target = frame_target(frame)
        if self.train and _uniform(self.seed, epoch, frame.name, "flip") < self.hflip_p:
            width = image.shape[-1]
            image = image.flip(-1)
            boxes = target["boxes"].clone()
            boxes[:, 0] = width - target["boxes"][:, 2]
            boxes[:, 2] = width - target["boxes"][:, 0]
            target["boxes"] = boxes
        return image, target


def collate(batch):
    return tuple(zip(*batch, strict=True))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_dataset.py -v && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/snz/train/dataset.py tests/test_dataset.py
git commit -m "snz: BDD detection dataset with deterministic stylized-copy mixing"
```

---

### Task 8: Recipe, arms, and the resumable training loop

**Files:**
- Create: `src/snz/train/config.py`, `src/snz/train/train.py`, `configs/recipe.toml`, `configs/arms.toml`
- Modify: `src/snz/cli.py` (add `train`)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `build_model`, `load_backbone_state` (Task 6); `BDDDetection`, `collate` (Task 7); `load_frames`, `select_frames` (Task 2); `read_manifest` (Task 3); `load_stylize_config` (Task 5).
- Produces:
  - `Recipe` (frozen dataclass; fields exactly the keys of `configs/recipe.toml`), `load_recipe(path) -> Recipe` (rejects missing/unknown keys), `recipe_sha256(recipe) -> str`
  - `Arm(name, backbone, stylized)`, `load_arms(path) -> dict[str, Arm]`
  - `SeededStream(n, seed, start=0)`: infinite sampler yielding `(index, epoch)`
  - `lr_factor(iteration, warmup_iters, step_iters) -> float`
  - `train_run(*, arm, seed, recipe, frames, image_dir, stylized_dir, n_copies, backbone_state, out_dir, device, stop_at=None, pilot=False) -> dict`; writes `out_dir/{last.pt, final.pt, log.jsonl, run.json}`; resumes from `last.pt`. Returns the `run.json` contents, or `{"stopped_at": n}` when `stop_at` ended it early.

- [ ] **Step 1: Write the configs**

`configs/recipe.toml`:

```toml
# Training recipe shared by all four arms (spec §5.3).
# Lines marked PILOT are chosen by the G3 pilot; then set `frozen = true` and commit.
# `snz train` refuses non-pilot runs while frozen = false.
frozen = false

batch_size = 8                 # PILOT
lr = 0.01                      # PILOT  SGD learning rate at batch_size
momentum = 0.9
weight_decay = 1e-4
iterations = 27000             # PILOT  identical for every arm
warmup_iters = 1000
lr_steps = [0.67, 0.89]        # PILOT  fractions of `iterations`; LR x0.1 at each
min_sizes = [576, 608, 640, 672, 704, 720]  # PILOT scale jitter (short side); last = eval size
max_size = 1280
hflip_p = 0.5
p_original = 0.5               # half-stylized arms: P(original image)
trainable_layers = 3
bf16 = true
num_workers = 8
log_every = 20
checkpoint_every = 500
```

`configs/arms.toml`:

```toml
# The 2x2 design (spec §3).
[BASE]
backbone = "in1k_v1"
stylized = false

[STY]
backbone = "in1k_v1"
stylized = true

[SIN]
backbone = "sin"
stylized = false

[BOTH]
backbone = "sin"
stylized = true
```

- [ ] **Step 2: Write the failing tests**

`tests/test_train.py`:

```python
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from helpers import write_image
from torchvision.models import resnet50

from snz.data.bdd import Box, Frame
from snz.train.config import Arm, load_arms, load_recipe
from snz.train.train import SeededStream, lr_factor, train_run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def tiny_recipe():
    return replace(
        load_recipe(CONFIGS / "recipe.toml"),
        batch_size=2, iterations=4, warmup_iters=2, lr_steps=[0.5], min_sizes=[48],
        max_size=64, bf16=False, num_workers=0, log_every=1, checkpoint_every=1,
    )


def test_configs_load():
    r = load_recipe(CONFIGS / "recipe.toml")
    assert r.min_sizes[-1] == 720 and r.p_original == 0.5 and r.trainable_layers == 3
    arms = load_arms(CONFIGS / "arms.toml")
    assert arms["BOTH"] == Arm("BOTH", "sin", True)
    assert list(arms) == ["BASE", "STY", "SIN", "BOTH"]


def test_load_recipe_rejects_unknown_keys(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text((CONFIGS / "recipe.toml").read_text() + "\nlearning_rate = 1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        load_recipe(p)


def test_seeded_stream_resumes_mid_epoch():
    it = iter(SeededStream(5, seed=3))
    first = [next(it) for _ in range(12)]
    assert sorted(i for i, e in first[:5]) == list(range(5))
    assert {e for _, e in first[5:10]} == {1}
    resumed = iter(SeededStream(5, seed=3, start=7))
    assert [next(resumed) for _ in range(5)] == first[7:12]


def test_lr_factor():
    assert lr_factor(0, 10, [50]) == pytest.approx(0.001)
    assert lr_factor(10, 10, [50]) == 1.0
    assert lr_factor(50, 10, [50]) == pytest.approx(0.1)


def _data(tmp_path):
    frames = []
    for i in range(4):
        write_image(tmp_path / "img" / f"f{i}.jpg", size=(64, 48), color=(40 * i, 90, 160))
        boxes = (Box("car", 5, 5, 30, 30),) if i != 3 else ()  # one empty frame
        frames.append(Frame(f"f{i}.jpg", "clear", "daytime", boxes))
    return frames


def _state():
    torch.manual_seed(123)
    return resnet50(weights=None).state_dict()


def _run(tmp_path, out, recipe, **kw):
    return train_run(arm=Arm("BASE", "in1k_v1", False), seed=0, recipe=recipe,
                     frames=_data(tmp_path), image_dir=tmp_path / "img", stylized_dir=None,
                     n_copies=2, backbone_state=_state(), out_dir=out, device="cpu", **kw)


def test_train_run_writes_outputs(tmp_path):
    summary = _run(tmp_path, tmp_path / "run", tiny_recipe())
    out = tmp_path / "run"
    assert (out / "final.pt").exists() and (out / "last.pt").exists()
    lines = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert [x["iteration"] for x in lines] == [1, 2, 3, 4]
    assert all(math.isfinite(x["loss"]) for x in lines)
    run = json.loads((out / "run.json").read_text())
    assert run["arm"] == "BASE" and run["iterations"] == 4 and summary == run


def test_resume_matches_uninterrupted_run(tmp_path):
    recipe = tiny_recipe()
    _run(tmp_path, tmp_path / "straight", recipe)
    assert _run(tmp_path, tmp_path / "resumed", recipe, stop_at=2) == {"stopped_at": 2}
    assert not (tmp_path / "resumed" / "final.pt").exists()
    _run(tmp_path, tmp_path / "resumed", recipe)
    a = torch.load(tmp_path / "straight" / "final.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=True)
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_resume_refuses_a_different_config(tmp_path):
    recipe = tiny_recipe()
    _run(tmp_path, tmp_path / "r", recipe, stop_at=2)
    with pytest.raises(ValueError, match="different"):
        _run(tmp_path, tmp_path / "r", replace(recipe, lr=0.5))


def test_divergence_stops_the_run(tmp_path):
    with pytest.raises(FloatingPointError, match="non-finite"):
        _run(tmp_path, tmp_path / "nan", replace(tiny_recipe(), lr=1e12, warmup_iters=0))
    assert not (tmp_path / "nan" / "final.pt").exists()
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.train.config'`.

- [ ] **Step 4: Implement `src/snz/train/config.py`**

```python
"""Training recipe and arm definitions (configs/recipe.toml, configs/arms.toml)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    frozen: bool
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    iterations: int
    warmup_iters: int
    lr_steps: list[float]
    min_sizes: list[int]
    max_size: int
    hflip_p: float
    p_original: float
    trainable_layers: int
    bf16: bool
    num_workers: int
    log_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class Arm:
    name: str
    backbone: str
    stylized: bool


def load_recipe(path: Path) -> Recipe:
    raw = tomllib.loads(Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(Recipe)}
    unknown, missing = set(raw) - fields, fields - set(raw)
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return Recipe(**raw)


def recipe_sha256(recipe: Recipe) -> str:
    blob = json.dumps(dataclasses.asdict(recipe), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def load_arms(path: Path) -> dict[str, Arm]:
    raw = tomllib.loads(Path(path).read_text())
    return {name: Arm(name, v["backbone"], bool(v["stylized"])) for name, v in raw.items()}
```

- [ ] **Step 5: Implement `src/snz/train/train.py`**

```python
"""Iteration-based Faster R-CNN training with exact resume (spec §5.3).

Determinism: model init uses torch.manual_seed(seed); data order comes from
SeededStream; augmentation choices are hashes (dataset.py); the global RNG used by
the RPN/ROI samplers is saved in every checkpoint and restored on resume.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from snz.data.bdd import Frame
from snz.train.config import Arm, Recipe, recipe_sha256
from snz.train.dataset import BDDDetection, collate
from snz.train.model import build_model


class SeededStream(Sampler):
    """Endless stream of (index, epoch): a fresh seeded permutation each epoch."""

    def __init__(self, n: int, seed: int, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self) -> Iterator[tuple[int, int]]:
        pos = self.start
        while True:
            epoch, offset = divmod(pos, self.n)
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            for i in torch.randperm(self.n, generator=g).tolist()[offset:]:
                yield i, epoch
            pos = (epoch + 1) * self.n


def lr_factor(iteration: int, warmup_iters: int, step_iters: Sequence[int]) -> float:
    factor = 0.1 ** sum(iteration >= s for s in step_iters)
    if iteration < warmup_iters:
        factor *= 0.001 + 0.999 * iteration / warmup_iters
    return factor


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train_run(
    *,
    arm: Arm,
    seed: int,
    recipe: Recipe,
    frames: Sequence[Frame],
    image_dir: Path,
    stylized_dir: Path | None,
    n_copies: int,
    backbone_state: Mapping[str, Tensor],
    out_dir: Path,
    device: str = "cuda",
    stop_at: int | None = None,
    pilot: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if arm.stylized and stylized_dir is None:
        raise ValueError(f"arm {arm.name} trains on stylized copies; stylized_dir is required")
    identity = {"arm": arm.name, "backbone": arm.backbone, "stylized": arm.stylized,
                "seed": seed, "recipe_sha256": recipe_sha256(recipe), "frames": len(frames)}

    torch.manual_seed(seed)
    model = build_model(backbone_state, trainable_layers=recipe.trainable_layers,
                        min_size=tuple(recipe.min_sizes), max_size=recipe.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=recipe.lr, momentum=recipe.momentum,
                          weight_decay=recipe.weight_decay)

    start = 0
    last = out_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["identity"] != identity:
            raise ValueError(f"{last} was written by a different run config: "
                             f"{ckpt['identity']} != {identity}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng"])
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        start = ckpt["iteration"]

    dataset = BDDDetection(
        frames, image_dir, stylized_dir=stylized_dir if arm.stylized else None,
        p_original=recipe.p_original, n_copies=n_copies, seed=seed, train=True,
        hflip_p=recipe.hflip_p,
    )
    loader = DataLoader(
        dataset, batch_size=recipe.batch_size,
        sampler=SeededStream(len(dataset), seed, start * recipe.batch_size),
        num_workers=recipe.num_workers, collate_fn=collate,
        pin_memory=device.startswith("cuda"),
        # own generator: creating the iterator must not draw from the global RNG
        generator=torch.Generator().manual_seed(seed),
    )
    step_iters = [round(f * recipe.iterations) for f in recipe.lr_steps]
    use_bf16 = recipe.bf16 and device.startswith("cuda")
    end = recipe.iterations if stop_at is None else min(stop_at, recipe.iterations)

    def checkpoint(iteration: int) -> None:
        _atomic_save({
            "identity": identity, "iteration": iteration, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, last)

    model.train()
    it, t0 = start, time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if it < end:
        batches = iter(loader)
        with open(out_dir / "log.jsonl", "a") as log:
            while it < end:
                images, targets = next(batches)
                for group in opt.param_groups:
                    group["lr"] = recipe.lr * lr_factor(it, recipe.warmup_iters, step_iters)
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    losses = model(images, targets)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    parts = {k: float(v) for k, v in losses.items()}
                    raise FloatingPointError(f"non-finite loss at iteration {it}: {parts} "
                                             f"(last good checkpoint: {last})")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                it += 1
                if it % recipe.log_every == 0 or it == end:
                    log.write(json.dumps({"iteration": it, "loss": float(loss),
                                          "lr": opt.param_groups[0]["lr"],
                                          **{k: float(v) for k, v in losses.items()}}) + "\n")
                    log.flush()
                if it % recipe.checkpoint_every == 0 or it == end:
                    checkpoint(it)
        del batches

    if it < recipe.iterations:  # stopped early on purpose (stop_at)
        return {"stopped_at": it}
    seconds = time.perf_counter() - t0
    _atomic_save(model.state_dict(), out_dir / "final.pt")
    summary = {
        **identity, "iterations": recipe.iterations, "recipe": dataclasses.asdict(recipe),
        "pilot": pilot, "git_commit": _git_commit(), "device": device,
        "torch": torch.__version__,
        "iters_per_sec_this_session": round((it - start) / seconds, 4) if it > start else None,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if device.startswith("cuda") else None,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 6: Add the `train` command to `src/snz/cli.py`**

Add `import dataclasses` at the top. Inside `build_parser`:

```python
    p = commands.add_parser("train", help="train one arm/seed (pilot at G3, full runs at G5)")
    p.add_argument("--arm", choices=["BASE", "STY", "SIN", "BOTH"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--labels", type=Path, required=True, help="det_train.json")
    p.add_argument("--manifest", type=Path, default=Path("manifests/train.csv"))
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--stylized", type=Path, help="stylized/train (STY, BOTH)")
    p.add_argument("--sin-weights", type=Path, help="resnet50_train_60_epochs-c8e5653e.pth.tar")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--recipe", type=Path, default=Path("configs/recipe.toml"))
    p.add_argument("--arms", type=Path, default=Path("configs/arms.toml"))
    p.add_argument("--stylize-config", type=Path, default=Path("configs/stylize.toml"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--pilot", action="store_true", help="G3 pilot: BASE only, overrides allowed")
    p.add_argument("--iterations", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.set_defaults(func=_cmd_train)
```

Module level:

```python
def _cmd_train(args) -> int:
    from snz.data.bdd import load_frames, select_frames
    from snz.data.manifest import read_manifest
    from snz.stylize.build import load_stylize_config
    from snz.train.config import load_arms, load_recipe
    from snz.train.model import load_backbone_state
    from snz.train.train import train_run

    recipe = load_recipe(args.recipe)
    overrides = {k: v for k, v in (("iterations", args.iterations), ("lr", args.lr),
                                   ("batch_size", args.batch_size)) if v is not None}
    if args.pilot:
        if args.arm != "BASE":
            raise SystemExit("error: the pilot trains the BASE arm only (spec §5.4)")
        recipe = dataclasses.replace(recipe, **overrides)
    elif overrides:
        raise SystemExit("error: --iterations/--lr/--batch-size are pilot-only; edit the recipe")
    elif not recipe.frozen:
        raise SystemExit("error: configs/recipe.toml is not frozen (gate G3); use --pilot")
    arm = load_arms(args.arms)[args.arm]
    if arm.stylized and args.stylized is None:
        raise SystemExit(f"error: arm {arm.name} needs --stylized")
    frames, _ = load_frames(args.labels)
    frames = select_frames(frames, [n for n, _ in read_manifest(args.manifest)])
    summary = train_run(
        arm=arm, seed=args.seed, recipe=recipe, frames=frames, image_dir=args.images,
        stylized_dir=args.stylized,
        n_copies=load_stylize_config(args.stylize_config).train_copies,
        backbone_state=load_backbone_state(arm.backbone, args.sin_weights),
        out_dir=args.out, device=args.device, pilot=args.pilot,
    )
    print(json.dumps(summary, indent=2))
    return 0
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_train.py -v && uv run ruff check src tests`
Expected: all pass (the training tests take several seconds each on CPU). If `test_resume_matches_uninterrupted_run` fails with tiny float differences only, check that `num_workers=0` and that nothing draws from the global RNG between model build and the training loop. Do not loosen the tolerance without finding the cause.

- [ ] **Step 8: Commit**

```bash
git add src/snz configs/recipe.toml configs/arms.toml tests/test_train.py
git commit -m "snz: recipe/arms configs and resumable training loop; train command"
```

---

### Task 9: Predict, COCO evaluation, and the test-split gate

**Files:**
- Create: `src/snz/eval/__init__.py` (empty), `src/snz/eval/guard.py`, `src/snz/eval/predict.py`, `src/snz/eval/coco_eval.py`
- Modify: `src/snz/cli.py` (add `evaluate`)
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `Frame`, `EVAL_CLASSES`, `CLASSES`, `TRAIN_SPLITS` (Task 2); `BDDDetection`, `collate` (Task 7); `build_model` (Task 6); `load_recipe` (Task 8).
- Produces:
  - `PREREG_TAG = "prereg-v1"`; `ensure_split_allowed(split, repo=Path(".")) -> None` (raises `PermissionError` for any split not in `TRAIN_SPLITS` unless the tag exists)
  - `predict(model, frames, image_dir, *, device, batch_size=4, num_workers=4) -> list[dict]` (records `{"name", "boxes": [[x1,y1,x2,y2]], "labels": [category], "scores": [float]}`); `write_detections(path, records)`; `read_detections(path) -> list[dict]`
  - `evaluate(frames, detections, classes=EVAL_CLASSES) -> dict` with keys `mAP`, `AP50`, `per_class` (class → AP, or `None` if the class has no ground truth), `images`, `detections`
  - Plan 2 builds on these record formats.

- [ ] **Step 1: Write the failing tests**

`tests/test_eval.py`:

```python
import subprocess

import pytest
import torch
from helpers import write_image
from torchvision.models import resnet50

from snz.data.bdd import Box, Frame
from snz.eval.coco_eval import evaluate
from snz.eval.guard import ensure_split_allowed
from snz.eval.predict import predict, read_detections, write_detections
from snz.train.model import build_model


def _frame(name, *boxes):
    return Frame(name, "clear", "daytime", tuple(Box(*b) for b in boxes))


def _det(name, *items):
    return {"name": name, "boxes": [list(i[1:5]) for i in items],
            "labels": [i[0] for i in items], "scores": [i[5] for i in items]}


FRAMES = [
    _frame("a.jpg", ("car", 0, 0, 10, 10), ("pedestrian", 20, 20, 30, 40)),
    _frame("b.jpg", ("car", 5, 5, 25, 25)),
]


def test_perfect_detections_score_one():
    dets = [_det("a.jpg", ("car", 0, 0, 10, 10, 0.9), ("pedestrian", 20, 20, 30, 40, 0.8)),
            _det("b.jpg", ("car", 5, 5, 25, 25, 0.7))]
    r = evaluate(FRAMES, dets)
    assert r["mAP"] == pytest.approx(1.0) and r["AP50"] == pytest.approx(1.0)
    assert r["per_class"]["car"] == pytest.approx(1.0)
    assert r["per_class"]["bus"] is None  # no ground truth: excluded, not zero


def test_hand_computed_half_recall():
    # one class, 2 GT, one exact hit: precision 1 up to recall 0.5 -> 51/101 per IoU threshold
    frames = [_frame("a.jpg", ("car", 0, 0, 10, 10), ("car", 50, 50, 60, 60))]
    r = evaluate(frames, [_det("a.jpg", ("car", 0, 0, 10, 10, 0.9))])
    assert r["mAP"] == pytest.approx(51 / 101, abs=1e-6)


def test_train_class_and_missing_images_and_empty():
    frames = FRAMES + [_frame("c.jpg", ("train", 0, 0, 50, 50))]
    dets = [_det("a.jpg", ("car", 0, 0, 10, 10, 0.9), ("train", 0, 0, 5, 5, 0.9))]
    r = evaluate(frames, dets)  # b.jpg / c.jpg have no detection records: treated as empty
    assert "train" not in r["per_class"] and r["images"] == 3
    assert evaluate(FRAMES, [])["mAP"] == 0.0
    with pytest.raises(ValueError, match="zzz.jpg"):
        evaluate(FRAMES, [_det("zzz.jpg")])


def test_guard_requires_prereg_tag(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    ensure_split_allowed("id-val", tmp_path)
    for split in ("t-clean", "t-night", "t-texture", "footage"):
        with pytest.raises(PermissionError, match="prereg-v1"):
            ensure_split_allowed(split, tmp_path)
    subprocess.run(["git", "tag", "prereg-v1"], cwd=tmp_path, check=True)
    ensure_split_allowed("t-clean", tmp_path)


def test_predict_and_detections_roundtrip(tmp_path):
    frames = [_frame(f"f{i}.jpg", ("car", 5, 5, 30, 30)) for i in range(3)]
    for f in frames:
        write_image(tmp_path / "img" / f.name, size=(64, 48))
    torch.manual_seed(0)
    model = build_model(resnet50(weights=None).state_dict(), min_size=48, max_size=64,
                        box_score_thresh=0.0)
    records = predict(model, frames, tmp_path / "img", device="cpu", batch_size=2, num_workers=0)
    assert [r["name"] for r in records] == ["f0.jpg", "f1.jpg", "f2.jpg"]
    write_detections(tmp_path / "d.jsonl", records)
    assert read_detections(tmp_path / "d.jsonl") == records
    evaluate(frames, records)  # runs end to end on real model output
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.eval'`.

- [ ] **Step 3: Implement `src/snz/eval/guard.py`**

```python
"""Gate G4: no test split is evaluated before the preregistration tag exists (spec §11)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from snz.data.bdd import TRAIN_SPLITS

PREREG_TAG = "prereg-v1"


def ensure_split_allowed(split: str, repo: Path = Path(".")) -> None:
    if split in TRAIN_SPLITS:
        return
    tags = subprocess.run(["git", "tag", "--list", PREREG_TAG], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.split()
    if PREREG_TAG not in tags:
        raise PermissionError(
            f"split {split!r} is held out: evaluation is blocked until the preregistration "
            f"is committed and tagged {PREREG_TAG!r} (gate G4)")
```

- [ ] **Step 4: Implement `src/snz/eval/predict.py`**

```python
"""Run a detector over a split and store detections as JSONL."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snz.data.bdd import CLASSES, Frame
from snz.train.dataset import BDDDetection, collate


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    frames: Sequence[Frame],
    image_dir: Path,
    *,
    device: str,
    batch_size: int = 4,
    num_workers: int = 4,
) -> list[dict]:
    model.eval().to(device)
    loader = DataLoader(BDDDetection(frames, image_dir, train=False), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=collate)
    records, names = [], iter(f.name for f in frames)
    for images, _ in loader:
        for out in model([im.to(device) for im in images]):
            records.append({
                "name": next(names),
                "boxes": [[round(v, 2) for v in b] for b in out["boxes"].float().cpu().tolist()],
                "labels": [CLASSES[i - 1] for i in out["labels"].cpu().tolist()],
                "scores": [round(s, 5) for s in out["scores"].float().cpu().tolist()],
            })
    return records


def write_detections(path: Path, records: Sequence[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_detections(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
```

- [ ] **Step 5: Implement `src/snz/eval/coco_eval.py`**

```python
"""COCO-style mAP@[.5:.95], AP50 and per-class AP via pycocotools (spec §7.1)."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from snz.data.bdd import EVAL_CLASSES, Frame


def evaluate(
    frames: Sequence[Frame], detections: Sequence[dict], classes: Sequence[str] = EVAL_CLASSES
) -> dict:
    cat_ids = {c: i + 1 for i, c in enumerate(classes)}
    image_ids = {f.name: i + 1 for i, f in enumerate(frames)}
    unknown = [d["name"] for d in detections if d["name"] not in image_ids]
    if unknown:
        raise ValueError(f"{len(unknown)} detection record(s) for images not in this split, "
                         f"e.g. {unknown[:3]}")

    annotations = []
    for f in frames:
        for b in f.boxes:
            if b.category in cat_ids:
                w, h = b.x2 - b.x1, b.y2 - b.y1
                annotations.append({"id": len(annotations) + 1, "image_id": image_ids[f.name],
                                    "category_id": cat_ids[b.category],
                                    "bbox": [b.x1, b.y1, w, h], "area": w * h, "iscrowd": 0})
    dts = []
    for d in detections:
        for (x1, y1, x2, y2), label, score in zip(d["boxes"], d["labels"], d["scores"],
                                                  strict=True):
            if label in cat_ids:
                dts.append({"image_id": image_ids[d["name"]], "category_id": cat_ids[label],
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})

    result = {"images": len(frames), "detections": len(dts)}
    with_gt = {a["category_id"] for a in annotations}
    if not dts or not annotations:
        result.update(mAP=0.0, AP50=0.0,
                      per_class={c: (0.0 if cat_ids[c] in with_gt else None) for c in classes})
        return result

    gt = COCO()
    gt.dataset = {"images": [{"id": i} for i in image_ids.values()], "annotations": annotations,
                  "categories": [{"id": i, "name": c} for c, i in cat_ids.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dts)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval["precision"]  # [IoU, recall, class, area, maxDets]
    per_class = {}
    for k, c in enumerate(classes):
        if cat_ids[c] not in with_gt:
            per_class[c] = None
            continue
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        per_class[c] = float(np.mean(p)) if p.size else 0.0
    result.update(mAP=float(ev.stats[0]), AP50=float(ev.stats[1]), per_class=per_class)
    return result
```

- [ ] **Step 6: Add the `evaluate` command to `src/snz/cli.py`**

Inside `build_parser`:

```python
    p = commands.add_parser("evaluate", help="predict + mAP on one split")
    p.add_argument("--split", required=True, help="id-val before G4; test splits after")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="<run>/final.pt")
    p.add_argument("--recipe", type=Path, default=Path("configs/recipe.toml"))
    p.add_argument("--out", type=Path, required=True, help="directory for detections + metrics")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_evaluate)
```

Module level:

```python
def _cmd_evaluate(args) -> int:
    import torch

    from snz.data.bdd import load_frames, select_frames
    from snz.data.manifest import read_manifest
    from snz.eval.coco_eval import evaluate
    from snz.eval.guard import ensure_split_allowed
    from snz.eval.predict import predict, write_detections
    from snz.train.config import load_recipe
    from snz.train.model import build_model

    ensure_split_allowed(args.split)
    recipe = load_recipe(args.recipe)
    frames, _ = load_frames(args.labels)
    frames = select_frames(frames, [n for n, _ in read_manifest(args.manifest)])
    model = build_model(None, trainable_layers=recipe.trainable_layers,
                        min_size=tuple(recipe.min_sizes), max_size=recipe.max_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    records = predict(model, frames, args.images, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    write_detections(args.out / f"{args.split}.detections.jsonl", records)
    metrics = {"split": args.split, "checkpoint": str(args.checkpoint),
               **evaluate(frames, records)}
    (args.out / f"{args.split}.metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("split", "mAP", "AP50")}, indent=2))
    return 0
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_eval.py -v && uv run ruff check src tests`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/snz/eval src/snz/cli.py tests/test_eval.py
git commit -m "snz: predict, pycocotools mAP, and prereg gate on test splits; evaluate command"
```

---

### Task 10: G1 smoke command and GX10 scripts

**Files:**
- Create: `src/snz/train/smoke.py`, `scripts/gx10_setup.sh`, `scripts/fetch_weights.sh`, `scripts/fetch_styles.sh`, `scripts/fetch_bdd.sh`, `docs/runbook.md`
- Modify: `src/snz/cli.py` (add `smoke`)
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `train_run`, `Arm`, `Recipe` (Task 8); `build_model`, `load_backbone_state`, `BACKBONES` (Task 6); `Box`, `Frame`, `CLASSES` (Task 2).
- Produces: `run_smoke(states: Mapping[str, state_dict], recipe, work_dir, *, device, iterations=50, image_size=(1280, 720), n_images=8) -> dict`, with per-backbone `{"loaded": True, "trainable_params", "iterations", "iters_per_sec", "peak_mem_gb", "final_loss"}` plus environment info. The CLI writes it to `results/g1_smoke.json`.

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:

```python
from dataclasses import replace
from pathlib import Path

import torch
from torchvision.models import resnet50

from snz.train.config import load_recipe
from snz.train.smoke import run_smoke

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_run_smoke_tiny(tmp_path):
    torch.manual_seed(0)
    states = {"in1k_v1": resnet50(weights=None).state_dict(),
              "sin": resnet50(weights=None).state_dict()}
    recipe = replace(load_recipe(CONFIGS / "recipe.toml"), batch_size=2, min_sizes=[48],
                     max_size=64, bf16=False, num_workers=0, warmup_iters=1)
    report = run_smoke(states, recipe, tmp_path, device="cpu", iterations=2,
                       image_size=(64, 48), n_images=4)
    for kind in ("in1k_v1", "sin"):
        r = report["backbones"][kind]
        assert r["loaded"] and r["iterations"] == 2 and r["final_loss"] > 0
    assert (report["backbones"]["in1k_v1"]["trainable_params"]
            == report["backbones"]["sin"]["trainable_params"])
    assert report["torch"] == torch.__version__
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'snz.train.smoke'`.

- [ ] **Step 3: Implement `src/snz/train/smoke.py`**

```python
"""Gate G1: both backbones load into the detector and a short training run completes.

Runs on synthetic images (before any dataset is downloaded) at full 1280x720, so the
measured speed and memory are a first estimate for the pilot.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import random
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image

from snz.data.bdd import CLASSES, Box, Frame
from snz.train.config import Arm, Recipe
from snz.train.model import build_model
from snz.train.train import train_run


def _synthetic(image_dir: Path, n: int, size: tuple[int, int]) -> list[Frame]:
    rng = random.Random(0)
    w, h = size
    frames = []
    for i in range(n):
        name = f"synthetic_{i}.jpg"
        Image.effect_noise(size, 64).convert("RGB").save(image_dir / name)
        boxes = []
        for _ in range(rng.randint(1, 6)):
            x1, y1 = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
            boxes.append(Box(rng.choice(CLASSES), x1, y1,
                             x1 + rng.uniform(8, w * 0.2), y1 + rng.uniform(8, h * 0.2)))
        frames.append(Frame(name, "clear", "daytime", tuple(boxes)))
    return frames


def run_smoke(
    states: Mapping[str, Mapping[str, torch.Tensor]],
    recipe: Recipe,
    work_dir: Path,
    *,
    device: str,
    iterations: int = 50,
    image_size: tuple[int, int] = (1280, 720),
    n_images: int = 8,
) -> dict:
    work_dir = Path(work_dir)
    (work_dir / "images").mkdir(parents=True, exist_ok=True)
    frames = _synthetic(work_dir / "images", n_images, image_size)
    recipe = dataclasses.replace(recipe, iterations=iterations, lr_steps=[],
                                 log_every=1, checkpoint_every=iterations)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "device_name": (torch.cuda.get_device_name(0) if device.startswith("cuda")
                        else platform.processor()),
        "recipe": dataclasses.asdict(recipe),
        "backbones": {},
    }
    for kind, state in states.items():
        model = build_model(state, trainable_layers=recipe.trainable_layers)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        out = work_dir / f"run_{kind}"
        summary = train_run(arm=Arm(f"smoke-{kind}", kind, False), seed=0, recipe=recipe,
                            frames=frames, image_dir=work_dir / "images", stylized_dir=None,
                            n_copies=2, backbone_state=state, out_dir=out, device=device)
        last = json.loads((out / "log.jsonl").read_text().splitlines()[-1])
        report["backbones"][kind] = {
            "loaded": True,
            "trainable_params": trainable,
            "iterations": last["iteration"],
            "iters_per_sec": summary["iters_per_sec_this_session"],
            "peak_mem_gb": summary["peak_mem_gb"],
            "final_loss": last["loss"],
        }
    return report
```

- [ ] **Step 4: Add the `smoke` command to `src/snz/cli.py`**

Inside `build_parser`:

```python
    p = commands.add_parser("smoke", help="G1: load both backbones, short training run")
    p.add_argument("--sin-weights", type=Path, required=True)
    p.add_argument("--recipe", type=Path, default=Path("configs/recipe.toml"))
    p.add_argument("--work-dir", type=Path, required=True, help="scratch dir outside git")
    p.add_argument("--out", type=Path, default=Path("results/g1_smoke.json"))
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_smoke)
```

Module level:

```python
def _cmd_smoke(args) -> int:
    from snz.train.config import load_recipe
    from snz.train.model import BACKBONES, load_backbone_state
    from snz.train.smoke import run_smoke

    states = {kind: load_backbone_state(kind, args.sin_weights) for kind in BACKBONES}
    report = run_smoke(states, load_recipe(args.recipe), args.work_dir, device=args.device,
                       iterations=args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["backbones"], indent=2))
    return 0
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -v && uv run ruff check src tests`
Expected: the full suite passes.

- [ ] **Step 6: Write the GX10 scripts** (all `chmod +x`)

`scripts/gx10_setup.sh`:

```bash
#!/usr/bin/env bash
# One-time GX10 environment: uv + project venv with CUDA 13.0 PyTorch (spec §5.5).
# Fallback if the wheels fail: NVIDIA PyTorch container (nvcr.io/nvidia/pytorch).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
print("device", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
EOF
```

`scripts/fetch_weights.sh`:

```bash
#!/usr/bin/env bash
# Backbone + AdaIN weights into $SNZ_DATA/weights, verified against README §5.1.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
DATA="${SNZ_DATA:-$HOME/snz-data}"
mkdir -p "$DATA/weights"
fetch() {  # url file
  [ -s "$DATA/weights/$2" ] && return
  curl -fL --retry 3 -o "$DATA/weights/$2.part" "$1"
  mv "$DATA/weights/$2.part" "$DATA/weights/$2"
}
fetch "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar" \
  resnet50_train_60_epochs-c8e5653e.pth.tar
fetch https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth decoder.pth
fetch https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth \
  vgg_normalised.pth
(cd "$DATA/weights" && sha256sum -c -) <<'EOF'
c8e5653e47bbf3f6efad5da531f3acea9963acb93362b6530cf39b2ba4750da3  resnet50_train_60_epochs-c8e5653e.pth.tar
379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472  decoder.pth
804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a  vgg_normalised.pth
EOF
# torchvision IN1K_V1 (cached under ~/.cache/torch; hash prefix checked by torchvision)
uv run python -c "from torchvision.models import ResNet50_Weights as W; W.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True); print('in1k_v1 ok')"
```

`scripts/fetch_styles.sh`:

```bash
#!/usr/bin/env bash
# Describable Textures Dataset ("for research purposes") into $SNZ_DATA/dtd.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
DATA="${SNZ_DATA:-$HOME/snz-data}"
mkdir -p "$DATA" manifests
cd "$DATA"
[ -s dtd-r1.0.1.tar.gz ] || curl -fL --retry 3 -o dtd-r1.0.1.tar.gz \
  https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
[ -d dtd/images ] || tar -xzf dtd-r1.0.1.tar.gz
sha256sum dtd-r1.0.1.tar.gz | tee -a "$REPO/manifests/downloads.sha256"
echo "styles: $(find dtd/images -name '*.jpg' | wc -l) images"
```

`scripts/fetch_bdd.sh`:

```bash
#!/usr/bin/env bash
# BDD100K 100K images + det_20 labels into $SNZ_DATA/bdd100k.
# Read and accept the BDD100K terms (education, research, not-for-profit) first:
#   https://doc.bdd100k.com/download.html
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
[ "${BDD_TERMS_ACCEPTED:-}" = "yes" ] || {
  echo "Set BDD_TERMS_ACCEPTED=yes after reading the BDD100K terms." >&2; exit 1; }
DATA="${SNZ_DATA:-$HOME/snz-data}"
BASE=https://dl.cv.ethz.ch/bdd100k/data
FILES=(100k_images_train.zip 100k_images_val.zip bdd100k_det_20_labels_trainval.zip)
mkdir -p "$DATA" manifests
cd "$DATA"
for f in "${FILES[@]}"; do
  [ -s "$f" ] || curl -fL --retry 3 -C - -o "$f" "$BASE/$f"
done
md5sum "${FILES[@]}" | tee -a "$REPO/manifests/downloads.md5"
for f in "${FILES[@]}"; do unzip -qn "$f"; done
ls bdd100k/images/100k bdd100k/labels/det_20
```

(Executor: before relying on `fetch_bdd.sh`, open `https://doc.bdd100k.com/download.html` and the `$BASE` index. Correct the three file names if they differ, compare against the README md5s, and record any corrections in the commit.)

`docs/runbook.md` lists the operator's G1–G3 commands:

````markdown
# GX10 runbook: gates G1–G3

All commands run on the GX10 from the repository root. Data lives in `$SNZ_DATA`
(default `~/snz-data`) and never enters git.

## G1 — smoke test

```bash
scripts/gx10_setup.sh
scripts/fetch_weights.sh
uv run pytest
uv run snz smoke --sin-weights ~/snz-data/weights/resnet50_train_60_epochs-c8e5653e.pth.tar \
  --work-dir ~/snz-data/smoke
```

Commit `results/g1_smoke.json`.

## G2 — data

1. Read the BDD100K terms and the DTD terms; note the date in the commit message.
2. Download and build:

```bash
BDD_TERMS_ACCEPTED=yes scripts/fetch_bdd.sh
scripts/fetch_styles.sh
D=~/snz-data; W=$D/weights
uv run snz build-splits \
  --train-labels $D/bdd100k/labels/det_20/det_train.json \
  --val-labels   $D/bdd100k/labels/det_20/det_val.json \
  --train-images $D/bdd100k/images/100k/train --val-images $D/bdd100k/images/100k/val
uv run snz stylize --set train --images $D/bdd100k/images/100k/train \
  --styles-root $D/dtd/images --out $D/stylized/train \
  --vgg $W/vgg_normalised.pth --decoder $W/decoder.pth
uv run snz stylize --set t-texture --images $D/bdd100k/images/100k/val \
  --styles-root $D/dtd/images --out $D/stylized/t-texture \
  --vgg $W/vgg_normalised.pth --decoder $W/decoder.pth
```

Commit `manifests/`. Both `stylize` runs can be stopped and rerun; finished images are skipped.

## G3 — pilot (BASE, seed 0, id-val only)

```bash
D=~/snz-data
for LR in 0.01 0.02; do
  uv run snz train --pilot --arm BASE --seed 0 --lr $LR --iterations 6000 \
    --labels $D/bdd100k/labels/det_20/det_train.json --images $D/bdd100k/images/100k/train \
    --out $D/runs/pilot/lr$LR
  uv run snz evaluate --split id-val --labels $D/bdd100k/labels/det_20/det_train.json \
    --manifest manifests/id-val.csv --images $D/bdd100k/images/100k/train \
    --checkpoint $D/runs/pilot/lr$LR/final.pt --out results/pilot/lr$LR
done
```

Pick the learning rate with the higher id-val mAP. From `iters_per_sec_this_session` in
each `run.json`, choose `iterations` so that 12 runs fit the budget (spec §5.4). If the
budget is too large, reduce the schedule or input scale **identically for all arms**.
Then edit `configs/recipe.toml`, set `frozen = true`, and commit `G3: recipe frozen`
with `results/pilot/*/id-val.metrics.json` (not the detection files).
Only id-val is evaluated here; test splits stay blocked until `prereg-v1`.
````

- [ ] **Step 7: Run all checks and commit**

Run: `chmod +x scripts/*.sh && uv run pytest && uv run ruff check src tests && bash -n scripts/*.sh`
Expected: all pass; no shell syntax errors.

```bash
git add src/snz scripts docs/runbook.md tests/test_smoke.py
git commit -m "snz: G1 smoke command, GX10 setup/fetch scripts, G1-G3 runbook"
git pull --rebase origin main && git push origin main
```

---

### Task 11: Run gate G1 on the GX10

**Files:**
- Create: `results/g1_smoke.json` (from the GX10)
- Modify: `README.md` (status line, §8 layout), `docs/runbook.md` (any corrections found while running)

**Interfaces:**
- Consumes: everything above. Host `asus@100.104.225.90` (Tailscale SSH, key auth).

- [ ] **Step 1: Sync the repository to the GX10**

```bash
rsync -az --delete --exclude .venv --exclude node_modules --exclude .git \
  ~/Shapes-n-Sizes/ asus@100.104.225.90:~/Shapes-n-Sizes/
```

Expected: completes without errors. `.git` is excluded, so the GX10 copy is only a working tree; commits happen on the Mac.

- [ ] **Step 2: Environment**

Run: `ssh asus@100.104.225.90 'cd ~/Shapes-n-Sizes && scripts/gx10_setup.sh'`
Expected: torch ≥ 2.9 with cuda 13.0, `device NVIDIA GB10`, `bf16 matmul ok True`. If the cu130 aarch64 wheels are missing, stop and report. The fallback (NVIDIA container) needs Docker permission for `asus`, which the user must grant.

- [ ] **Step 3: Weights**

Run: `ssh asus@100.104.225.90 'cd ~/Shapes-n-Sizes && scripts/fetch_weights.sh'`
Expected: three `OK` lines from `sha256sum -c`, then `in1k_v1 ok`.

- [ ] **Step 4: Test suite on the GX10**

Run: `ssh asus@100.104.225.90 'cd ~/Shapes-n-Sizes && export PATH=$HOME/.local/bin:$PATH && uv run pytest'`
Expected: all pass. The guard test runs `git` in a temporary repo, so it does not need `.git` here.

- [ ] **Step 5: Smoke test**

```bash
ssh asus@100.104.225.90 'cd ~/Shapes-n-Sizes && export PATH=$HOME/.local/bin:$PATH && \
  uv run snz smoke --sin-weights ~/snz-data/weights/resnet50_train_60_epochs-c8e5653e.pth.tar \
  --work-dir ~/snz-data/smoke'
```

Expected: both backbones report `loaded: true`, `iterations: 50`, finite `final_loss`, identical `trainable_params`, plus `iters_per_sec` and `peak_mem_gb`. If the real SIN checkpoint fails under `weights_only=True`, apply the Task 6 Step 4 fallback, rerun the tests, and rerun this step.

- [ ] **Step 6: Bring results back and record G1**

```bash
scp asus@100.104.225.90:~/Shapes-n-Sizes/results/g1_smoke.json results/g1_smoke.json
```

In `README.md`, replace the status bullet with:

```markdown
- **Status:** G1 passed (GX10 smoke test, [`results/g1_smoke.json`](results/g1_smoke.json)); next: G2 data download and build ([`docs/runbook.md`](docs/runbook.md)). Implementation plan: [`docs/superpowers/plans/2026-10-11-snz-training-pipeline.md`](docs/superpowers/plans/2026-10-11-snz-training-pipeline.md).
```

Replace the `# planned:` block in README §8 with the actual layout: `src/snz/`, `configs/`, `manifests/`, `results/`, `scripts/`, `tests/`, `docs/runbook.md`.

- [ ] **Step 7: Commit and push**

```bash
git add results/g1_smoke.json README.md docs/runbook.md
git commit -m "G1: GX10 smoke test passed; README status"
git pull --rebase origin main && git push origin main
```

G2 then starts with the user. Reading and accepting the BDD100K terms (`BDD_TERMS_ACCEPTED=yes`) is their decision.
