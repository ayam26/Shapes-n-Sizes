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
