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
