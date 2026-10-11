"""Convert BDD100K 2018 detection labels to the det_20 file layout.

The official det_20 labels could not be downloaded in 2026 (BDD servers offline); the
maintainer's Berkeley mirror (dl.yf.io) still serves `bdd100k_labels.zip`: one JSON per
image under `100k/<split>/`, objects in `frames[0].objects`, names without `.jpg`, and
2018 class names (person/bike/motor, which `load_frames` maps). Only box labels are kept.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def convert_2018(zip_path: Path, split: str) -> list[dict]:
    prefix = f"100k/{split}/"
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for member in sorted(n for n in z.namelist()
                             if n.startswith(prefix) and n.endswith(".json")):
            item = json.loads(z.read(member))
            objects = item["frames"][0].get("objects") or [] if item.get("frames") else []
            name = item["name"] if item["name"].endswith(".jpg") else item["name"] + ".jpg"
            frames.append({
                "name": name,
                "attributes": item.get("attributes") or {},
                "labels": [{"id": str(o.get("id")), "category": o["category"],
                            "box2d": o["box2d"]} for o in objects if "box2d" in o],
            })
    return frames


def write_det20_layout(zip_path: Path, out_dir: Path) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        frames = convert_2018(zip_path, split)
        (out_dir / f"det_{split}.json").write_text(json.dumps(frames))
        counts[split] = len(frames)
    return counts
