"""Scoring the blind audit and the hand-labeled footage frames (spec §6)."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from snz.data.bdd import ALIASES, CLASSES, IGNORED_MAPPING, Box, Frame
from snz.eval.matching import match_split
from snz.eval.operating import false_detections, recall_at

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None, float | None]:
    if n == 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def read_labels(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["item_id"]: r["label"] for r in csv.DictReader(f)}


def _scaled(total: int, k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    if p is None:
        return {"estimate": None, "ci95": [None, None]}
    return {"estimate": total * p, "ci95": [total * lo, total * hi]}


def score_audit(labels: Mapping[str, str], key: Mapping[str, dict],
                total_detections: Mapping[str, int]) -> dict:
    out = {}
    for arm in sorted({v["arm"] for v in key.values()}):
        ids = [i for i, v in key.items() if v["arm"] == arm]
        counts = Counter(labels[i] for i in ids if i in labels)
        n = sum(counts[c] for c in ("correct", "wrong_class", "invented"))
        p, lo, hi = wilson(counts["correct"], n)
        q, qlo, qhi = wilson(counts["invented"], n)
        total = total_detections[arm]
        out[arm] = {
            "items": len(ids), "unlabeled": len(ids) - sum(counts.values()),
            "unsure": counts["unsure"], "n": n, "counts": dict(counts),
            "precision": {"p": p, "ci95": [lo, hi]},
            "invented_share": {"p": q, "ci95": [qlo, qhi]},
            "total_detections": total,
            "est_real": _scaled(total, counts["correct"], n),
            "est_invented": _scaled(total, counts["invented"], n),
        }
    return out


def cohen_kappa(a: Mapping[str, str], b: Mapping[str, str]) -> float | None:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    po = sum(a[i] == b[i] for i in common) / len(common)
    ca, cb = Counter(a[i] for i in common), Counter(b[i] for i in common)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / len(common) ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_coco_labels(path: Path) -> list[Frame]:
    """Hand labels exported as COCO (CVAT / Label Studio) -> frames (no attributes)."""
    data = json.loads(Path(path).read_text())
    cats = {}
    for c in data["categories"]:
        name = ALIASES.get(c["name"], c["name"])
        if name not in CLASSES and name not in IGNORED_MAPPING:
            raise ValueError(f"label category {c['name']!r} is not a study class {CLASSES}")
        cats[c["id"]] = name
    boxes = {img["id"]: [] for img in data["images"]}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        name = cats[a["category_id"]]
        ignore = name in IGNORED_MAPPING or bool(a.get("iscrowd", 0))
        boxes[a["image_id"]].append(Box(IGNORED_MAPPING.get(name, name), x, y, x + w, y + h,
                                        ignore))
    return [Frame(Path(img["file_name"]).name, "undefined", "undefined",
                  tuple(boxes[img["id"]])) for img in data["images"]]


def footage_recall(frames: Sequence[Frame], dets_by_arm: Mapping[str, Sequence[dict]],
                   thresholds: Mapping[str, float]) -> dict:
    names = {f.name for f in frames}
    out = {}
    for arm, dets in dets_by_arm.items():
        m = match_split(frames, [d for d in dets if d["name"] in names])
        fd = false_detections(m, thresholds[arm])
        out[arm] = {"frames": len(frames), "recall": recall_at(m, thresholds[arm]),
                    "false_per_frame": fd["per_image"], "by_type": fd["by_type"]}
    return out
