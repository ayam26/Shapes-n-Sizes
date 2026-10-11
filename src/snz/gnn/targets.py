"""id-val folds and second-stage targets (spec §4)."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from snz.data.bdd import EVAL_CLASSES, Frame
from snz.eval.matching import box_iou, match_split
from snz.gnn import BACKGROUND
from snz.gnn.outlines import OutlineSet


def fold_split(names: Sequence[str], seed: int = 0) -> tuple[list[str], list[str]]:
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)
    half = (len(shuffled) + 1) // 2
    return sorted(shuffled[:half]), sorted(shuffled[half:])


def network_targets(o: OutlineSet, frames: Sequence[Frame]) -> np.ndarray:
    """0-8 = class, 9 = background, -1 = excluded from network training."""
    by_name = {f.name: f for f in frames}
    out = np.full(len(o.score), -1, np.int64)
    for r in range(len(o.score)):
        f = by_name[o.names[o.image[r]]]
        if not f.boxes:
            out[r] = BACKGROUND
            continue
        gt = np.array([[b.x1, b.y1, b.x2, b.y2] for b in f.boxes], float)
        iou = box_iou(o.box[r:r + 1].astype(float), gt)[0]
        real = np.array([not b.ignore for b in f.boxes])
        if (iou * real).max(initial=0) >= 0.5:
            best = f.boxes[int(np.argmax(np.where(real, iou, -1)))]
            out[r] = EVAL_CLASSES.index(best.category) if best.category in EVAL_CLASSES else -1
        elif iou.max(initial=0) < 0.3:
            out[r] = BACKGROUND
    return out


def outline_records(o: OutlineSet, scores: Sequence[float] | None = None) -> list[dict]:
    """prereg-v1 detection records for every image in `o.names` (selected detections only)."""
    scores = o.score if scores is None else np.asarray(scores, float)
    recs = [{"name": n, "boxes": [], "labels": [], "scores": []} for n in o.names]
    for r in range(len(o.score)):
        rec = recs[o.image[r]]
        rec["boxes"].append([float(v) for v in o.box[r]])
        rec["labels"].append(o.label[r])
        rec["scores"].append(float(scores[r]))
    return recs


def tp_targets(o: OutlineSet, frames: Sequence[Frame]) -> np.ndarray:
    """True positive at IoU 0.5 under prereg-v1 matching, per outline row."""
    by_name = {f.name: f for f in frames}
    m = match_split([by_name[n] for n in o.names], outline_records(o))
    # match_split orders rows by image, then class (EVAL_CLASSES order), then score (stable)
    cls = np.array([EVAL_CLASSES.index(lab) for lab in o.label], np.int64)
    order = sorted(range(len(o.score)), key=lambda r: (o.image[r], cls[r], -o.score[r], r))
    out = np.zeros(len(o.score), bool)
    out[np.asarray(order, np.int64)] = m.det_tp[:, 0]
    return out
