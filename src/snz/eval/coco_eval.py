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
