"""Shared fixtures for the GNN tests."""

from __future__ import annotations

import numpy as np

from snz.gnn.graph import box_contour
from snz.gnn.outlines import OutlineSet


class FakeSegmenter:
    """Fills each box (clipped to the image)."""

    def __call__(self, image, boxes):
        h, w = image.shape[:2]
        masks = np.zeros((len(boxes), h, w), bool)
        for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes)):
            masks[k, max(int(y1), 0):min(int(np.ceil(y2)), h),
                  max(int(x1), 0):min(int(np.ceil(x2)), w)] = True
        return masks


class EmptySegmenter:
    def __call__(self, image, boxes):
        return np.zeros((len(boxes), *image.shape[:2]), bool)


def outline_set_for(name, boxes, labels=None, scores=None, extra_names=(), image_wh=(1280, 720)):
    m = len(boxes)
    labels = labels or ["car"] * m
    scores = scores if scores is not None else [0.9 - 0.01 * k for k in range(m)]
    return OutlineSet(
        names=[name, *extra_names],
        image=np.zeros(m, np.int32), det_index=np.arange(m, dtype=np.int32),
        box=np.asarray(boxes, np.float32).reshape(-1, 4), score=np.asarray(scores, np.float32),
        label=list(labels),
        contour=np.stack([box_contour(b) for b in boxes]).astype(np.float32)
        if m else np.zeros((0, 64, 2), np.float32),
        color=np.zeros((m, 64, 3), np.uint8), ok=np.ones(m, bool),
        image_wh=np.tile(np.asarray(image_wh, np.int32), (m, 1)))


def synthetic_outlines(color=(255, 0, 0), n=3):
    o = outline_set_for("a.jpg", boxes=[[10 + 5 * k, 20, 60 + 5 * k, 50] for k in range(n)])
    o.color[:] = np.asarray(color, np.uint8)
    return o


def squares_and_circles(n=200, seed=0):
    """Outlines of squares (target 0) and circles (target 1) at random sizes and positions."""
    rng = np.random.default_rng(seed)
    contours, boxes, targets = [], [], []
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for i in range(n):
        cx, cy, r = rng.uniform(100, 600), rng.uniform(100, 400), rng.uniform(10, 80)
        if i % 2 == 0:
            c = box_contour([cx - r, cy - r, cx + r, cy + r])
        else:
            c = np.column_stack([cx + r * np.sin(t), cy - r * np.cos(t)]).astype(np.float32)
        contours.append(c)
        boxes.append([cx - r, cy - r, cx + r, cy + r])
        targets.append(i % 2)
    o = outline_set_for("a.jpg", boxes=boxes)
    o.contour = np.asarray(contours, np.float32)
    return o, np.asarray(targets, np.int64)


def tiny_idval(tmp_path, n_images=8):
    """id-val-like fixture: images, labels JSON, manifest, and an outlines npz."""
    import json

    from helpers import write_image
    from snz.data.manifest import write_manifest
    from snz.gnn.outlines import extract_outlines, save_outlines

    imgs = tmp_path / "imgs"
    frames, recs = [], []
    rng = np.random.default_rng(0)
    for i in range(n_images):
        name = f"f{i}.jpg"
        write_image(imgs / name, size=(128, 96), color=(40 + 20 * i, 90, 120))
        gt = [("car", 10, 10, 50, 40), ("pedestrian", 70, 20, 85, 70)]
        frames.append({"name": name, "attributes": {"weather": "clear", "timeofday": "daytime"},
                       "labels": [{"id": str(k), "category": c,
                                   "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
                                  for k, (c, x1, y1, x2, y2) in enumerate(gt)]})
        boxes = [[10, 10, 50, 40], [70, 20, 85, 70], [90, 60, 120, 90], [5, 60, 30, 85]]
        recs.append({"name": name, "boxes": boxes,
                     "labels": ["car", "pedestrian", "car", "truck"],
                     "scores": [float(x) for x in rng.uniform(0.1, 0.95, 4)]})
    labels = tmp_path / "det_train.json"
    labels.write_text(json.dumps(frames))
    manifest = tmp_path / "manifests" / "id-val.csv"
    write_manifest(manifest, [(f["name"], "h") for f in frames])
    outlines = tmp_path / "id-val.npz"
    save_outlines(outlines, extract_outlines(recs, imgs, FakeSegmenter()))
    return imgs, labels, manifest, outlines
