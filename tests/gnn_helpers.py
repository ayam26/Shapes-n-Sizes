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
