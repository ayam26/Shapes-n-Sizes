"""Vectorized COCO AP from precomputed matches, with bootstrap image weights (spec §7.1).

Reproduces pycocotools' accumulate (101-point interpolated precision, ignored detections
excluded) from a `Matches` object. `weights[i]` is how many times image i is drawn in a
bootstrap replicate: detections and ground truth of that image count `weights[i]` times.
Sorting is done once per class, so each replicate is only weighted cumulative sums.
"""

from __future__ import annotations

import numpy as np

from snz.eval.matching import Matches

RECALL_POINTS = np.linspace(0.0, 1.0, 101)
_EPS = np.spacing(1)


class APCalculator:
    def __init__(self, m: Matches):
        self.m = m
        self.per_class = []
        for k in range(len(m.classes)):
            sel = np.flatnonzero(m.det_class == k)
            order = sel[np.argsort(-m.det_score[sel], kind="mergesort")]
            valid = ~m.det_ignored[order]
            self.per_class.append((
                m.det_image[order],
                (m.det_tp[order] & valid).astype(float),
                (~m.det_tp[order] & valid).astype(float),
            ))

    def precision(self, weights: np.ndarray | None = None) -> np.ndarray:
        """[T, R, K] interpolated precision; NaN for classes without ground truth."""
        m = self.m
        w_img = np.ones(len(m.image_names)) if weights is None else np.asarray(weights, float)
        t, k_n = m.det_tp.shape[1], len(m.classes)
        out = np.full((t, len(RECALL_POINTS), k_n), np.nan)
        for k, (image, tp, fp) in enumerate(self.per_class):
            npig = float(m.gt_count[:, k] @ w_img)
            if npig == 0:
                continue
            q = np.zeros((t, len(RECALL_POINTS)))
            if len(image):
                w = w_img[image][:, None]
                tps, fps = np.cumsum(tp * w, axis=0), np.cumsum(fp * w, axis=0)
                rc = tps / npig
                pr = tps / (tps + fps + _EPS)
                pr = np.maximum.accumulate(pr[::-1], axis=0)[::-1]
                n = len(image)
                for ti in range(t):
                    idx = np.searchsorted(rc[:, ti], RECALL_POINTS, side="left")
                    ok = idx < n
                    q[ti, ok] = pr[idx[ok], ti]
            out[:, :, k] = q
        return out

    def __call__(self, weights: np.ndarray | None = None) -> dict:
        p = self.precision(weights)
        has_gt = ~np.isnan(p[0, 0])
        per_class = {c: (float(np.mean(p[:, :, k])) if has_gt[k] else None)
                     for k, c in enumerate(self.m.classes)}
        if not has_gt.any():
            return {"mAP": 0.0, "AP50": 0.0, "per_class": per_class}
        return {"mAP": float(np.mean(p[:, :, has_gt])), "AP50": float(np.mean(p[0][:, has_gt])),
                "per_class": per_class}


def average_precision(m: Matches, weights: np.ndarray | None = None) -> dict:
    return APCalculator(m)(weights)
