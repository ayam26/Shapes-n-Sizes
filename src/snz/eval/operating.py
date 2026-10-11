"""Equal-recall operating points and false detections per image (spec §7.3).

Recall is pooled over classes at IoU 0.5: matched detections with score >= threshold,
divided by all real ground-truth boxes. A run's operating point is the highest
threshold whose recall reaches the target; a run that never reaches it is taken at its
maximum recall (the highest threshold attaining it) and flagged. Weights are bootstrap
image multiplicities, so thresholds can be re-estimated in every replicate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snz.eval.matching import FALSE_TYPES, FD_NAMES, Matches


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    recall: float
    flagged: bool


def _w(m: Matches, weights: np.ndarray | None) -> np.ndarray:
    return np.ones(len(m.image_names)) if weights is None else np.asarray(weights, float)


def recall_at(m: Matches, threshold: float, weights: np.ndarray | None = None) -> float:
    w = _w(m, weights)
    n_gt = float(m.gt_count.sum(axis=1) @ w)
    hit = (m.det_score >= threshold) & m.det_tp[:, 0]
    return float(w[m.det_image[hit]].sum() / n_gt) if n_gt else 0.0


class RecallCurve:
    """Score-sorted detections of one run, reused across bootstrap replicates."""

    def __init__(self, m: Matches):
        self.m = m
        order = np.argsort(-m.det_score, kind="mergesort")
        self.scores = m.det_score[order]
        self.image = m.det_image[order]
        self.tp = m.det_tp[order, 0].astype(float)
        # recall at "score >= s" includes every tie of s: read it at the last index of a tie
        self.last_of_tie = np.r_[self.scores[1:] != self.scores[:-1], True]
        self.gt_per_image = m.gt_count.sum(axis=1)

    def threshold(self, target: float, weights: np.ndarray | None = None) -> OperatingPoint:
        w = _w(self.m, weights)
        n_gt = float(self.gt_per_image @ w)
        if not len(self.scores) or n_gt == 0:
            return OperatingPoint(float("inf"), 0.0, True)
        recall = np.cumsum(self.tp * w[self.image]) / n_gt
        ok = self.last_of_tie
        reach = np.flatnonzero(ok & (recall >= target - 1e-12))
        if reach.size:
            i = reach[0]
            return OperatingPoint(float(self.scores[i]), float(recall[i]), False)
        best = recall[ok].max()
        i = np.flatnonzero(ok & (recall >= best - 1e-12))[0]
        return OperatingPoint(float(self.scores[i]), float(recall[i]), True)


def threshold_for_recall(m: Matches, target: float,
                         weights: np.ndarray | None = None) -> OperatingPoint:
    return RecallCurve(m).threshold(target, weights)


def false_detections(
    m: Matches, threshold: float, weights: np.ndarray | None = None
) -> dict:
    w = _w(m, weights)
    n_images = float(w.sum())
    keep = m.det_score >= threshold
    counts = {}
    for code in FALSE_TYPES:
        sel = keep & (m.det_fd_type == code)
        counts[FD_NAMES[code]] = float(w[m.det_image[sel]].sum())
    total = sum(counts.values())
    return {"per_image": total / n_images, "count": total,
            "by_type": {k: v / n_images for k, v in counts.items()}, "counts_by_type": counts}
