"""Equal-recall operating points and false detections per image (spec §7.3).

Recall is pooled over classes at IoU 0.5: matched detections with score >= threshold,
divided by all real ground-truth boxes. A run's operating point is the highest
threshold whose recall reaches the target; a run that never reaches it is taken at its
maximum recall (lowest score) and flagged.
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


def threshold_for_recall(m: Matches, target: float) -> OperatingPoint:
    n_gt = float(m.gt_count.sum())
    if not len(m.det_score) or n_gt == 0:
        return OperatingPoint(float("inf"), 0.0, True)
    order = np.argsort(-m.det_score, kind="mergesort")
    scores = m.det_score[order]
    recall = np.cumsum(m.det_tp[order, 0]) / n_gt
    # recall at "score >= s" includes every tie of s: evaluate at the last index of each score
    last_of_tie = np.r_[scores[1:] != scores[:-1], True]
    reach = np.flatnonzero(last_of_tie & (recall >= target - 1e-12))
    if reach.size:
        i = reach[0]
        return OperatingPoint(float(scores[i]), float(recall[i]), False)
    return OperatingPoint(float(scores[-1]), float(recall[-1]), True)


def false_detections(
    m: Matches, threshold: float, weights: np.ndarray | None = None
) -> dict:
    w = _w(m, weights)
    n_images = float(w.sum())
    keep = m.det_score >= threshold
    by_type = {}
    for code in FALSE_TYPES:
        sel = keep & (m.det_fd_type == code)
        by_type[FD_NAMES[code]] = float(w[m.det_image[sel]].sum() / n_images)
    return {"per_image": float(sum(by_type.values())), "by_type": by_type}
