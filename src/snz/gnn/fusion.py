"""Fusion of detector score and second-stage probabilities (spec §5.3): IRLS logistic."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from snz.gnn import BACKGROUND

FEATURES = ("logit_det_score", "log_p_det_class", "log_p_background", "log_max_p_other_class")


def fusion_features(det_score: np.ndarray, probs: np.ndarray, label_idx: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(det_score, float), 1e-6, 1 - 1e-6)
    p = np.clip(np.asarray(probs, float), 1e-6, 1.0)
    rows = np.arange(len(p))
    other = p[:, :BACKGROUND].copy()
    other[rows, label_idx] = 0.0
    return np.column_stack([np.log(s / (1 - s)), np.log(p[rows, label_idx]),
                            np.log(p[:, BACKGROUND]), np.log(np.clip(other.max(axis=1), 1e-6, 1))])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-4, iters: int = 50) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    w = np.zeros(X1.shape[1])
    pen = np.full(X1.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        p = _sigmoid(X1 @ w)
        g = X1.T @ (p - y) + pen * w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.clip(_sigmoid(w[0] + np.asarray(X, float) @ np.asarray(w[1:])), 1e-12, 1 - 1e-12)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if not n_pos or not n_neg:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
