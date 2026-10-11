"""Outline geometry → shape-graph features (spec §5.2). No pixel values ever enter here."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

N_POINTS = 64
NEIGHBOR_OFFSETS = (1, -1, 2, -2, 4, -4)


def neighbor_index(n: int = N_POINTS) -> np.ndarray:
    return np.stack([(np.arange(n) + o) % n for o in NEIGHBOR_OFFSETS], axis=1).astype(np.int64)


def _shoelace(c: np.ndarray) -> float:
    x, y = c[:, 0], c[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    """n points equally spaced by arc length along a closed polygon."""
    nxt = np.roll(poly, -1, axis=0)
    seg = np.linalg.norm(nxt - poly, axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = cum[-1]
    if total <= 0:
        return np.repeat(poly[:1], n, axis=0).astype(np.float32)
    t = np.linspace(0.0, total, n, endpoint=False)
    i = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(poly) - 1)
    frac = np.divide(t - cum[i], seg[i], out=np.zeros_like(t), where=seg[i] > 0)
    return (poly[i] + frac[:, None] * (nxt[i] - poly[i])).astype(np.float32)


def _orient(c: np.ndarray) -> np.ndarray:
    """Clockwise on screen (positive shoelace with y down), starting at the top-most point."""
    if _shoelace(c) < 0:
        c = c[::-1]
    start = np.lexsort((c[:, 0], c[:, 1]))[0]
    return np.roll(c, -start, axis=0).astype(np.float32)


def mask_to_contour(mask: np.ndarray, n: int = N_POINTS) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    if len(big) < 3 or cv2.contourArea(big) <= 0:
        return None
    return _orient(_resample(big[:, 0, :].astype(np.float64), n))


def box_contour(box: Sequence[float], n: int = N_POINTS) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return _orient(_resample(rect, n))


def node_features(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    centroid = c.mean(axis=0)
    s = np.sqrt(max(abs(_shoelace(c)), 1e-6))
    rel = (c - centroid) / s
    e = np.roll(c, -1, axis=0) - c
    prev = np.roll(e, 1, axis=0)
    turn = np.arctan2(prev[:, 0] * e[:, 1] - prev[:, 1] * e[:, 0],
                      (prev * e).sum(axis=1)) / np.pi
    perimeter = np.linalg.norm(e, axis=1).sum()
    curvature = turn * len(c) / max(perimeter / s, 1e-6)
    radius = np.linalg.norm(c - centroid, axis=1) / s
    out = np.column_stack([rel, turn, curvature, radius]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def size_features(box: Sequence[float], contour: np.ndarray,
                  image_wh: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    w_img, h_img = image_wh
    w, h = x2 - x1, y2 - y1
    fill = abs(_shoelace(np.asarray(contour, np.float64))) / max(w * h, 1e-6)
    return np.array([w / w_img, h / h_img, y2 / h_img, (x1 + x2) / 2 / w_img,
                     np.log(max(w, 1e-3) / max(h, 1e-3)), np.clip(fill, 0, 1)], np.float32)
