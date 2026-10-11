"""Per-image COCO-style matching, computed once per run and split (spec §7.1, §7.3).

Matching follows pycocotools (bbox, area "all"): per image and class, detections in
descending score order (stable), at most `max_dets`, each greedily takes the unmatched
ground truth with the highest IoU >= threshold, preferring real boxes over ignore
regions; ignore regions (BDD "other vehicle" etc.) use IoU = intersection / detection
area and may absorb any number of detections, which then count as neither TP nor FP.
Detections of classes outside `classes` (e.g. `train`) are left out entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from snz.data.bdd import EVAL_CLASSES, Frame

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
FD_TP, FD_DUPLICATE, FD_WRONG_CLASS, FD_POOR_LOC, FD_INVENTED, FD_NEAR_OTHER, FD_IGNORED = range(7)
FD_NAMES = ("tp", "duplicate", "wrong_class", "poor_localization", "invented",
            "near_other_class", "ignore_region")
FALSE_TYPES = (FD_DUPLICATE, FD_WRONG_CLASS, FD_POOR_LOC, FD_INVENTED, FD_NEAR_OTHER)


def _area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = _intersection(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _coco_iou(dets: np.ndarray, gts: np.ndarray, crowd: np.ndarray) -> np.ndarray:
    iou = box_iou(dets, gts)
    if crowd.any():
        inter = _intersection(dets, gts[crowd])
        area = _area(dets)[:, None]
        iou[:, crowd] = np.divide(inter, area, out=np.zeros_like(inter), where=area > 0)
    return iou


@dataclass
class Matches:
    classes: tuple[str, ...]
    image_names: list[str]
    det_image: np.ndarray    # [N] index into image_names
    det_class: np.ndarray    # [N] index into classes
    det_score: np.ndarray    # [N]
    det_tp: np.ndarray       # [N, T] bool
    det_ignored: np.ndarray  # [N, T] bool: matched an ignore region
    det_fd_type: np.ndarray  # [N] FD_* code at IoU 0.5
    gt_count: np.ndarray     # [n_images, K] real (non-ignore) ground-truth boxes

    def subset(self, names: Sequence[str]) -> Matches:
        """Matches restricted to `names` (in that image order; detections keep their order)."""
        index = {n: i for i, n in enumerate(self.image_names)}
        old = np.array([index[n] for n in names], dtype=np.int64)  # KeyError if unknown
        remap = np.full(len(self.image_names), -1, dtype=np.int64)
        remap[old] = np.arange(len(old))
        keep = remap[self.det_image] >= 0
        return Matches(
            self.classes, list(names), remap[self.det_image[keep]], self.det_class[keep],
            self.det_score[keep], self.det_tp[keep], self.det_ignored[keep],
            self.det_fd_type[keep], self.gt_count[old],
        )


def _greedy(iou: np.ndarray, crowd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COCO greedy matching for one image and class, all IoU thresholds at once."""
    n, g = iou.shape
    t = len(IOU_THRESHOLDS)
    tp = np.zeros((n, t), bool)
    ignored = np.zeros((n, t), bool)
    if g == 0:
        return tp, ignored
    matched = np.zeros((t, g), bool)
    thr = np.minimum(IOU_THRESHOLDS, 1 - 1e-10)[:, None]
    rows = np.arange(t)
    for j in range(n):
        cand = (iou[j][None, :] >= thr) & (~matched | crowd[None, :])
        real = cand & ~crowd[None, :]
        # pycocotools keeps the last of equal IoUs, hence argmax over the reversed axis
        pick_real = g - 1 - np.argmax(np.where(real, iou[j], -1)[:, ::-1], axis=1)
        has_real, has_crowd = real.any(axis=1), (cand & crowd[None, :]).any(axis=1)
        tp[j] = has_real
        ignored[j] = ~has_real & has_crowd
        matched[rows[has_real], pick_real[has_real]] = True
    return tp, ignored


def match_split(
    frames: Sequence[Frame],
    detections: Sequence[dict],
    classes: Sequence[str] = EVAL_CLASSES,
    max_dets: int = 100,
) -> Matches:
    classes = tuple(classes)
    names = [f.name for f in frames]
    known = set(names)
    unknown = [d["name"] for d in detections if d["name"] not in known]
    if unknown:
        raise ValueError(f"{len(unknown)} detection record(s) for images not in this split, "
                         f"e.g. {unknown[:3]}")
    dets_by = {d["name"]: d for d in detections}
    cls_index = {c: k for k, c in enumerate(classes)}
    out = {k: [] for k in ("image", "cls", "score", "tp", "ign", "fd")}
    gt_count = np.zeros((len(frames), len(classes)), np.int64)

    for i, f in enumerate(frames):
        gt_box = np.array([[b.x1, b.y1, b.x2, b.y2] for b in f.boxes], float).reshape(-1, 4)
        gt_cat = np.array([b.category for b in f.boxes], dtype=object)
        gt_ign = np.array([b.ignore for b in f.boxes], bool)
        for k, c in enumerate(classes):
            gt_count[i, k] = int(((gt_cat == c) & ~gt_ign).sum())
        d = dets_by.get(f.name)
        if d is None or not d["labels"]:
            continue
        d_box = np.asarray(d["boxes"], float).reshape(-1, 4)
        d_lab = np.array(d["labels"], dtype=object)
        d_score = np.asarray(d["scores"], float)
        real_iou_all = box_iou(d_box, gt_box[~gt_ign])  # vs every real box, any class
        real_cat = gt_cat[~gt_ign]
        any_iou = box_iou(d_box, gt_box)  # incl. ignore regions: "every ground truth" (§7.3)
        for c, k in cls_index.items():
            sel = np.flatnonzero(d_lab == c)
            if not sel.size:
                continue
            sel = sel[np.argsort(-d_score[sel], kind="mergesort")][:max_dets]
            gmask = gt_cat == c
            iou = _coco_iou(d_box[sel], gt_box[gmask], gt_ign[gmask])
            tp, ign = _greedy(iou, gt_ign[gmask])
            same = real_iou_all[sel][:, real_cat == c]
            other = real_iou_all[sel][:, real_cat != c]
            best_same = np.maximum(same.max(axis=1, initial=0.0),
                                   any_iou[sel][:, gt_cat == c].max(axis=1, initial=0.0))
            best_other = other.max(axis=1, initial=0.0)
            best_any = any_iou[sel].max(axis=1, initial=0.0)
            fd = np.full(len(sel), FD_NEAR_OTHER, np.int8)
            fd[best_any < 0.1] = FD_INVENTED
            fd[(best_same >= 0.1) & (best_same < 0.5) & (best_other < 0.5)] = FD_POOR_LOC
            fd[best_other >= 0.5] = FD_WRONG_CLASS
            fd[same.max(axis=1, initial=0.0) >= 0.5] = FD_DUPLICATE  # real box already taken
            fd[ign[:, 0]] = FD_IGNORED
            fd[tp[:, 0]] = FD_TP
            out["image"].append(np.full(len(sel), i))
            out["cls"].append(np.full(len(sel), k))
            out["score"].append(d_score[sel])
            out["tp"].append(tp)
            out["ign"].append(ign)
            out["fd"].append(fd)

    def cat(key, empty):
        return np.concatenate(out[key]) if out[key] else empty

    t = len(IOU_THRESHOLDS)
    return Matches(
        classes, names,
        cat("image", np.zeros(0, np.int64)).astype(np.int64),
        cat("cls", np.zeros(0, np.int64)).astype(np.int64),
        cat("score", np.zeros(0)),
        cat("tp", np.zeros((0, t), bool)),
        cat("ign", np.zeros((0, t), bool)),
        cat("fd", np.zeros(0, np.int8)),
        gt_count,
    )
