"""Detection selection, box-prompted segmentation, and the OutlineSet store (spec §5.1).

Every image of the input appears in `names`, including images with no selected detection,
so re-scored files cover the whole split.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from snz.data.bdd import EVAL_CLASSES
from snz.fsutil import atomic_replace
from snz.gnn.graph import N_POINTS, box_contour, mask_to_contour

PATCH = 2  # 5x5 color patch around each contour point


def select_detections(record: dict, min_score: float = 0.05, top_k: int = 50) -> list[int]:
    idx = [i for i, (lab, s) in enumerate(zip(record["labels"], record["scores"], strict=True))
           if lab in EVAL_CLASSES and s >= min_score]
    idx.sort(key=lambda i: -record["scores"][i])  # stable: ties keep detection order
    return idx[:top_k]


class BoxSegmenter(Protocol):
    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray: ...


class Sam2Segmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda",
                 checkpoint: Path | None = None):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch, self.device = torch, device
        if checkpoint is None:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        else:
            from sam2.build_sam import build_sam2

            self.predictor = SAM2ImagePredictor(
                build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(checkpoint), device=device))

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device.split(":")[0],
                                                    dtype=torch.bfloat16,
                                                    enabled=self.device.startswith("cuda")):
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(box=np.asarray(boxes, np.float32),
                                                 multimask_output=False)
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks.reshape(len(boxes), *image.shape[:2]) > 0


class GrabCutSegmenter:
    def __init__(self, max_side: int = 128, iters: int = 5):
        self.max_side, self.iters = max_side, iters

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        out = np.zeros((len(boxes), h, w), bool)
        for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, float)):
            mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            cx1, cy1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
            cx2, cy2 = min(int(np.ceil(x2 + mx)), w), min(int(np.ceil(y2 + my)), h)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.shape[0] < 3 or crop.shape[1] < 3:
                continue
            f = min(1.0, self.max_side / max(crop.shape[:2]))
            size = (max(int(crop.shape[1] * f), 3), max(int(crop.shape[0] * f), 3))
            small = cv2.resize(crop, size)
            rect = (int((x1 - cx1) * f), int((y1 - cy1) * f),
                    max(int((x2 - x1) * f), 1), max(int((y2 - y1) * f), 1))
            mask = np.zeros(small.shape[:2], np.uint8)
            try:
                cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect,
                            np.zeros((1, 65)), np.zeros((1, 65)), self.iters,
                            cv2.GC_INIT_WITH_RECT)
            except cv2.error:
                continue
            fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
            out[k, cy1:cy2, cx1:cx2] = cv2.resize(fg, (crop.shape[1], crop.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST) > 0
        return out


@dataclass
class OutlineSet:
    names: list[str]
    image: np.ndarray      # int32 [M] index into names
    det_index: np.ndarray  # int32 [M] index into that image's detection record
    box: np.ndarray        # float32 [M, 4]
    score: np.ndarray      # float32 [M]
    label: list[str]
    contour: np.ndarray    # float32 [M, 64, 2]
    color: np.ndarray      # uint8 [M, 64, 3] (GNN-COLOR only)
    ok: np.ndarray         # bool [M]: False = box rectangle used
    image_wh: np.ndarray   # int32 [M, 2]


def _colors(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    out = np.zeros((len(contour), 3), np.uint8)
    for k, (x, y) in enumerate(contour):
        xi, yi = int(round(x)), int(round(y))
        patch = image[max(yi - PATCH, 0):min(yi + PATCH + 1, h),
                      max(xi - PATCH, 0):min(xi + PATCH + 1, w)]
        if patch.size:
            out[k] = patch.reshape(-1, 3).mean(axis=0)
    return out


def extract_outlines(records: Sequence[dict], image_dir: Path, segmenter: BoxSegmenter,
                     file_name: Callable[[str], str] | None = None,
                     progress: Callable[[int, int], None] | None = None) -> OutlineSet:
    rows = {k: [] for k in ("image", "det", "box", "score", "label", "contour", "color", "ok",
                            "wh")}
    names = []
    for i, rec in enumerate(records):
        names.append(rec["name"])
        sel = select_detections(rec)
        if sel:
            path = Path(image_dir) / (file_name(rec["name"]) if file_name else rec["name"])
            image = np.asarray(Image.open(path).convert("RGB"))
            boxes = np.asarray([rec["boxes"][j] for j in sel], np.float32)
            masks = segmenter(image, boxes)
            for j, box, mask in zip(sel, boxes, masks, strict=True):
                c = mask_to_contour(mask)
                ok = c is not None
                c = c if ok else box_contour(box)
                rows["image"].append(i)
                rows["det"].append(j)
                rows["box"].append(box)
                rows["score"].append(rec["scores"][j])
                rows["label"].append(rec["labels"][j])
                rows["contour"].append(c)
                rows["color"].append(_colors(image, c))
                rows["ok"].append(ok)
                rows["wh"].append((image.shape[1], image.shape[0]))
        if progress:
            progress(i + 1, len(records))
    m = len(rows["image"])
    return OutlineSet(
        names=names, image=np.asarray(rows["image"], np.int32),
        det_index=np.asarray(rows["det"], np.int32),
        box=np.asarray(rows["box"], np.float32).reshape(m, 4),
        score=np.asarray(rows["score"], np.float32), label=rows["label"],
        contour=np.asarray(rows["contour"], np.float32).reshape(m, N_POINTS, 2),
        color=np.asarray(rows["color"], np.uint8).reshape(m, N_POINTS, 3),
        ok=np.asarray(rows["ok"], bool), image_wh=np.asarray(rows["wh"], np.int32).reshape(m, 2))


def save_outlines(path: Path, o: OutlineSet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, names=np.asarray(o.names, dtype=str), image=o.image,
                                det_index=o.det_index, box=o.box, score=o.score,
                                label=np.asarray(o.label, dtype=str), contour=o.contour,
                                color=o.color, ok=o.ok, image_wh=o.image_wh)
        atomic_replace(tmp, path)
    finally:
        if tmp.exists():
            os.remove(tmp)


def load_outlines(path: Path) -> OutlineSet:
    z = np.load(path, allow_pickle=False)
    return OutlineSet(names=[str(n) for n in z["names"]], image=z["image"],
                      det_index=z["det_index"], box=z["box"], score=z["score"],
                      label=[str(x) for x in z["label"]], contour=z["contour"], color=z["color"],
                      ok=z["ok"], image_wh=z["image_wh"])
