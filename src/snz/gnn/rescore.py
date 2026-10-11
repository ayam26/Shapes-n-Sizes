"""Per-arm re-scoring of the selected BASE-s0 detections (spec §5.4).

Boxes and labels never change; BASE keeps the detector score; other arms get the fused
P(true positive). Every image of the split is written, including empty ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from snz.gnn.fusion import fusion_features, predict_logistic
from snz.gnn.model import predict_probs
from snz.gnn.outlines import OutlineSet
from snz.gnn.targets import outline_records
from snz.gnn.train import label_index, make_dataset, make_model, subset_outlines


def rescore(o: OutlineSet, arm: str, models_dir: Path, *, image_dir: Path,
            file_name: Callable[[str], str] | None = None, device: str,
            chunk: int = 1000) -> list[dict]:
    if arm == "BASE":
        return outline_records(o)
    model = make_model(arm, pretrained=False)
    model.load_state_dict(torch.load(Path(models_dir) / f"{arm}.pt", map_location="cpu",
                                     weights_only=True))
    w = np.asarray(json.loads((Path(models_dir) / f"{arm}.fusion.json").read_text())["w"])
    scores = np.zeros(len(o.score))
    for start in range(0, len(o.names), chunk):  # chunks bound memory (RGB caches crops)
        names = o.names[start:start + chunk]
        part = subset_outlines(o, names)
        if not len(part.score):
            continue
        rows = np.flatnonzero((o.image >= start) & (o.image < start + len(names)))
        ds = make_dataset(arm, part, np.zeros(len(part.score), np.int64), image_dir=image_dir,
                          file_name=file_name, augment=False)
        probs = predict_probs(model, ds, device=device)
        scores[rows] = predict_logistic(w, fusion_features(part.score, probs, label_index(part)))
    return outline_records(o, scores)
