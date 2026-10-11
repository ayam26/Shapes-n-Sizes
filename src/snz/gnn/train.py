"""gnn-train: networks on fold A, fusion on fold B (spec §4, §5.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from snz.data.bdd import EVAL_CLASSES, load_frames
from snz.data.manifest import read_manifest, write_manifest
from snz.gnn import GNN_CLASSES
from snz.gnn.fusion import FEATURES, auroc, fit_logistic, fusion_features, predict_logistic
from snz.gnn.model import (
    CropDataset,
    GraphDataset,
    ShapeGNN,
    predict_probs,
    rgb_model,
    train_network,
)
from snz.gnn.outlines import OutlineSet, load_outlines
from snz.gnn.targets import fold_split, network_targets, tp_targets
from snz.hashing import sha256_file

GRAPH_ARMS = {"GNN": ("gnn", 5, True), "GNN-SHAPE": ("shape", 5, False),
              "GNN-COLOR": ("color", 8, True)}
GNN_CFG = {"epochs": 30, "lr": 1e-3, "batch_size": 512, "weight_decay": 1e-4}
RGB_CFG = {"epochs": 8, "lr": 3e-4, "batch_size": 256, "weight_decay": 1e-4}


def subset_outlines(o: OutlineSet, names: list[str]) -> OutlineSet:
    keep_names = set(names)
    new_names = [n for n in o.names if n in keep_names]
    index = {n: i for i, n in enumerate(new_names)}
    rows = np.asarray([r for r in range(len(o.score)) if o.names[o.image[r]] in keep_names],
                      np.int64)
    return OutlineSet(
        names=new_names, image=np.asarray([index[o.names[o.image[r]]] for r in rows], np.int32),
        det_index=o.det_index[rows], box=o.box[rows], score=o.score[rows],
        label=[o.label[r] for r in rows], contour=o.contour[rows], color=o.color[rows],
        ok=o.ok[rows], image_wh=o.image_wh[rows])


def make_dataset(arm: str, o: OutlineSet, targets: np.ndarray, *, image_dir: Path,
                 file_name=None, augment: bool):
    if arm == "RGB":
        return CropDataset(o, targets, image_dir, file_name, augment=augment)
    return GraphDataset(o, targets, GRAPH_ARMS[arm][0], augment)


def make_model(arm: str, pretrained: bool = True) -> torch.nn.Module:
    if arm == "RGB":
        return rgb_model(pretrained)
    _, node_dim, use_size = GRAPH_ARMS[arm]
    return ShapeGNN(node_dim=node_dim, use_size=use_size)


def label_index(o: OutlineSet) -> np.ndarray:
    return np.asarray([EVAL_CLASSES.index(lab) for lab in o.label], np.int64)


def _counts(t: np.ndarray) -> dict:
    out = {"excluded": int((t < 0).sum())}
    out.update({c: int((t == k).sum()) for k, c in enumerate(GNN_CLASSES)})
    return out


def train_arms(outlines: Path, labels: Path, manifest: Path, image_dir: Path, out: Path,
               arms: list[str], device: str, fold_dir: Path, pretrained: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    o = load_outlines(outlines)
    frames, _ = load_frames(labels)
    rows = read_manifest(manifest)
    fold_a, fold_b = fold_split([n for n, _ in rows])
    for tag, names in (("a", fold_a), ("b", fold_b)):
        keep = set(names)
        write_manifest(fold_dir / f"gnn_fold_{tag}.csv", [r for r in rows if r[0] in keep])
    oa, ob = subset_outlines(o, fold_a), subset_outlines(o, fold_b)
    ta, tb = network_targets(oa, frames), network_targets(ob, frames)
    tp_b = tp_targets(ob, frames)
    summary = {"counts": {"fold_a": _counts(ta), "fold_b": _counts(tb),
                          "fold_b_tp": int(tp_b.sum()), "fold_b_rows": int(len(tp_b))},
               "fold_b_auroc": {"BASE": auroc(tp_b, ob.score)}, "sha256": {},
               "config": {"gnn": GNN_CFG, "rgb": RGB_CFG, "fusion_features": list(FEATURES),
                          "arms": arms, "outlines": str(outlines),
                          "outlines_sha256": sha256_file(outlines)}}
    for arm in arms:
        cfg = RGB_CFG if arm == "RGB" else GNN_CFG
        torch.manual_seed(0)
        model = train_network(make_model(arm, pretrained),
                              make_dataset(arm, oa, ta, image_dir=image_dir, augment=True),
                              device=device, seed=0, **cfg)
        torch.save(model.state_dict(), out / f"{arm}.pt")
        probs = predict_probs(model, make_dataset(arm, ob, tb, image_dir=image_dir,
                                                  augment=False), device=device)
        X = fusion_features(ob.score, probs, label_index(ob))
        w = fit_logistic(X, tp_b)
        (out / f"{arm}.fusion.json").write_text(json.dumps({"w": w.tolist(),
                                                             "features": list(FEATURES)}))
        summary["fold_b_auroc"][arm] = auroc(tp_b, predict_logistic(w, X))
        summary["sha256"][f"{arm}.pt"] = sha256_file(out / f"{arm}.pt")
        (out / "gnn-train.json").write_text(json.dumps(summary, indent=2))  # partial progress
    (out / "gnn-train.json").write_text(json.dumps(summary, indent=2))
    return summary

