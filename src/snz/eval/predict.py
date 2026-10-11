"""Run a detector over a split and store detections as JSONL."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snz.data.bdd import CLASSES, Frame
from snz.train.dataset import BDDDetection, collate


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    frames: Sequence[Frame],
    image_dir: Path,
    *,
    device: str,
    batch_size: int = 4,
    num_workers: int = 4,
) -> list[dict]:
    model.eval().to(device)
    loader = DataLoader(BDDDetection(frames, image_dir, train=False), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=collate)
    records, names = [], iter(f.name for f in frames)
    for images, _ in loader:
        for out in model([im.to(device) for im in images]):
            records.append({
                "name": next(names),
                "boxes": [[round(v, 2) for v in b] for b in out["boxes"].float().cpu().tolist()],
                "labels": [CLASSES[i - 1] for i in out["labels"].cpu().tolist()],
                "scores": [round(s, 5) for s in out["scores"].float().cpu().tolist()],
            })
    return records


def write_detections(path: Path, records: Sequence[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_detections(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
