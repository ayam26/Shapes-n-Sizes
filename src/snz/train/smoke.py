"""Gate G1: both backbones load into the detector and a short training run completes.

Runs on synthetic images (before any dataset is downloaded) at full 1280x720, so the
measured speed and memory are a first estimate for the pilot.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import random
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image

from snz.data.bdd import CLASSES, Box, Frame
from snz.train.config import Arm, Recipe
from snz.train.model import build_model
from snz.train.train import train_run


def _synthetic(image_dir: Path, n: int, size: tuple[int, int]) -> list[Frame]:
    rng = random.Random(0)
    w, h = size
    frames = []
    for i in range(n):
        name = f"synthetic_{i}.jpg"
        Image.effect_noise(size, 64).convert("RGB").save(image_dir / name)
        boxes = []
        for _ in range(rng.randint(1, 6)):
            x1, y1 = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
            boxes.append(Box(rng.choice(CLASSES), x1, y1,
                             x1 + rng.uniform(8, w * 0.2), y1 + rng.uniform(8, h * 0.2)))
        frames.append(Frame(name, "clear", "daytime", tuple(boxes)))
    return frames


def run_smoke(
    states: Mapping[str, Mapping[str, torch.Tensor]],
    recipe: Recipe,
    work_dir: Path,
    *,
    device: str,
    iterations: int = 50,
    image_size: tuple[int, int] = (1280, 720),
    n_images: int = 8,
) -> dict:
    work_dir = Path(work_dir)
    (work_dir / "images").mkdir(parents=True, exist_ok=True)
    frames = _synthetic(work_dir / "images", n_images, image_size)
    recipe = dataclasses.replace(recipe, iterations=iterations, lr_steps=[],
                                 log_every=1, checkpoint_every=iterations)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "device_name": (torch.cuda.get_device_name(0) if device.startswith("cuda")
                        else platform.processor()),
        "recipe": dataclasses.asdict(recipe),
        "backbones": {},
    }
    for kind, state in states.items():
        model = build_model(state, trainable_layers=recipe.trainable_layers)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        out = work_dir / f"run_{kind}"
        summary = train_run(arm=Arm(f"smoke-{kind}", kind, False), seed=0, recipe=recipe,
                            frames=frames, image_dir=work_dir / "images", stylized_dir=None,
                            n_copies=2, backbone_state=state, out_dir=out, device=device)
        last = json.loads((out / "log.jsonl").read_text().splitlines()[-1])
        report["backbones"][kind] = {
            "loaded": True,
            "trainable_params": trainable,
            "iterations": last["iteration"],
            "iters_per_sec": summary["iters_per_sec_this_session"],
            "peak_mem_gb": summary["peak_mem_gb"],
            "final_loss": last["loss"],
        }
    return report
