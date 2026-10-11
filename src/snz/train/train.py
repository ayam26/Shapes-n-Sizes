"""Iteration-based Faster R-CNN training with exact resume (spec §5.3).

Determinism: model init uses torch.manual_seed(seed); data order comes from
SeededStream; augmentation choices are hashes (dataset.py); the global RNG used by
the RPN/ROI samplers is saved in every checkpoint and restored on resume.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from snz.data.bdd import Frame
from snz.fsutil import atomic_replace
from snz.train.config import Arm, Recipe, recipe_sha256
from snz.train.dataset import BDDDetection, collate
from snz.train.model import build_model


class SeededStream(Sampler):
    """Endless stream of (index, epoch): a fresh seeded permutation each epoch."""

    def __init__(self, n: int, seed: int, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self) -> Iterator[tuple[int, int]]:
        pos = self.start
        while True:
            epoch, offset = divmod(pos, self.n)
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            for i in torch.randperm(self.n, generator=g).tolist()[offset:]:
                yield i, epoch
            pos = (epoch + 1) * self.n


def lr_factor(iteration: int, warmup_iters: int, step_iters: Sequence[int]) -> float:
    factor = 0.1 ** sum(iteration >= s for s in step_iters)
    if iteration < warmup_iters:
        factor *= 0.001 + 0.999 * iteration / warmup_iters
    return factor


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    atomic_replace(tmp, path)


def train_run(
    *,
    arm: Arm,
    seed: int,
    recipe: Recipe,
    frames: Sequence[Frame],
    image_dir: Path,
    stylized_dir: Path | None,
    n_copies: int,
    backbone_state: Mapping[str, Tensor],
    out_dir: Path,
    device: str = "cuda",
    stop_at: int | None = None,
    pilot: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if arm.stylized and stylized_dir is None:
        raise ValueError(f"arm {arm.name} trains on stylized copies; stylized_dir is required")
    identity = {"arm": arm.name, "backbone": arm.backbone, "stylized": arm.stylized,
                "seed": seed, "recipe_sha256": recipe_sha256(recipe), "frames": len(frames)}

    torch.manual_seed(seed)
    model = build_model(backbone_state, trainable_layers=recipe.trainable_layers,
                        min_size=tuple(recipe.min_sizes), max_size=recipe.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=recipe.lr, momentum=recipe.momentum,
                          weight_decay=recipe.weight_decay)

    start = 0
    last = out_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["identity"] != identity:
            raise ValueError(f"{last} was written by a different run config: "
                             f"{ckpt['identity']} != {identity}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng"])
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        start = ckpt["iteration"]

    dataset = BDDDetection(
        frames, image_dir, stylized_dir=stylized_dir if arm.stylized else None,
        p_original=recipe.p_original, n_copies=n_copies, seed=seed, train=True,
        hflip_p=recipe.hflip_p,
    )
    loader = DataLoader(
        dataset, batch_size=recipe.batch_size,
        sampler=SeededStream(len(dataset), seed, start * recipe.batch_size),
        num_workers=recipe.num_workers, collate_fn=collate,
        pin_memory=device.startswith("cuda"),
        # own generator: creating the iterator must not draw from the global RNG
        generator=torch.Generator().manual_seed(seed),
    )
    step_iters = [round(f * recipe.iterations) for f in recipe.lr_steps]
    use_bf16 = recipe.bf16 and device.startswith("cuda")
    end = recipe.iterations if stop_at is None else min(stop_at, recipe.iterations)

    def checkpoint(iteration: int) -> None:
        _atomic_save({
            "identity": identity, "iteration": iteration, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, last)

    model.train()
    it, t0 = start, time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if it < end:
        batches = iter(loader)
        with open(out_dir / "log.jsonl", "a") as log:
            while it < end:
                images, targets = next(batches)
                for group in opt.param_groups:
                    group["lr"] = recipe.lr * lr_factor(it, recipe.warmup_iters, step_iters)
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    losses = model(images, targets)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    parts = {k: v.item() for k, v in losses.items()}
                    raise FloatingPointError(f"non-finite loss at iteration {it}: {parts} "
                                             f"(last good checkpoint: {last})")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                it += 1
                if it % recipe.log_every == 0 or it == end:
                    log.write(json.dumps({"iteration": it, "loss": loss.item(),
                                          "lr": opt.param_groups[0]["lr"],
                                          **{k: v.item() for k, v in losses.items()}}) + "\n")
                    log.flush()
                if it % recipe.checkpoint_every == 0 or it == end:
                    checkpoint(it)
        del batches

    if it < recipe.iterations:  # stopped early on purpose (stop_at)
        return {"stopped_at": it}
    seconds = time.perf_counter() - t0
    _atomic_save(model.state_dict(), out_dir / "final.pt")
    summary = {
        **identity, "iterations": recipe.iterations, "recipe": dataclasses.asdict(recipe),
        "pilot": pilot, "git_commit": _git_commit(), "device": device,
        "torch": torch.__version__,
        "iters_per_sec_this_session": round((it - start) / seconds, 4) if it > start else None,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if device.startswith("cuda") else None,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2))
    return summary
