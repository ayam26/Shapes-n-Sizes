"""BDD detection dataset with original/stylized mixing (spec §4.3, §5.3).

Every random choice (which copy, whether to flip) is a hash of (seed, epoch, name),
so data is identical across worker counts and across a resumed run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from snz.data.bdd import CLASSES, Frame
from snz.stylize.build import stylized_name

LABEL_INDEX = {c: i + 1 for i, c in enumerate(CLASSES)}


def _uniform(*parts) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_variant(
    name: str, seed: int, epoch: int, p_original: float = 0.5, n_copies: int = 2
) -> int:
    u = _uniform(seed, epoch, name, "variant")
    if u < p_original:
        return 0
    return 1 + min(int((u - p_original) / (1 - p_original) * n_copies), n_copies - 1)


def frame_target(frame: Frame) -> dict[str, Tensor]:
    # torchvision Faster R-CNN has no ignore-region support; ignore boxes are left out
    kept = [b for b in frame.boxes if not b.ignore]
    boxes = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in kept], dtype=torch.float32)
    labels = torch.tensor([LABEL_INDEX[b.category] for b in kept], dtype=torch.int64)
    return {"boxes": boxes.reshape(-1, 4), "labels": labels}


class BDDDetection(Dataset):
    def __init__(
        self,
        frames: Sequence[Frame],
        image_dir: Path,
        *,
        stylized_dir: Path | None = None,
        p_original: float = 0.5,
        n_copies: int = 2,
        seed: int = 0,
        train: bool = True,
        hflip_p: float = 0.5,
        file_name: Callable[[str], str] | None = None,
    ):
        self.frames = list(frames)
        self.image_dir = Path(image_dir)
        self.stylized_dir = Path(stylized_dir) if stylized_dir is not None else None
        self.p_original = p_original
        self.n_copies = n_copies
        self.seed = seed
        self.train = train
        self.hflip_p = hflip_p
        self.file_name = file_name  # eval only: e.g. T-texture copies; records keep frame.name

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, key: int | tuple[int, int]) -> tuple[Tensor, dict[str, Tensor]]:
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        frame = self.frames[index]
        variant = 0
        if self.train and self.stylized_dir is not None:
            variant = choose_variant(frame.name, self.seed, epoch, self.p_original, self.n_copies)
        if variant == 0:
            name = self.file_name(frame.name) if self.file_name else frame.name
            path = self.image_dir / name
        else:
            path = self.stylized_dir / stylized_name(frame.name, variant)
            if not path.exists():
                raise FileNotFoundError(f"missing stylized copy {path} (run `snz stylize`)")
        image = TF.to_tensor(Image.open(path).convert("RGB"))
        target = frame_target(frame)
        if self.train and _uniform(self.seed, epoch, frame.name, "flip") < self.hflip_p:
            width = image.shape[-1]
            image = image.flip(-1)
            boxes = target["boxes"].clone()
            boxes[:, 0] = width - target["boxes"][:, 2]
            boxes[:, 2] = width - target["boxes"][:, 0]
            target["boxes"] = boxes
        return image, target


def collate(batch):
    return tuple(zip(*batch, strict=True))
