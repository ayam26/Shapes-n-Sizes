"""Faster R-CNN ResNet-50 + FPN with a chosen backbone initialization (spec §5.1-5.2).

The detector is assembled by hand rather than with fasterrcnn_resnet50_fpn(weights=None),
because that path switches to trainable BatchNorm and all-layers-trainable when no
pretrained weights are passed. Here the backbone always has frozen BN and 3 trainable
stages, and FPN/RPN/ROI heads are freshly initialized (no COCO weights).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.misc import FrozenBatchNorm2d

from snz.data.bdd import CLASSES
from snz.hashing import KNOWN_SHA256, verify_sha256

NUM_CLASSES = len(CLASSES) + 1  # + background
BACKBONES = ("in1k_v1", "sin")
SIN_FILENAME = "resnet50_train_60_epochs-c8e5653e.pth.tar"


def load_backbone_state(
    kind: str, sin_path: Path | None = None, expected_sha256: str | None = None
) -> dict[str, Tensor]:
    if kind == "in1k_v1":
        return ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
    if kind == "sin":
        if sin_path is None:
            raise ValueError("backbone 'sin' needs sin_path (Stylized-ImageNet ResNet-50)")
        verify_sha256(sin_path, expected_sha256 or KNOWN_SHA256[SIN_FILENAME])
        ckpt = torch.load(sin_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        return {k.removeprefix("module."): v for k, v in state.items()}
    raise ValueError(f"unknown backbone {kind!r}; expected one of {BACKBONES}")


def build_model(
    backbone_state: Mapping[str, Tensor] | None,
    *,
    trainable_layers: int = 3,
    min_size: int | tuple[int, ...] = 720,
    max_size: int = 1280,
    **kwargs,
) -> FasterRCNN:
    body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    if backbone_state is not None:
        body.load_state_dict(backbone_state)
    backbone = _resnet_fpn_extractor(body, trainable_layers)
    return FasterRCNN(backbone, num_classes=NUM_CLASSES, min_size=min_size, max_size=max_size,
                      **kwargs)
