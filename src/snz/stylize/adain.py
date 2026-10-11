"""AdaIN style transfer (Huang & Belongie, 2017).

Network layout matches naoto0804/pytorch-AdaIN, so its released `vgg_normalised.pth`
and `decoder.pth` load directly. Inputs are RGB tensors in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor, nn

from snz.hashing import KNOWN_SHA256, verify_sha256

ENCODER_LAYERS = 31  # vgg children up to relu4_1


def _conv(cin: int, cout: int) -> list[nn.Module]:
    return [nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(cin, cout, (3, 3)), nn.ReLU()]


def _pool() -> nn.Module:
    return nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)


def make_vgg() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        *_conv(3, 64), *_conv(64, 64), _pool(),
        *_conv(64, 128), *_conv(128, 128), _pool(),
        *_conv(128, 256), *_conv(256, 256), *_conv(256, 256), *_conv(256, 256), _pool(),
        *_conv(256, 512),  # relu4_1 = child 30
        *_conv(512, 512), *_conv(512, 512), *_conv(512, 512), _pool(),
        *_conv(512, 512), *_conv(512, 512), *_conv(512, 512), *_conv(512, 512),
    )


def make_decoder() -> nn.Sequential:
    def up() -> nn.Module:
        return nn.Upsample(scale_factor=2, mode="nearest")

    return nn.Sequential(
        *_conv(512, 256), up(),
        *_conv(256, 256), *_conv(256, 256), *_conv(256, 256), *_conv(256, 128), up(),
        *_conv(128, 128), *_conv(128, 64), up(),
        *_conv(64, 64),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
    )


def _mean_std(feat: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    n, c = feat.shape[:2]
    flat = feat.reshape(n, c, -1)
    mean = flat.mean(dim=2).reshape(n, c, 1, 1)
    std = (flat.var(dim=2) + eps).sqrt().reshape(n, c, 1, 1)
    return mean, std


def adain(content_feat: Tensor, style_feat: Tensor, eps: float = 1e-5) -> Tensor:
    c_mean, c_std = _mean_std(content_feat, eps)
    s_mean, s_std = _mean_std(style_feat, eps)
    return (content_feat - c_mean) / c_std * s_std + s_mean


@torch.no_grad()
def stylize_tensor(
    content: Tensor, style: Tensor, encoder: nn.Module, decoder: nn.Module, alpha: float = 1.0
) -> Tensor:
    f_c = encoder(content)
    t = adain(f_c, encoder(style))
    t = alpha * t + (1 - alpha) * f_c
    out = decoder(t)
    if out.shape[-2:] != content.shape[-2:]:
        out = F.interpolate(out, size=content.shape[-2:], mode="bilinear", align_corners=False)
    return out.clamp(0, 1)


class Stylizer:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: str = "cpu",
        style_size: int = 512,
        alpha: float = 1.0,
    ):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.device = device
        self.style_size = style_size
        self.alpha = alpha

    @classmethod
    def from_weights(
        cls, vgg_path: Path, decoder_path: Path, device: str = "cuda",
        style_size: int = 512, alpha: float = 1.0,
    ) -> Stylizer:
        verify_sha256(vgg_path, KNOWN_SHA256["vgg_normalised.pth"])
        verify_sha256(decoder_path, KNOWN_SHA256["decoder.pth"])
        vgg = make_vgg()
        vgg.load_state_dict(torch.load(vgg_path, map_location="cpu", weights_only=True))
        decoder = make_decoder()
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
        return cls(vgg[:ENCODER_LAYERS], decoder, device, style_size, alpha)

    def __call__(self, content: Image.Image, style: Image.Image) -> Image.Image:
        c = TF.to_tensor(content.convert("RGB")).unsqueeze(0).to(self.device)
        s = TF.resize(style.convert("RGB"), self.style_size)
        s = TF.to_tensor(s).unsqueeze(0).to(self.device)
        out = stylize_tensor(c, s, self.encoder, self.decoder, self.alpha)
        return TF.to_pil_image(out[0].float().cpu())
