import torch
from PIL import Image
from torch import nn

from helpers import write_image
from snz.stylize.adain import (
    ENCODER_LAYERS,
    Stylizer,
    adain,
    make_decoder,
    make_vgg,
    stylize_tensor,
)
from snz.stylize.pool import list_styles, split_style_pool


def test_adain_matches_style_statistics():
    g = torch.Generator().manual_seed(0)
    content = torch.randn(1, 4, 16, 16, generator=g) * 3 + 1
    style = torch.randn(1, 4, 16, 16, generator=g) * 0.5 - 2
    out = adain(content, style)
    flat_out, flat_style = out.reshape(4, -1), style.reshape(4, -1)
    torch.testing.assert_close(flat_out.mean(1), flat_style.mean(1), atol=1e-4, rtol=0)
    torch.testing.assert_close(flat_out.std(1), flat_style.std(1), atol=1e-3, rtol=0)


def test_real_architecture_shapes():
    encoder = make_vgg()[:ENCODER_LAYERS]
    decoder = make_decoder()
    x = torch.rand(1, 3, 64, 64)
    feat = encoder(x)
    assert feat.shape == (1, 512, 8, 8)
    assert decoder(feat).shape == (1, 3, 64, 64)


def _stub():
    encoder = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8, padding=1))
    decoder = nn.Sequential(nn.Upsample(scale_factor=8), nn.Conv2d(4, 3, 1))
    return encoder, decoder


def test_stylize_tensor_preserves_odd_sizes():
    encoder, decoder = _stub()
    out = stylize_tensor(torch.rand(1, 3, 23, 37), torch.rand(1, 3, 50, 50), encoder, decoder)
    assert out.shape == (1, 3, 23, 37)
    assert out.min() >= 0 and out.max() <= 1


def test_stylizer_returns_same_size_rgb_image():
    encoder, decoder = _stub()
    stylizer = Stylizer(encoder, decoder, device="cpu", style_size=32)
    result = stylizer(Image.new("RGB", (37, 23)), Image.new("RGB", (80, 60), (10, 200, 30)))
    assert result.size == (37, 23) and result.mode == "RGB"


def test_style_pool_split(tmp_path):
    for cat in ("banded", "dotted"):
        for i in range(10):
            write_image(tmp_path / cat / f"{cat}_{i:02d}.jpg")
    (tmp_path / "readme.txt").write_text("not an image")
    styles = list_styles(tmp_path)
    assert len(styles) == 20 and styles[0] == "banded/banded_00.jpg"
    train, test = split_style_pool(styles, seed=0, test_fraction=0.2)
    assert len(test) == 4 and len(train) == 16
    assert not set(train) & set(test)
    assert split_style_pool(list(reversed(styles)), seed=0) == (train, test)
    assert split_style_pool(styles, seed=1) != (train, test)
