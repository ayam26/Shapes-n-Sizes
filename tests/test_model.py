import pytest
import torch
from torch import nn
from torchvision.models import resnet50

from snz.hashing import ChecksumError, sha256_file
from snz.train.model import NUM_CLASSES, build_model, load_backbone_state


def _state(seed):
    torch.manual_seed(seed)
    return resnet50(weights=None).state_dict()


def _trainable(model):
    return [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]


def test_both_backbones_build_identical_architectures():
    a, b = build_model(_state(0)), build_model(_state(1))
    assert _trainable(a) == _trainable(b)
    assert a.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES == 11


def test_backbone_weights_loaded_and_frozen_as_specified():
    state = _state(0)
    model = build_model(state)
    body = model.backbone.body
    torch.testing.assert_close(body.conv1.weight, state["conv1.weight"])
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert not body.conv1.weight.requires_grad
    assert not any(p.requires_grad for p in body.layer1.parameters())
    for layer in (body.layer2, body.layer3, body.layer4):
        assert all(p.requires_grad for p in layer.parameters())


def test_sin_checkpoint_loading(tmp_path):
    state = _state(0)
    path = tmp_path / "sin.pth.tar"
    torch.save({"epoch": 60, "arch": "resnet50",
                "state_dict": {f"module.{k}": v for k, v in state.items()}}, path)
    loaded = load_backbone_state("sin", path, expected_sha256=sha256_file(path))
    assert set(loaded) == set(state)
    with pytest.raises(ChecksumError):
        load_backbone_state("sin", path, expected_sha256="0" * 64)


def test_bad_inputs():
    with pytest.raises(ValueError, match="sin_path"):
        load_backbone_state("sin")
    with pytest.raises(ValueError, match="unknown backbone"):
        load_backbone_state("vit")


def test_forward_on_cpu():
    model = build_model(_state(0), min_size=48, max_size=64).eval()
    with torch.no_grad():
        out = model([torch.rand(3, 48, 64)])
    assert set(out[0]) == {"boxes", "labels", "scores"}


def test_roi_pooler_scales_are_fixed_up_front():
    model = build_model(_state(0))
    pool = model.roi_heads.box_roi_pool
    assert pool.scales == [1 / 4, 1 / 8, 1 / 16, 1 / 32]
    assert pool.map_levels is not None
