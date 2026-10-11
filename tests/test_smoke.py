from dataclasses import replace
from pathlib import Path

import torch
from torchvision.models import resnet50

from snz.train.config import load_recipe
from snz.train.smoke import run_smoke

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_run_smoke_tiny(tmp_path):
    torch.manual_seed(0)
    states = {"in1k_v1": resnet50(weights=None).state_dict(),
              "sin": resnet50(weights=None).state_dict()}
    recipe = replace(load_recipe(CONFIGS / "recipe.toml"), batch_size=2, min_sizes=[48],
                     max_size=64, bf16=False, num_workers=0, warmup_iters=1)
    report = run_smoke(states, recipe, tmp_path, device="cpu", iterations=2,
                       image_size=(64, 48), n_images=4)
    for kind in ("in1k_v1", "sin"):
        r = report["backbones"][kind]
        assert r["loaded"] and r["iterations"] == 2 and r["final_loss"] > 0
    assert (report["backbones"]["in1k_v1"]["trainable_params"]
            == report["backbones"]["sin"]["trainable_params"])
    assert report["torch"] == torch.__version__
