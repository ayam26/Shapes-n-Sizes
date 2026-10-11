import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torchvision.models import resnet50

from helpers import write_image
from snz.data.bdd import Box, Frame
from snz.train.config import Arm, load_arms, load_recipe
from snz.train.train import SeededStream, lr_factor, train_run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def tiny_recipe():
    return replace(
        load_recipe(CONFIGS / "recipe.toml"),
        batch_size=2, iterations=4, warmup_iters=2, lr_steps=[0.5], min_sizes=[48],
        max_size=64, bf16=False, num_workers=0, log_every=1, checkpoint_every=1,
    )


def test_configs_load():
    r = load_recipe(CONFIGS / "recipe.toml")
    assert r.min_sizes[-1] == 720 and r.p_original == 0.5 and r.trainable_layers == 3
    arms = load_arms(CONFIGS / "arms.toml")
    assert arms["BOTH"] == Arm("BOTH", "sin", True)
    assert list(arms) == ["BASE", "STY", "SIN", "BOTH"]


def test_load_recipe_rejects_unknown_keys(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text((CONFIGS / "recipe.toml").read_text() + "\nlearning_rate = 1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        load_recipe(p)


def test_seeded_stream_resumes_mid_epoch():
    it = iter(SeededStream(5, seed=3))
    first = [next(it) for _ in range(12)]
    assert sorted(i for i, e in first[:5]) == list(range(5))
    assert {e for _, e in first[5:10]} == {1}
    resumed = iter(SeededStream(5, seed=3, start=7))
    assert [next(resumed) for _ in range(5)] == first[7:12]


def test_lr_factor():
    assert lr_factor(0, 10, [50]) == pytest.approx(0.001)
    assert lr_factor(10, 10, [50]) == 1.0
    assert lr_factor(50, 10, [50]) == pytest.approx(0.1)


def _data(tmp_path):
    frames = []
    for i in range(4):
        write_image(tmp_path / "img" / f"f{i}.jpg", size=(64, 48), color=(40 * i, 90, 160))
        boxes = (Box("car", 5, 5, 30, 30),) if i != 3 else ()  # one empty frame
        frames.append(Frame(f"f{i}.jpg", "clear", "daytime", boxes))
    return frames


def _state():
    torch.manual_seed(123)
    return resnet50(weights=None).state_dict()


def _run(tmp_path, out, recipe, **kw):
    return train_run(arm=Arm("BASE", "in1k_v1", False), seed=0, recipe=recipe,
                     frames=_data(tmp_path), image_dir=tmp_path / "img", stylized_dir=None,
                     n_copies=2, backbone_state=_state(), out_dir=out, device="cpu", **kw)


def test_train_run_writes_outputs(tmp_path):
    summary = _run(tmp_path, tmp_path / "run", tiny_recipe())
    out = tmp_path / "run"
    assert (out / "final.pt").exists() and (out / "last.pt").exists()
    lines = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert [x["iteration"] for x in lines] == [1, 2, 3, 4]
    assert all(math.isfinite(x["loss"]) for x in lines)
    run = json.loads((out / "run.json").read_text())
    assert run["arm"] == "BASE" and run["iterations"] == 4 and summary == run


def test_resume_matches_uninterrupted_run(tmp_path):
    recipe = tiny_recipe()
    _run(tmp_path, tmp_path / "straight", recipe)
    assert _run(tmp_path, tmp_path / "resumed", recipe, stop_at=2) == {"stopped_at": 2}
    assert not (tmp_path / "resumed" / "final.pt").exists()
    _run(tmp_path, tmp_path / "resumed", recipe)
    a = torch.load(tmp_path / "straight" / "final.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=True)
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_resume_refuses_a_different_config(tmp_path):
    recipe = tiny_recipe()
    _run(tmp_path, tmp_path / "r", recipe, stop_at=2)
    with pytest.raises(ValueError, match="different"):
        _run(tmp_path, tmp_path / "r", replace(recipe, lr=0.5))


def test_divergence_stops_the_run(tmp_path):
    with pytest.raises(FloatingPointError, match="non-finite"):
        _run(tmp_path, tmp_path / "nan", replace(tiny_recipe(), lr=1e12, warmup_iters=0))
    assert not (tmp_path / "nan" / "final.pt").exists()
