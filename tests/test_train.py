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
    # the last scale is the evaluation size: the largest, and max_size fits 16:9 at that size
    assert r.min_sizes[-1] == max(r.min_sizes)
    assert r.max_size >= round(r.min_sizes[-1] * 16 / 9)
    assert r.p_original == 0.5 and r.trainable_layers == 3 and r.frozen
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


def test_resume_is_exact_with_scale_jitter(tmp_path):
    recipe = replace(tiny_recipe(), min_sizes=[32, 40, 48, 56], max_size=80)
    _run(tmp_path, tmp_path / "straight", recipe)
    _run(tmp_path, tmp_path / "resumed", recipe, stop_at=2)
    _run(tmp_path, tmp_path / "resumed", recipe)
    a = torch.load(tmp_path / "straight" / "final.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=True)
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_non_finite_gradient_stops_before_the_step(tmp_path, monkeypatch):
    import snz.train.train as train_mod

    monkeypatch.setattr(train_mod, "clip_grad_norm_",
                        lambda params, max_norm: torch.tensor(float("inf")))
    with pytest.raises(FloatingPointError, match="gradient"):
        _run(tmp_path, tmp_path / "g", tiny_recipe())


def test_missing_stylized_copies_fail_before_training(tmp_path):
    frames = _data(tmp_path)
    write_image(tmp_path / "sty" / "f0_s1.jpg", size=(64, 48))  # the rest are missing
    with pytest.raises(FileNotFoundError, match="7 stylized"):
        train_run(arm=Arm("STY", "in1k_v1", True), seed=0, recipe=tiny_recipe(), frames=frames,
                  image_dir=tmp_path / "img", stylized_dir=tmp_path / "sty", n_copies=2,
                  backbone_state=_state(), out_dir=tmp_path / "s", device="cpu")
    assert not (tmp_path / "s" / "log.jsonl").exists()


def test_evaluate_uses_the_runs_own_recipe(tmp_path):
    from helpers import frame_json, write_labels
    from snz.cli import main
    from snz.data.manifest import write_manifest

    _run(tmp_path, tmp_path / "run", tiny_recipe())
    labels = write_labels(tmp_path / "l.json",
                          [frame_json(f"f{i}.jpg", labels=[("car", 5, 5, 30, 30)])
                           for i in range(4)])
    write_manifest(tmp_path / "m" / "id-val.csv", [(f"f{i}.jpg", "h") for i in range(4)])
    args = ["evaluate", "--split", "id-val", "--labels", str(labels),
            "--manifest", str(tmp_path / "m" / "id-val.csv"), "--images", str(tmp_path / "img"),
            "--out", str(tmp_path / "ev"), "--device", "cpu"]
    assert main([*args, "--checkpoint", str(tmp_path / "run" / "final.pt")]) == 0
    metrics = json.loads((tmp_path / "ev" / "id-val.metrics.json").read_text())
    assert metrics["min_size"] == 48
    (tmp_path / "run" / "run.json").unlink()
    with pytest.raises(FileNotFoundError, match="run.json"):
        main([*args, "--checkpoint", str(tmp_path / "run" / "final.pt")])


def test_evaluate_reads_corrupted_copies(tmp_path):
    from helpers import frame_json, write_labels
    from snz.cli import main
    from snz.corrupt import corrupt_name
    from snz.data.manifest import write_manifest

    _run(tmp_path, tmp_path / "run", tiny_recipe())
    for i in range(2):
        write_image(tmp_path / "tc" / corrupt_name(f"f{i}.jpg", "fog", 3), size=(64, 48))
    labels = write_labels(tmp_path / "l.json", [frame_json(f"f{i}.jpg") for i in range(2)])
    write_manifest(tmp_path / "m" / "t-corrupt.csv", [(f"f{i}.jpg", "h") for i in range(2)])
    assert main(["evaluate", "--split", "t-corrupt-fog-3", "--labels", str(labels),
                 "--manifest", str(tmp_path / "m" / "t-corrupt.csv"),
                 "--images", str(tmp_path / "tc"), "--corruption", "fog", "--severity", "3",
                 "--checkpoint", str(tmp_path / "run" / "final.pt"), "--out", str(tmp_path / "ev"),
                 "--device", "cpu", "--no-guard"]) == 0
    assert (tmp_path / "ev" / "t-corrupt-fog-3.detections.jsonl").exists()


def test_train_cli_only_accepts_the_train_manifest(tmp_path):
    from snz.cli import main

    with pytest.raises(SystemExit, match="train.csv"):
        main(["train", "--pilot", "--arm", "BASE", "--seed", "0", "--labels", "x",
              "--manifest", "manifests/t-clean.csv", "--images", "x", "--out", str(tmp_path)])
