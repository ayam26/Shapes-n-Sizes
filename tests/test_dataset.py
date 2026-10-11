from collections import Counter

import pytest
import torch

from helpers import write_image
from snz.data.bdd import Box, Frame
from snz.train.dataset import LABEL_INDEX, BDDDetection, choose_variant, frame_target


def test_choose_variant_probabilities_and_determinism():
    names = [f"img{i}.jpg" for i in range(20000)]
    counts = Counter(choose_variant(n, seed=0, epoch=0) for n in names)
    assert abs(counts[0] / 20000 - 0.5) < 0.015
    assert abs(counts[1] / 20000 - 0.25) < 0.015
    assert abs(counts[2] / 20000 - 0.25) < 0.015
    assert choose_variant("a.jpg", 0, 3) == choose_variant("a.jpg", 0, 3)
    assert any(choose_variant(n, 0, 0) != choose_variant(n, 0, 1) for n in names[:50])
    assert any(choose_variant(n, 0, 0) != choose_variant(n, 1, 0) for n in names[:50])


def test_frame_target_including_empty():
    f = Frame("a.jpg", "clear", "daytime",
              (Box("car", 1, 2, 30, 40), Box("traffic sign", 0, 0, 5, 5)))
    t = frame_target(f)
    assert t["boxes"].tolist() == [[1, 2, 30, 40], [0, 0, 5, 5]]
    assert t["labels"].tolist() == [LABEL_INDEX["car"], LABEL_INDEX["traffic sign"]]
    empty = frame_target(Frame("b.jpg", "clear", "daytime", ()))
    assert empty["boxes"].shape == (0, 4) and empty["labels"].dtype == torch.int64


def _frames(tmp_path, n=40):
    frames = []
    for i in range(n):
        write_image(tmp_path / "img" / f"f{i}.jpg", size=(64, 48), color=(0, 0, 0))
        write_image(tmp_path / "sty" / f"f{i}_s1.jpg", size=(64, 48), color=(255, 255, 255))
        write_image(tmp_path / "sty" / f"f{i}_s2.jpg", size=(64, 48), color=(255, 255, 255))
        frames.append(Frame(f"f{i}.jpg", "clear", "daytime", (Box("car", 10, 5, 20, 15),)))
    return frames


def test_dataset_reads_stylized_copies_when_chosen(tmp_path):
    frames = _frames(tmp_path)
    ds = BDDDetection(frames, tmp_path / "img", stylized_dir=tmp_path / "sty", seed=0,
                      hflip_p=0.0)
    for i, f in enumerate(frames):
        image, _ = ds[(i, 0)]
        is_stylized = bool(image.mean() > 0.5)
        assert is_stylized == (choose_variant(f.name, 0, 0) != 0)
    plain = BDDDetection(frames, tmp_path / "img", seed=0, hflip_p=0.0)
    assert all(plain[(i, 0)][0].mean() < 0.5 for i in range(len(frames)))


def test_missing_stylized_copy_is_an_error(tmp_path):
    frames = _frames(tmp_path, n=40)
    for p in (tmp_path / "sty").glob("*_s2.jpg"):
        p.unlink()
    ds = BDDDetection(frames, tmp_path / "img", stylized_dir=tmp_path / "sty", seed=0)
    with pytest.raises(FileNotFoundError, match="stylized"):
        for i in range(len(frames)):
            ds[(i, 0)]


def test_hflip_mirrors_boxes(tmp_path):
    frames = _frames(tmp_path, n=1)
    ds = BDDDetection(frames, tmp_path / "img", seed=0, hflip_p=1.0)
    _, target = ds[0]
    assert target["boxes"].tolist() == [[44, 5, 54, 15]]  # width 64: x1' = 64-20, x2' = 64-10
    eval_ds = BDDDetection(frames, tmp_path / "img", seed=0, train=False, hflip_p=1.0)
    assert eval_ds[0][1]["boxes"].tolist() == [[10, 5, 20, 15]]
