import functools

import numpy as np
import pytest

from gnn_helpers import EmptySegmenter, FakeSegmenter
from helpers import write_image
from snz.gnn.outlines import extract_outlines, load_outlines, save_outlines, select_detections
from snz.stylize.build import stylized_name


def test_select_detections():
    r = {"name": "a.jpg", "boxes": [[0, 0, 5, 5]] * 4,
         "labels": ["car", "train", "car", "pedestrian"], "scores": [0.9, 0.95, 0.04, 0.3]}
    assert select_detections(r, top_k=50) == [0, 3]
    assert select_detections(r, top_k=1) == [0]


def test_extract_outlines_and_roundtrip(tmp_path):
    write_image(tmp_path / "a.jpg", size=(64, 48), color=(200, 10, 10))
    write_image(tmp_path / "b.jpg", size=(64, 48))
    recs = [{"name": "a.jpg", "boxes": [[10, 10, 40, 30]], "labels": ["car"], "scores": [0.8]},
            {"name": "b.jpg", "boxes": [], "labels": [], "scores": []}]
    o = extract_outlines(recs, tmp_path, FakeSegmenter())
    assert o.names == ["a.jpg", "b.jpg"] and len(o.score) == 1 and o.ok.all()
    assert o.color[0].mean(axis=0)[0] > 150
    assert o.image_wh[0].tolist() == [64, 48]
    save_outlines(tmp_path / "o.npz", o)
    p = load_outlines(tmp_path / "o.npz")
    assert p.names == o.names and np.array_equal(p.contour, o.contour) and p.label == o.label


def test_failed_mask_falls_back_to_box(tmp_path):
    write_image(tmp_path / "a.jpg")
    recs = [{"name": "a.jpg", "boxes": [[10, 10, 40, 30]], "labels": ["car"], "scores": [0.8]}]
    o = extract_outlines(recs, tmp_path, EmptySegmenter())
    assert not o.ok[0] and np.isfinite(o.contour).all()


def test_stylized_file_name_used(tmp_path):
    write_image(tmp_path / "a_s1.jpg")
    recs = [{"name": "a.jpg", "boxes": [[1, 1, 9, 9]], "labels": ["car"], "scores": [0.9]}]
    o = extract_outlines(recs, tmp_path, FakeSegmenter(),
                         file_name=functools.partial(stylized_name, copy=1))
    assert o.names == ["a.jpg"] and len(o.score) == 1


def test_save_is_atomic(tmp_path, monkeypatch):
    o = extract_outlines([], tmp_path, FakeSegmenter())
    monkeypatch.setattr(np, "savez_compressed",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        save_outlines(tmp_path / "o.npz", o)
    assert not (tmp_path / "o.npz").exists()
