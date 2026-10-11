import numpy as np
import pytest

from snz.data.bdd import EVAL_CLASSES, Box, Frame
from snz.eval.ap import APCalculator, average_precision
from snz.eval.coco_eval import evaluate
from snz.eval.matching import match_split


def synthetic_split(seed, n_images=40, prefix=""):
    rng = np.random.default_rng(seed)
    classes = EVAL_CLASSES[:5]
    frames, dets = [], []
    for i in range(n_images):
        boxes, d_boxes, d_labels, d_scores = [], [], [], []
        for _ in range(rng.integers(0, 6)):
            x, y = rng.uniform(0, 600, 2)
            w, h = rng.uniform(5, 120, 2)
            c = classes[rng.integers(len(classes))]
            ignore = bool(rng.random() < 0.08)
            boxes.append(Box(c, x, y, x + w, y + h, ignore))
            if rng.random() < 0.8:  # a detection near it, sometimes the wrong class
                j = rng.normal(0, 0.15 * min(w, h), 4)
                lab = c if rng.random() < 0.85 else classes[rng.integers(len(classes))]
                d_boxes.append([x + j[0], y + j[1], x + w + j[2], y + h + j[3]])
                d_labels.append(lab)
                d_scores.append(float(rng.uniform(0.05, 1)))
        for _ in range(rng.integers(0, 4)):  # false detections
            x, y = rng.uniform(0, 600, 2)
            w, h = rng.uniform(5, 80, 2)
            d_boxes.append([x, y, x + w, y + h])
            d_labels.append(classes[rng.integers(len(classes))])
            d_scores.append(float(rng.uniform(0.05, 1)))
        name = f"{prefix}{i}.jpg"
        frames.append(Frame(name, "clear", "daytime", tuple(boxes)))
        dets.append({"name": name, "boxes": d_boxes, "labels": d_labels, "scores": d_scores})
    return frames, dets


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_pycocotools(seed):
    frames, dets = synthetic_split(seed)
    ref = evaluate(frames, dets)
    ours = average_precision(match_split(frames, dets))
    assert ours["mAP"] == pytest.approx(ref["mAP"], abs=1e-6)
    assert ours["AP50"] == pytest.approx(ref["AP50"], abs=1e-6)
    for c, ap in ref["per_class"].items():
        assert (ours["per_class"][c] is None) == (ap is None)
        if ap is not None:
            assert ours["per_class"][c] == pytest.approx(ap, abs=1e-6)


def test_hand_computed_half_recall():
    frames = [Frame("a.jpg", "clear", "daytime",
                    (Box("car", 0, 0, 10, 10), Box("car", 50, 50, 60, 60)))]
    dets = [{"name": "a.jpg", "boxes": [[0, 0, 10, 10]], "labels": ["car"], "scores": [0.9]}]
    assert average_precision(match_split(frames, dets))["mAP"] == pytest.approx(51 / 101)


def test_integer_weights_equal_duplicated_images():
    frames, dets = synthetic_split(5, n_images=20)
    weights = np.random.default_rng(0).integers(0, 3, len(frames))
    dup_frames, dup_dets = [], []
    for copy in range(2):
        for f, d, w in zip(frames, dets, weights, strict=True):
            if w > copy:
                name = f"c{copy}_{f.name}"
                dup_frames.append(Frame(name, f.weather, f.timeofday, f.boxes))
                dup_dets.append({**d, "name": name})
    expected = average_precision(match_split(dup_frames, dup_dets))
    got = APCalculator(match_split(frames, dets))(weights)
    assert got["mAP"] == pytest.approx(expected["mAP"], abs=1e-9)
    assert got["AP50"] == pytest.approx(expected["AP50"], abs=1e-9)
