import numpy as np
import pytest

from snz.data.bdd import Box, Frame
from snz.eval.matching import match_split
from snz.eval.operating import false_detections, recall_at, threshold_for_recall


def _split():
    # 4 cars over two images; scores: TP .9, FP .8 (invented), TP .7, TP .4, FP .3 (duplicate)
    frames = [
        Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10), Box("car", 20, 0, 30, 10))),
        Frame("b.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10), Box("car", 40, 0, 50, 10))),
    ]
    dets = [
        {"name": "a.jpg", "boxes": [[0, 0, 10, 10], [300, 300, 310, 310], [20, 0, 30, 10]],
         "labels": ["car"] * 3, "scores": [0.9, 0.8, 0.7]},
        {"name": "b.jpg", "boxes": [[0, 0, 10, 10], [0, 0, 10, 10]],
         "labels": ["car"] * 2, "scores": [0.4, 0.3]},
    ]
    return match_split(frames, dets)


def test_recall_at():
    m = _split()
    assert recall_at(m, 0.5) == pytest.approx(2 / 4)
    assert recall_at(m, 0.0) == pytest.approx(3 / 4)
    # image b drawn twice, a not at all
    assert recall_at(m, 0.0, weights=np.array([0, 2])) == pytest.approx(2 / 4)


def test_threshold_for_recall_picks_highest_threshold_reaching_target():
    m = _split()
    op = threshold_for_recall(m, 0.5)
    assert op.threshold == pytest.approx(0.7) and op.recall == pytest.approx(0.5)
    assert not op.flagged
    op = threshold_for_recall(m, 0.75)
    assert op.threshold == pytest.approx(0.4) and not op.flagged


def test_unreachable_recall_is_flagged_at_max_recall():
    # max recall .75 is first reached at score .4; the duplicate at .3 adds nothing
    op = threshold_for_recall(_split(), 1.0)
    assert op.flagged and op.recall == pytest.approx(0.75) and op.threshold == pytest.approx(0.4)


def test_weighted_threshold_for_recall():
    m = _split()
    # image a dropped: only b's GT count (2); its TP at .4 gives recall .5
    op = threshold_for_recall(m, 0.5, weights=np.array([0, 1]))
    assert op.threshold == pytest.approx(0.4) and not op.flagged


def test_false_detection_counts_are_absolute_too():
    assert false_detections(_split(), 0.0)["count"] == pytest.approx(2)


def test_false_detections_per_image_and_types():
    m = _split()
    fd = false_detections(m, 0.0)
    assert fd["per_image"] == pytest.approx(2 / 2)
    assert fd["by_type"]["invented"] == pytest.approx(1 / 2)
    assert fd["by_type"]["duplicate"] == pytest.approx(1 / 2)
    assert false_detections(m, 0.85)["per_image"] == 0
    assert false_detections(m, 0.0, weights=np.array([1, 3]))["per_image"] == pytest.approx(
        (1 + 3) / 4)
