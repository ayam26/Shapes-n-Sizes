import numpy as np
import pytest

from snz.data.bdd import Box, Frame
from snz.eval.matching import (
    FD_DUPLICATE,
    FD_IGNORED,
    FD_INVENTED,
    FD_NEAR_OTHER,
    FD_POOR_LOC,
    FD_TP,
    FD_WRONG_CLASS,
    box_iou,
    match_split,
)


def _det(name, *items):
    return {"name": name, "boxes": [list(i[1:5]) for i in items],
            "labels": [i[0] for i in items], "scores": [i[5] for i in items]}


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])


def test_false_detection_types():
    frame = Frame("a.jpg", "clear", "daytime", (
        Box("car", 0, 0, 10, 10),
        Box("pedestrian", 20, 20, 30, 40),
        Box("train", 50, 0, 90, 30),
    ))
    det = _det("a.jpg",
               ("car", 0, 0, 10, 10, 0.9),        # TP
               ("car", 0, 0, 10, 10, 0.8),        # duplicate
               ("car", 20, 20, 30, 40, 0.7),      # wrong class (pedestrian)
               ("car", 0, 0, 10, 30, 0.6),        # IoU 1/3 with the car: poor localization
               ("car", 200, 200, 210, 210, 0.5),  # invented
               ("car", 50, 0, 90, 30, 0.4),       # wrong class (a `train` GT counts)
               ("car", 20, 20, 30, 25, 0.3))      # IoU .25 with pedestrian only
    m = match_split([frame], [det])
    by_score = dict(zip(m.det_score.round(2), m.det_fd_type, strict=True))
    assert by_score == {0.9: FD_TP, 0.8: FD_DUPLICATE, 0.7: FD_WRONG_CLASS, 0.6: FD_POOR_LOC,
                        0.5: FD_INVENTED, 0.4: FD_WRONG_CLASS, 0.3: FD_NEAR_OTHER}
    car, ped = m.classes.index("car"), m.classes.index("pedestrian")
    assert m.gt_count[0, car] == 1 and m.gt_count[0, ped] == 1
    assert m.det_tp[:, 0].sum() == 1


def test_iou_thresholds_and_ignore_regions():
    frame = Frame("a.jpg", "clear", "daytime", (
        Box("car", 0, 0, 10, 10),
        Box("car", 100, 100, 200, 200, ignore=True),
    ))
    det = _det("a.jpg", ("car", 0, 0, 10, 8, 0.9),        # IoU .8 with the car
               ("car", 120, 120, 140, 140, 0.8),           # inside the ignore region
               ("car", 150, 150, 160, 160, 0.7))           # also inside: regions rematch
    m = match_split([frame], [det])
    assert m.det_tp[0].tolist() == [True] * 7 + [False] * 3  # .5 .. .8 match, .85+ do not
    assert m.det_ignored[1].all() and m.det_ignored[2].all()
    assert m.det_fd_type.tolist() == [FD_TP, FD_IGNORED, FD_IGNORED]
    assert m.gt_count.sum() == 1


def test_max_dets_truncation_and_unknown_labels():
    frame = Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10),))
    det = _det("a.jpg", ("car", 0, 0, 10, 10, 0.2), ("car", 1, 1, 9, 9, 0.9),
               ("car", 2, 2, 8, 8, 0.5), ("train", 0, 0, 10, 10, 0.99))
    m = match_split([frame], [det], max_dets=2)
    assert m.det_score.tolist() == [0.9, 0.5]


def test_subset_reindexes_images():
    frames = [Frame(f"{i}.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10),))
              for i in range(3)]
    dets = [_det(f"{i}.jpg", ("car", 0, 0, 10, 10, 0.5 + i / 10)) for i in range(3)]
    sub = match_split(frames, dets).subset(["2.jpg", "0.jpg"])
    assert sub.image_names == ["2.jpg", "0.jpg"]
    assert sub.det_image.tolist() == [1, 0] and sub.det_score.round(1).tolist() == [0.5, 0.7]
    assert sub.gt_count.shape[0] == 2
    with pytest.raises(KeyError):
        sub.subset(["1.jpg"])


def test_unknown_image_rejected():
    with pytest.raises(ValueError, match="zzz.jpg"):
        match_split([], [_det("zzz.jpg")])
