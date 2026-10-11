from gnn_helpers import outline_set_for
from snz.data.bdd import EVAL_CLASSES, Box, Frame
from snz.gnn.targets import fold_split, network_targets, outline_records, tp_targets


def test_fold_split_deterministic_disjoint():
    names = [f"{i}.jpg" for i in range(11)]
    a, b = fold_split(names)
    assert fold_split(names) == (a, b) and not set(a) & set(b) and sorted(a + b) == sorted(names)
    assert abs(len(a) - len(b)) <= 1


def test_network_targets_rules():
    f = Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10), Box("truck", 50, 50, 60, 60),
                                            Box("car", 100, 100, 110, 110, ignore=True)))
    o = outline_set_for("a.jpg", boxes=[[0, 0, 10, 10], [50, 50, 60, 62], [200, 200, 210, 210],
                                        [0, 0, 10, 25], [100, 100, 110, 110]])
    car, truck = EVAL_CLASSES.index("car"), EVAL_CLASSES.index("truck")
    assert list(network_targets(o, [f])) == [car, truck, 9, -1, -1]


def test_outline_records_cover_every_image():
    o = outline_set_for("a.jpg", boxes=[[0, 0, 5, 5]], extra_names=["empty.jpg"])
    recs = outline_records(o)
    assert [r["name"] for r in recs] == ["a.jpg", "empty.jpg"] and recs[1]["labels"] == []
    assert outline_records(o, scores=[0.123])[0]["scores"] == [0.123]


def test_tp_targets_match_prereg_v1_matching():
    f = Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10),))
    o = outline_set_for("a.jpg", boxes=[[0, 0, 10, 10], [0, 0, 10, 11]],
                        labels=["car", "car"], scores=[0.9, 0.8])
    assert list(tp_targets(o, [f])) == [True, False]
