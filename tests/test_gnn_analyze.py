import json

import numpy as np
import pytest

from snz.data.bdd import Box, Frame
from snz.eval.matching import match_split
from snz.eval.operating import threshold_for_false_rate
from snz.eval.predict import write_detections
from snz.gnn.analyze import gnn_analyze
from test_analyze import _val


def test_threshold_for_false_rate():
    frames = [Frame("a", "c", "d", (Box("car", 0, 0, 10, 10), Box("car", 50, 0, 60, 10))),
              Frame("b", "c", "d", ())]
    dets = [{"name": "a", "boxes": [[0, 0, 10, 10], [300, 300, 310, 310], [50, 0, 60, 10]],
             "labels": ["car"] * 3, "scores": [0.9, 0.8, 0.7]},
            {"name": "b", "boxes": [[300, 300, 310, 310]], "labels": ["car"], "scores": [0.6]}]
    m = match_split(frames, dets)
    assert threshold_for_false_rate(m, 0.5).threshold == pytest.approx(0.7)  # 1 FD / 2 images
    op = threshold_for_false_rate(m, 0.0)
    assert op.threshold == pytest.approx(0.9) and op.flagged is False


def _planted(tmp_path):
    frames, labels, manifests = _val(tmp_path)
    rng = np.random.default_rng(0)
    base, gnn = [], []
    for f in frames:
        boxes, labs, sb, sg = [], [], [], []
        for lab in f["labels"]:
            b = lab["box2d"]
            if rng.random() < 0.8:
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                s = float(rng.uniform(0.4, 1))
                sb.append(s)
                sg.append(s)
        for _ in range(rng.poisson(1.5)):
            x, y = rng.uniform(600, 900, 2)
            boxes.append([x, y, x + 30, y + 30])
            sb.append(float(rng.uniform(0.3, 0.95)))
            sg.append(0.01)
        labs = ["car"] * len(boxes)
        base.append({"name": f["name"], "boxes": boxes, "labels": labs, "scores": sb})
        gnn.append({"name": f["name"], "boxes": boxes, "labels": labs, "scores": sg})
    runs = tmp_path / "gnn_runs"
    for arm, recs in (("BASE", base), ("GNN", gnn)):
        write_detections(runs / f"{arm}-s0" / "eval" / "val-all.detections.jsonl", recs)
        clean = {n for n, *_ in (line.split(",") for line in
                                 (manifests / "t-clean.csv").read_text().splitlines()[1:])}
        write_detections(runs / f"{arm}-s0" / "eval" / "t-texture.detections.jsonl",
                         [r for r in recs if r["name"] in clean])
    return runs, labels, manifests


def test_planted_gnn_improvement_is_detected(tmp_path):
    runs, labels, manifests = _planted(tmp_path)
    res = gnn_analyze(runs, labels, manifests, n_boot=200)
    c = res["confirmatory"]
    assert c["G1"]["supported"] and c["G2"]["supported"]
    assert c["G3"]["status"] == "missing arm" and c["G3"]["p_holm"] == 1.0
    assert "RGB-s0" in res["missing"]
    json.dumps(res)


def test_gnn_report(tmp_path):
    from snz.gnn.analyze import render_gnn_report

    runs, labels, manifests = _planted(tmp_path)
    md = render_gnn_report(gnn_analyze(runs, labels, manifests, n_boot=30))
    for text in ("G1", "G2", "G3", "Holm", "R*", "F*", "t-shift", "invented"):
        assert text in md
