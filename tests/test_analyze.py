import json

import numpy as np
import pytest

from helpers import frame_json, write_labels
from snz.data.manifest import write_manifest
from snz.eval.analyze import analyze
from snz.eval.predict import write_detections

ATTRS = ([("clear", "daytime")] * 30 + [("clear", "night")] * 20 + [("rainy", "daytime")] * 10
         + [("snowy", "daytime")] * 6 + [("foggy", "daytime")] * 2
         + [("clear", "dawn/dusk")] * 4 + [("overcast", "daytime")] * 4)


def _val(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for i, (weather, tod) in enumerate(ATTRS):
        labels = []
        for _ in range(rng.integers(1, 4)):
            x, y = rng.uniform(0, 500, 2)
            labels.append(("car", x, y, x + 60, y + 40))
        frames.append(frame_json(f"v{i:03d}.jpg", weather, tod, labels))
    labels_path = write_labels(tmp_path / "det_val.json", frames)
    m = tmp_path / "manifests"
    groups = {"t-clean": ("clear", "daytime"), "t-night": None, "t-rain": ("rainy", "daytime"),
              "t-snow": ("snowy", "daytime"), "t-fog": ("foggy", "daytime"),
              "t-dusk": None, "t-overcast": ("overcast", "daytime")}
    for split, key in groups.items():
        if split == "t-night":
            names = [f["name"] for f in frames if f["attributes"]["timeofday"] == "night"]
        elif split == "t-dusk":
            names = [f["name"] for f in frames if f["attributes"]["timeofday"] == "dawn/dusk"]
        else:
            names = [f["name"] for f in frames
                     if (f["attributes"]["weather"], f["attributes"]["timeofday"]) == key]
        write_manifest(m / f"{split}.csv", [(n, "h") for n in names])
    write_manifest(m / "val-all.csv", [(f["name"], "h") for f in frames])
    return frames, labels_path, m


def _dets(frames, names, p_hit, n_fp, seed):
    rng = np.random.default_rng(seed)
    out = []
    for f in frames:
        if f["name"] not in names:
            continue
        boxes, scores = [], []
        for lab in f["labels"]:
            b = lab["box2d"]
            if rng.random() < p_hit:
                boxes.append([b["x1"] + 1, b["y1"], b["x2"], b["y2"] + 1])
                scores.append(float(rng.uniform(0.5, 1)))
        for _ in range(rng.poisson(n_fp)):
            x, y = rng.uniform(600, 900, 2)
            boxes.append([x, y, x + 30, y + 30])
            scores.append(float(rng.uniform(0.05, 0.9)))
        out.append({"name": f["name"], "boxes": boxes, "labels": ["car"] * len(boxes),
                    "scores": scores})
    return out


def _runs(tmp_path, frames, skip=()):
    shift = {f["name"] for f in frames if f["attributes"]["timeofday"] == "night"
             or f["attributes"]["weather"] in ("rainy", "snowy", "foggy")}
    clean = {f["name"] for f in frames
             if (f["attributes"]["weather"], f["attributes"]["timeofday"]) == ("clear", "daytime")}
    quality = {"BASE": (0.55, 2.0), "STY": (0.95, 0.3), "SIN": (0.6, 1.8), "BOTH": (0.9, 0.5)}
    for a, (arm, (p, fp)) in enumerate(quality.items()):
        for seed in (0, 1):
            if (arm, seed) in skip:
                continue
            ev = tmp_path / "runs" / f"{arm}-s{seed}" / "eval"
            s = 100 * a + seed
            all_names = {f["name"] for f in frames}
            dets = (_dets(frames, shift, p, fp, s)
                    + _dets(frames, all_names - shift, 0.9, 0.5, s + 50))
            write_detections(ev / "val-all.detections.jsonl", dets)
            write_detections(ev / "t-texture.detections.jsonl",
                             _dets(frames, clean, p, fp, s + 70))


def test_analyze_detects_planted_effect_and_reports_everything(tmp_path):
    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames)
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=200, seed=0)
    json.dumps(r)  # serializable
    c = r["confirmatory"]
    assert set(c) == {"H1b-STY", "H1b-SIN", "H2", "H3", "H4-STY", "H4-SIN"}
    assert c["H2"]["p"] < 0.05 and c["H2"]["supported"]
    assert c["H4-STY"]["estimate"] < 0 and c["H4-STY"]["supported"]
    assert c["H2"]["p_holm"] >= c["H2"]["p"]
    assert r["operating_point"]["R_star"] > 0
    assert set(r["thresholds"]["t-shift"]) == {f"{a}-s{s}" for a in
                                               ("BASE", "STY", "SIN", "BOTH") for s in (0, 1)}
    arm = r["arms"]["STY"]
    assert set(arm["subsets"]) >= {"t-clean", "t-shift", "t-fog", "t-dusk", "t-overcast"}
    assert arm["subsets"]["t-shift"]["mAP"]["n"] == 2
    assert "additivity_shift_mAP" in r["exploratory"]
    assert set(r["footage_thresholds"]) == set(r["thresholds"]["t-shift"])


def test_missing_runs_are_reported_not_fatal(tmp_path):
    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames, skip={("SIN", 0), ("SIN", 1), ("BOTH", 1)})
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=50, seed=0)
    assert r["missing_runs"] == ["BOTH-s1", "SIN-s0", "SIN-s1"]
    assert r["confirmatory"]["H3"]["status"] == "missing arm"
    assert r["arms"]["BOTH"]["subsets"]["t-clean"]["mAP"]["n"] == 1


def test_base_seed0_is_required(tmp_path):
    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames, skip={("BASE", 0)})
    with pytest.raises(FileNotFoundError, match="BASE-s0"):
        analyze(tmp_path / "runs", labels, manifests, n_boot=10, seed=0)


def test_report_and_cli(tmp_path):
    from snz.cli import main
    from snz.eval.report import render_report

    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames)
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=30, seed=0)
    md = render_report(r)
    for text in ("H2", "H4-SIN", "Holm", "t-fog", "t-overcast", "Retention", "invented", "R*"):
        assert text in md
    out = tmp_path / "results"
    assert main(["analyze", "--runs", str(tmp_path / "runs"), "--labels", str(labels),
                 "--manifests", str(manifests), "--out", str(out), "--bootstrap", "20",
                 "--no-guard"]) == 0
    assert json.loads((out / "analysis.json").read_text())["n_boot"] == 20
    assert "# Results" in (out / "report.md").read_text()
