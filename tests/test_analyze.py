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
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=200, seed=0, seeds=(0, 1))
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
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=50, seed=0, seeds=(0, 1))
    assert r["missing_runs"] == ["BOTH-s1", "SIN-s0", "SIN-s1"]
    assert r["confirmatory"]["H3"]["status"] == "missing arm"
    assert r["holm_family_size"] == 6  # missing tests enter Holm with p = 1
    assert r["confirmatory"]["H2"]["p_holm"] >= 3 * r["confirmatory"]["H2"]["p"] - 1e-12
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
    for text in ("H2", "H4-SIN", "Holm", "t-fog", "t-overcast", "Retention", "invented", "R*",
                 "H1a", "arm − BASE mAP by subset", "re-estimated per replicate"):
        assert text in md
    out = tmp_path / "results"
    assert main(["analyze", "--runs", str(tmp_path / "runs"), "--labels", str(labels),
                 "--manifests", str(manifests), "--out", str(out), "--bootstrap", "20",
                 "--seeds", "0", "1", "--no-guard"]) == 0
    got = json.loads((out / "analysis.json").read_text())
    assert got["n_boot"] == 20 and got["missing_runs"] == []
    assert "# Results" in (out / "report.md").read_text()
    assert (out / "figures" / "pr_curves.pdf").exists()


def test_corrupt_section_relative_performance(tmp_path):
    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames)
    clean = [f["name"] for f in frames
             if (f["attributes"]["weather"], f["attributes"]["timeofday"]) == ("clear", "daytime")]
    write_manifest(manifests / "t-corrupt.csv", [(n, "h") for n in clean[:10]])
    for arm, p in (("BASE", 0.3), ("STY", 0.9)):
        ev = tmp_path / "runs" / f"{arm}-s0" / "eval"
        for c in ("fog", "snow"):
            write_detections(ev / f"t-corrupt-{c}-3.detections.jsonl",
                             _dets(frames, set(clean[:10]), p, 0.2, 7))
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=10, seed=0)
    tc = r["t_corrupt"]
    assert set(tc) == {"BASE-s0", "STY-s0"}
    assert set(tc["STY-s0"]["mAP"]) == {"fog-3", "snow-3"}
    assert tc["STY-s0"]["rPC"] > tc["BASE-s0"]["rPC"]
    from snz.eval.report import render_report
    assert "T-corrupt" in render_report(r)


def test_cis_for_h1a_and_every_subset_and_provenance(tmp_path):
    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames)
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=40, seed=0)
    assert r["missing_runs"] == [f"{a}-s2" for a in ("BASE", "BOTH", "SIN", "STY")]
    h1a = r["descriptive"]["H1a_BASE_retention"]
    assert h1a["ci95"][0] <= h1a["estimate"] <= h1a["ci95"][1]
    sub = r["exploratory"]["subset_mAP_vs_BASE"]
    assert set(sub) == {"t-clean", "t-night", "t-rain", "t-snow", "t-fog", "t-dusk",
                        "t-overcast", "t-shift"}
    assert set(sub["t-night"]) == {"STY", "SIN", "BOTH"} and len(sub["t-night"]["STY"]["ci95"]) == 2
    assert r["thresholds_in_bootstrap"] == "re-estimated per replicate"
    assert set(r["provenance"]) >= {"git_commit", "prereg_tag"}


def test_pycocotools_cross_check(tmp_path):
    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames)
    ev = tmp_path / "runs" / "BASE-s0" / "eval"
    (ev / "val-all.metrics.json").write_text(json.dumps({"mAP": 0.123}))
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=5, seed=0, seeds=(0, 1))
    assert r["coco_check"]["BASE-s0"]["val-all"]["agrees"] is False


def test_pr_curves_and_figures(tmp_path):
    from snz.eval.figures import make_figures

    frames, labels, manifests = _val(tmp_path)
    _runs(tmp_path, frames, skip={("BOTH", 0), ("BOTH", 1)})
    r = analyze(tmp_path / "runs", labels, manifests, n_boot=10, seed=0, seeds=(0, 1))
    pr = r["pr_curves"]
    assert set(pr["t-shift"]) == {"BASE", "STY", "SIN"}  # arm means; BOTH missing
    assert len(pr["t-shift"]["STY"]) == 101 and len(r["pr_recall"]) == 101
    files = make_figures(r, tmp_path / "fig")
    assert sorted(f.name for f in files) == ["pr_curves.pdf", "pr_curves.png",
                                             "subset_diffs.pdf", "subset_diffs.png"]
    assert all(f.stat().st_size > 1000 for f in files)
