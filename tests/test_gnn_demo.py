import json

import numpy as np
import pytest
from PIL import Image

from gnn_helpers import FakeSegmenter, outline_set_for, synthetic_outlines, tiny_idval
from helpers import frame_json, write_image, write_labels
from snz.cli import main
from snz.data.bdd import Box, Frame
from snz.data.manifest import write_manifest
from snz.eval.matching import FD_DUPLICATE, FD_INVENTED, FD_TP
from snz.eval.predict import write_detections
from snz.gnn.demo import (
    build_gallery,
    compare_runs,
    detection_status,
    draw_outline_graph,
    gnn_inputs_color_invariant,
    image_diff,
    recolor,
    render_panel,
    run_live,
    select_examples,
)


def test_detection_status_follows_evaluation_matching():
    f = Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10),))
    rec = {"name": "a.jpg", "boxes": [[0, 0, 10, 11], [0, 0, 10, 10], [200, 200, 210, 210],
                                      [0, 0, 5, 5]],
           "labels": ["car", "car", "car", "train"], "scores": [0.8, 0.9, 0.7, 0.99]}
    status = detection_status(f, rec)
    # the higher-scoring box takes the car; the other is a duplicate; train is not evaluated
    assert list(status) == [FD_DUPLICATE, FD_TP, FD_INVENTED, -1]


def test_image_diff_counts_removed_and_added():
    f = Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10),))
    boxes = [[0, 0, 10, 10], [200, 200, 210, 210], [300, 300, 310, 310]]
    base = {"name": "a.jpg", "boxes": boxes, "labels": ["car"] * 3, "scores": [0.9, 0.8, 0.1]}
    arm = {"name": "a.jpg", "boxes": boxes, "labels": ["car"] * 3, "scores": [0.9, 0.05, 0.6]}
    d = image_diff(f, base, arm, thr_base=0.5, thr_arm=0.5)
    assert (d.removed_false, d.removed_true, d.added_false, d.added_true) == (1, 0, 1, 0)


def test_image_diff_rejects_mismatched_records():
    f = Frame("a.jpg", "clear", "daytime", ())
    base = {"name": "a.jpg", "boxes": [[0, 0, 1, 1]], "labels": ["car"], "scores": [0.9]}
    arm = {"name": "a.jpg", "boxes": [[5, 5, 6, 6]], "labels": ["car"], "scores": [0.9]}
    with pytest.raises(ValueError, match="same detections"):
        image_diff(f, base, arm, thr_base=0.5, thr_arm=0.5)


def test_select_examples_fixed_rule():
    from snz.gnn.demo import ImageDiff

    diffs = [ImageDiff("a", 3, 0, 0, 0), ImageDiff("b", 1, 0, 0, 0), ImageDiff("c", 0, 2, 0, 0),
             ImageDiff("d", 3, 0, 0, 0), ImageDiff("e", 0, 0, 0, 0)]
    best, worst = select_examples(diffs, n_best=2, n_worst=2)
    assert best == ["a", "d"]          # most net false removals, ties by name
    assert worst == ["c"]              # images where real objects were dropped; never empty ones


def test_render_panel_size_and_empty_record():
    img = Image.new("RGB", (64, 48), (10, 10, 10))
    rec = {"name": "a.jpg", "boxes": [], "labels": [], "scores": []}
    p = render_panel(img, rec, threshold=0.5, status=None, title="BASE")
    assert p.size[0] == 64 and p.size[1] > 48       # header strip on top
    rec = {"name": "a.jpg", "boxes": [[5, 5, 30, 30]], "labels": ["car"], "scores": [0.9]}
    p2 = render_panel(img, rec, threshold=0.5, status=np.array([FD_TP]), title="GNN",
                      dropped=np.array([False]))
    assert p2.size == p.size and p2.tobytes() != p.tobytes()


def test_render_panel_downsizes_wide_images():
    img = Image.new("RGB", (1280, 720), (10, 10, 10))
    rec = {"name": "a.jpg", "boxes": [[600, 300, 1270, 710]], "labels": ["car"], "scores": [0.9]}
    p = render_panel(img, rec, threshold=0.5, status=np.array([FD_TP]), title="BASE",
                     max_width=640)
    assert p.size == (640, 360 + 28)
    px = np.asarray(p)
    assert (px[28 + 150, 300:640] == (46, 204, 113)).all(axis=1).any()   # box top edge scaled


def test_draw_outline_graph_handles_degenerate_contour():
    c = np.zeros((64, 2), np.float32)
    g = draw_outline_graph(c, size=120, probs=np.full(10, 0.1), title="car 0.90")
    assert g.size[0] == 120


def test_recolor_modes():
    img = Image.new("RGB", (8, 6), (200, 40, 10))
    assert np.array_equal(np.asarray(recolor(recolor(img, "invert"), "invert")), np.asarray(img))
    g = np.asarray(recolor(img, "gray"))
    assert (g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all()
    h = recolor(img, "hue")
    assert h.size == img.size and np.asarray(h).tobytes() != np.asarray(img).tobytes()
    with pytest.raises(ValueError):
        recolor(img, "sepia")


def test_gnn_inputs_are_color_invariant():
    assert gnn_inputs_color_invariant(synthetic_outlines(color=(255, 0, 0)))
    assert gnn_inputs_color_invariant(outline_set_for("a.jpg", boxes=[]))


def _gallery_fixture(tmp_path):
    imgs = tmp_path / "val"
    frames, base, gnn, rgb = [], [], [], []
    for i in range(4):
        name = f"v{i}.jpg"
        write_image(imgs / name, size=(128, 96))
        frames.append(frame_json(name, timeofday="night", labels=[("car", 10, 10, 50, 40)]))
        boxes = [[10, 10, 50, 40], [80, 50, 110, 80]]           # a real car and a fake one
        base.append({"name": name, "boxes": boxes, "labels": ["car", "car"],
                     "scores": [0.9, 0.8]})
        gnn.append({"name": name, "boxes": boxes, "labels": ["car", "car"],
                    "scores": [0.9, 0.01 if i < 2 else 0.8]})   # GNN drops the fake in v0, v1
        rgb.append({"name": name, "boxes": boxes, "labels": ["car", "car"],
                    "scores": [0.9, 0.01]})
    labels = write_labels(tmp_path / "det_val.json", frames)
    man = tmp_path / "manifests"
    write_manifest(man / "t-night.csv", [(f["name"], "h") for f in frames])
    runs = tmp_path / "runs"
    for arm, recs in (("BASE", base), ("GNN", gnn), ("RGB", rgb)):
        write_detections(runs / f"{arm}-s0" / "eval" / "val-all.detections.jsonl", recs)
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"point": {a: {"threshold_at_R": {"threshold": 0.5}}
                                              for a in ("BASE", "GNN", "RGB")}}))
    return runs, analysis, labels, man, imgs


def test_build_gallery_end_to_end(tmp_path):
    runs, analysis, labels, man, imgs = _gallery_fixture(tmp_path)
    out = tmp_path / "gallery"
    g = build_gallery(runs_root=runs, analysis_path=analysis, labels=labels, manifests_dir=man,
                      image_dir=imgs, out=out, splits=("t-night",), n_best=3, n_worst=3)
    assert g["best"] == ["v0.jpg", "v1.jpg"] and g["worst"] == []
    assert (out / "v0.png").exists() and (out / "gallery.json").exists()
    html = (out / "index.html").read_text()
    assert "selected" in html.lower() and "v0.png" in html
    assert Image.open(out / "v0.png").size[0] == 3 * 128      # BASE | GNN | RGB


def test_gallery_cli(tmp_path):
    runs, analysis, labels, man, imgs = _gallery_fixture(tmp_path)
    assert main(["gnn-demo-gallery", "--runs", str(runs), "--analysis", str(analysis),
                 "--labels", str(labels), "--manifests", str(man), "--images", str(imgs),
                 "--out", str(tmp_path / "g"), "--splits", "t-night"]) == 0
    assert (tmp_path / "g" / "index.html").exists()


def _fake_detector(image_path):
    w, h = Image.open(image_path).size
    return {"name": image_path.name, "boxes": [[10, 10, 50, 40], [70, 20, 85, 70]],
            "labels": ["car", "pedestrian"], "scores": [0.9, 0.6]}


def _trained_models(tmp_path, monkeypatch):
    monkeypatch.setenv("SNZ_GNN_EPOCHS", "1")
    imgs, labels, manifest, outlines = tiny_idval(tmp_path, n_images=8)
    models = tmp_path / "models"
    main(["gnn-train", "--outlines", str(outlines), "--labels", str(labels), "--manifest",
          str(manifest), "--images", str(imgs), "--out", str(models), "--arms", "GNN,RGB",
          "--device", "cpu", "--fold-dir", str(tmp_path / "manifests"), "--no-pretrained"])
    return imgs, models


def test_run_live_and_recolor_comparison(tmp_path, monkeypatch):
    imgs, models = _trained_models(tmp_path, monkeypatch)
    image = imgs / "f0.jpg"
    thresholds = {"BASE": 0.5, "GNN": 0.2, "RGB": 0.2}
    a = run_live(image, detector=_fake_detector, segmenter=FakeSegmenter(), models_dir=models,
                 arms=("BASE", "GNN", "RGB"), thresholds=thresholds, device="cpu")
    assert set(a.timings_ms) >= {"detector", "outlines", "GNN", "RGB", "total"}
    assert len(a.outlines.score) == 2 and set(a.scores) == {"BASE", "GNN", "RGB"}
    assert a.kept["BASE"].tolist() == [True, True]
    recolored = tmp_path / "rec.png"
    recolor(Image.open(image), "invert").save(recolored)
    b = run_live(recolored, detector=_fake_detector, segmenter=FakeSegmenter(),
                 models_dir=models, arms=("BASE", "GNN", "RGB"), thresholds=thresholds,
                 device="cpu")
    c = compare_runs(a, b)
    assert c["matched"] == 2 and c["outline_iou_mean"] == pytest.approx(1.0)
    assert c["gnn_input_color_invariant"] is True
    assert 0 <= c["gnn_verdict_agreement"] <= 1


def test_live_cli_writes_outputs(tmp_path, monkeypatch):
    imgs, models = _trained_models(tmp_path, monkeypatch)
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"point": {a: {"threshold_at_R": {"threshold": 0.3}}
                                              for a in ("BASE", "GNN", "RGB")}}))
    monkeypatch.setattr("snz.gnn.demo.load_detector", lambda *a, **k: _fake_detector)
    monkeypatch.setattr("snz.gnn.demo.make_segmenter", lambda *a, **k: FakeSegmenter())
    out = tmp_path / "live"
    assert main(["gnn-demo-live", "--image", str(imgs / "f0.jpg"), "--checkpoint",
                 str(tmp_path / "final.pt"), "--models", str(models), "--analysis",
                 str(analysis), "--arms", "BASE,GNN,RGB", "--recolor", "invert",
                 "--device", "cpu", "--out", str(out)]) == 0
    result = json.loads((out / "live.json").read_text())
    assert "timings_ms" in result["original"] and "recolor" in result
    assert (out / "live.png").exists() and (out / "live_recolored.png").exists()
    assert (out / "live_graphs.png").exists()
