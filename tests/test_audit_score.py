import json

import pytest

from snz.audit.score import cohen_kappa, footage_recall, load_coco_labels, score_audit, wilson


def test_wilson_reference_values():
    p, lo, hi = wilson(8, 10)
    assert p == 0.8 and (lo, hi) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert wilson(0, 0) == (None, None, None)


def test_score_audit_estimates_real_and_invented():
    key = {f"a{i:03d}": {"arm": "BASE" if i < 10 else "STY", "frame": "f", "det": i}
           for i in range(20)}
    labels = {f"a{i:03d}": lab for i, lab in enumerate(
        ["correct"] * 6 + ["invented"] * 2 + ["unsure", "wrong_class"]
        + ["correct"] * 9 + ["invented"])}
    r = score_audit(labels, key, {"BASE": 1000, "STY": 800})
    base = r["BASE"]
    assert (base["items"], base["unsure"], base["n"]) == (10, 1, 9)
    assert base["precision"]["p"] == pytest.approx(6 / 9)
    assert base["est_real"]["estimate"] == pytest.approx(1000 * 6 / 9)
    assert base["est_invented"]["estimate"] == pytest.approx(1000 * 2 / 9)
    lo, hi = base["est_real"]["ci95"]
    assert lo < 1000 * 6 / 9 < hi
    assert r["STY"]["precision"]["p"] == pytest.approx(0.9) and r["STY"]["unlabeled"] == 0


def test_cohen_kappa():
    a = {"1": "correct", "2": "invented", "3": "correct", "4": "unsure"}
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    b = {"1": "invented", "2": "correct", "3": "invented", "4": "correct"}
    assert cohen_kappa(a, b) < 0


def test_load_coco_labels_and_footage_recall(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "frames/d1_00000000ms.jpg"}],
            "categories": [{"id": 3, "name": "car"}, {"id": 1, "name": "person"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10]},
                            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 5, 15]}]}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(coco))
    frames = load_coco_labels(path)
    assert frames[0].name == "d1_00000000ms.jpg"
    assert [(b.category, b.x2, b.y2) for b in frames[0].boxes] == [
        ("car", 10, 10), ("pedestrian", 25, 35)]
    dets = {"BASE": [{"name": "d1_00000000ms.jpg", "boxes": [[0, 0, 10, 10], [300, 0, 310, 9]],
                      "labels": ["car", "car"], "scores": [0.9, 0.8]},
                     {"name": "unlabeled.jpg", "boxes": [], "labels": [], "scores": []}]}
    r = footage_recall(frames, dets, {"BASE": 0.5})
    assert r["BASE"]["recall"] == pytest.approx(0.5)
    assert r["BASE"]["false_per_frame"] == pytest.approx(1.0)
    assert r["BASE"]["by_type"]["invented"] == pytest.approx(1.0)


def test_unknown_coco_category_is_an_error(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"images": [], "annotations": [],
                                "categories": [{"id": 1, "name": "dog"}]}))
    with pytest.raises(ValueError, match="dog"):
        load_coco_labels(path)


def test_g7_cli_end_to_end(tmp_path):
    from helpers import write_image
    from snz.cli import main
    from snz.data.manifest import write_manifest
    from test_train import _run, tiny_recipe

    run = tmp_path / "runs" / "BASE-s0"
    _run(tmp_path, run, tiny_recipe())
    for i in range(3):
        write_image(tmp_path / "foot" / "frames" / f"d1_{i:08d}ms.jpg", size=(64, 48))
    write_manifest(tmp_path / "foot" / "footage.csv",
                   [(f"d1_{i:08d}ms.jpg", "h") for i in range(3)])
    analysis = tmp_path / "analysis.json"
    predict_args = ["footage-predict", "--manifest", str(tmp_path / "foot" / "footage.csv"),
                    "--frames", str(tmp_path / "foot" / "frames"), "--analysis", str(analysis),
                    "--checkpoint", str(run / "final.pt"), "--device", "cpu", "--no-guard"]
    with pytest.raises(SystemExit, match="analysis.json"):
        main(predict_args)  # thresholds must be fixed first
    analysis.write_text(json.dumps(
        {"footage_thresholds": {"BASE-s0": {"threshold": 0.0, "recall": 0.5, "flagged": False}}}))
    assert main(predict_args) == 0
    assert (run / "eval" / "footage.detections.jsonl").exists()
    # the tiny CPU model may detect nothing; give the audit steps known detections
    from snz.eval.predict import write_detections

    write_detections(run / "eval" / "footage.detections.jsonl", [
        {"name": f"d1_{i:08d}ms.jpg", "boxes": [[2, 2, 30, 30], [5, 5, 20, 40]],
         "labels": ["car", "pedestrian"], "scores": [0.9, 0.7]} for i in range(3)])
    assert main(["audit-sample", "--runs", str(tmp_path / "runs"), "--analysis", str(analysis),
                 "--frames", str(tmp_path / "foot" / "frames"), "--out", str(tmp_path / "page"),
                 "--key", str(tmp_path / "key.json"), "--n", "5"]) == 0
    key = json.loads((tmp_path / "key.json").read_text())
    assert (tmp_path / "page" / "index.html").exists() and 0 < len(key) <= 5
    labels = tmp_path / "audit_labels.csv"
    labels.write_text("item_id,label\n" + "".join(f"{i},invented\n" for i in key))
    out = tmp_path / "footage.json"
    assert main(["audit-score", "--labels", str(labels), "--key", str(tmp_path / "key.json"),
                 "--runs", str(tmp_path / "runs"), "--analysis", str(analysis),
                 "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["audit"]["BASE"]["precision"]["p"] == 0.0
    assert r["detections"]["BASE"]["frames"] == 3


def test_kappa_undefined_when_both_raters_use_one_label():
    a = {"1": "correct", "2": "correct"}
    assert cohen_kappa(a, a) is None


def test_footage_recall_has_bootstrap_ci_and_per_class_counts():
    from snz.audit.score import detection_counts
    from snz.data.bdd import Box, Frame

    frames = [Frame(f"f{i}.jpg", "u", "u", (Box("car", 0, 0, 10, 10),)) for i in range(20)]
    dets = {"BASE": [{"name": f"f{i}.jpg", "boxes": [[0, 0, 10, 10]] if i % 2 else [],
                      "labels": ["car"] if i % 2 else [], "scores": [0.9] if i % 2 else []}
                     for i in range(20)]}
    r = footage_recall(frames, dets, {"BASE": 0.5}, n_boot=200)
    lo, hi = r["BASE"]["recall_ci95"]
    assert r["BASE"]["recall"] == pytest.approx(0.5) and lo < 0.5 < hi
    c = detection_counts(dets["BASE"], 0.5)
    assert c["per_class"] == {"car": 10} and c["per_frame_mean"] == pytest.approx(0.5)
    assert c["frames"] == 20
