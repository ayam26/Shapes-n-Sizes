import json

from snz.audit.sample import count_detections, sample_audit


def _dets(arm_seed, n_frames=20, per_frame=12):
    out = []
    for i in range(n_frames):
        scores = [round((k + arm_seed) % per_frame / per_frame, 3) for k in range(per_frame)]
        boxes = [[k, k, k + 10, k + 10] for k in range(per_frame)]
        out.append({"name": f"f{i}.jpg", "boxes": boxes, "labels": ["car"] * per_frame,
                    "scores": scores})
    return out


def test_sample_audit_is_blind_balanced_and_deterministic():
    dets = {"BASE": _dets(0), "STY": _dets(3), "SIN": _dets(5), "BOTH": _dets(7)}
    th = {"BASE": 0.5, "STY": 0.5, "SIN": 0.5, "BOTH": 0.5}
    items, key = sample_audit(dets, th, n_per_arm=30, seed=0)
    assert len(items) == 120 and len(key) == 120
    assert sorted(v["arm"] for v in key.values()).count("STY") == 30
    text = json.dumps(items)
    for secret in ("BASE", "STY", "SIN", "BOTH", "score"):
        assert secret not in text
    assert set(items[0]) == {"id", "frame", "box", "class"}
    for item in items:  # every sampled detection is above its arm's threshold
        k = key[item["id"]]
        rec = next(d for d in dets[k["arm"]] if d["name"] == k["frame"])
        assert rec["scores"][k["det"]] >= 0.5 and rec["boxes"][k["det"]] == item["box"]
    assert sample_audit(dets, th, n_per_arm=30, seed=0) == (items, key)
    assert sample_audit(dets, th, n_per_arm=30, seed=1)[0] != items


def test_small_pools_are_taken_whole_and_counts():
    dets = {"BASE": _dets(0, n_frames=1)}
    items, _ = sample_audit(dets, {"BASE": 0.9}, n_per_arm=150)
    assert len(items) == count_detections(dets["BASE"], 0.9) == 1  # only score .917


def test_build_audit_page(tmp_path):
    from helpers import write_image
    from snz.audit.page import build_audit_page

    write_image(tmp_path / "frames" / "f0.jpg", size=(64, 48))
    items = [{"id": "a000", "frame": "f0.jpg", "box": [5, 5, 30, 30], "class": "car"},
             {"id": "a001", "frame": "f0.jpg", "box": [40, 10, 60, 40], "class": "truck"}]
    build_audit_page(items, tmp_path / "frames", tmp_path / "page")
    html = (tmp_path / "page" / "index.html").read_text()
    assert "a000" in html and "a001" in html and "truck" in html
    assert (tmp_path / "page" / "items" / "a001.jpg").exists()
    for label in ("correct", "wrong_class", "invented", "unsure"):
        assert label in html
    for arm in ("BASE", "STY", "SIN", "BOTH"):
        assert arm not in html


def test_train_class_detections_are_neither_counted_nor_sampled():
    dets = {"BASE": [{"name": "f.jpg", "boxes": [[0, 0, 5, 5], [0, 0, 9, 9]],
                      "labels": ["train", "car"], "scores": [0.99, 0.6]}]}
    items, _ = sample_audit(dets, {"BASE": 0.5})
    assert [i["class"] for i in items] == ["car"]
    assert count_detections(dets["BASE"], 0.5) == 1


def test_page_storage_key_depends_on_the_items(tmp_path):
    from helpers import write_image
    from snz.audit.page import build_audit_page

    write_image(tmp_path / "fr" / "f.jpg", size=(64, 48))
    a = [{"id": "a000", "frame": "f.jpg", "box": [1, 1, 9, 9], "class": "car"}]
    b = [{"id": "a000", "frame": "f.jpg", "box": [2, 2, 9, 9], "class": "bus"}]
    ha = build_audit_page(a, tmp_path / "fr", tmp_path / "pa").read_text()
    hb = build_audit_page(b, tmp_path / "fr", tmp_path / "pb").read_text()
    key = lambda h: h.split('const KEY = "')[1].split('"')[0]  # noqa: E731
    assert key(ha) != key(hb)
