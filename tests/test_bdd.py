import json

import pytest

from helpers import frame_json, write_labels
from snz.data.bdd import CLASSES, EVAL_CLASSES, Frame, build_splits, load_frames, select_frames


def test_classes():
    assert len(CLASSES) == 10
    assert EVAL_CLASSES == tuple(c for c in CLASSES if c != "train")


def test_load_frames_normalizes_and_drops(tmp_path):
    frames = [
        frame_json("a.jpg", labels=[
            ("car", 0, 0, 10, 10),
            ("person", 1, 1, 5, 9),        # 2018 alias -> pedestrian
            ("bike", 1, 1, 5, 9),          # alias -> bicycle
            ("motor", 1, 1, 5, 9),         # alias -> motorcycle
            ("other vehicle", 0, 0, 5, 5), # unknown -> dropped, counted
            ("trailer", 0, 0, 5, 5),
            ("car", 5, 5, 5.5, 20),        # width < 1 px -> dropped
        ]),
        {"name": "b.jpg", "attributes": {"weather": "rainy", "timeofday": "night"}, "labels": None},
        {"name": "c.jpg"},                 # no attributes, no labels
    ]
    loaded, stats = load_frames(write_labels(tmp_path / "l.json", frames))
    a, b, c = loaded
    assert [x.category for x in a.boxes] == ["car", "pedestrian", "bicycle", "motorcycle"]
    assert b.boxes == () and (b.weather, b.timeofday) == ("rainy", "night")
    assert (c.weather, c.timeofday) == ("undefined", "undefined")
    assert stats.frames == 3 and stats.boxes == 4
    assert stats.dropped_unknown == {"other vehicle": 1, "trailer": 1}
    assert stats.dropped_degenerate == 1


def test_load_frames_ignores_non_box_labels(tmp_path):
    item = frame_json("a.jpg", labels=[("car", 0, 0, 10, 10)])
    item["labels"].append({"id": "9", "category": "lane", "poly2d": []})
    loaded, _ = load_frames(write_labels(tmp_path / "l.json", [item]))
    assert len(loaded[0].boxes) == 1


def test_load_frames_rejects_duplicate_names(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps([frame_json("a.jpg"), frame_json("a.jpg")]))
    with pytest.raises(ValueError, match="a.jpg"):
        load_frames(path)


def _f(name, weather, tod):
    return Frame(name, weather, tod, ())


def test_build_splits_membership():
    train = [_f(f"tr{i}.jpg", "clear", "daytime") for i in range(20)] + [
        _f("tr_night.jpg", "clear", "night"),
        _f("tr_rain.jpg", "rainy", "daytime"),
    ]
    val = [
        _f("v_clean.jpg", "clear", "daytime"),
        _f("v_night_clear.jpg", "clear", "night"),
        _f("v_night_rain.jpg", "rainy", "night"),
        _f("v_rain.jpg", "rainy", "daytime"),
        _f("v_snow.jpg", "snowy", "daytime"),
        _f("v_fog.jpg", "foggy", "daytime"),
        _f("v_dusk.jpg", "clear", "dawn/dusk"),
        _f("v_overcast.jpg", "overcast", "daytime"),
        _f("v_pcloudy.jpg", "partly cloudy", "daytime"),
        _f("v_undef.jpg", "undefined", "daytime"),
    ]
    s = build_splits(train, val, seed=0, idval_fraction=0.1)
    assert list(s) == ["train", "id-val", "t-clean", "t-night", "t-rain", "t-snow", "t-fog"]
    assert len(s["id-val"]) == 2 and len(s["train"]) == 18
    assert not set(s["train"]) & set(s["id-val"])
    assert "tr_night.jpg" not in s["train"] + s["id-val"]
    assert s["t-clean"] == ["v_clean.jpg"]
    assert s["t-night"] == ["v_night_clear.jpg", "v_night_rain.jpg"]
    assert (s["t-rain"], s["t-snow"], s["t-fog"]) == (["v_rain.jpg"], ["v_snow.jpg"], ["v_fog.jpg"])
    used = {n for split in s.values() for n in split}
    for excluded in ("v_dusk.jpg", "v_overcast.jpg", "v_pcloudy.jpg", "v_undef.jpg"):
        assert excluded not in used


def test_build_splits_deterministic_per_seed():
    train = [_f(f"tr{i:03d}.jpg", "clear", "daytime") for i in range(200)]
    a = build_splits(train, [], seed=0, idval_fraction=0.1)
    b = build_splits(list(reversed(train)), [], seed=0, idval_fraction=0.1)
    c = build_splits(train, [], seed=1, idval_fraction=0.1)
    assert a == b
    assert a["id-val"] != c["id-val"]


def test_select_frames_keeps_manifest_order_and_reports_missing():
    frames = [_f("a.jpg", "clear", "daytime"), _f("b.jpg", "clear", "daytime")]
    assert [f.name for f in select_frames(frames, ["b.jpg", "a.jpg"])] == ["b.jpg", "a.jpg"]
    with pytest.raises(KeyError, match="1 manifest name"):
        select_frames(frames, ["a.jpg", "zzz.jpg"])
