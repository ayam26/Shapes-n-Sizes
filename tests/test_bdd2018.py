import json
import zipfile

from snz.cli import main
from snz.data.bdd import load_frames
from snz.data.bdd2018 import convert_2018


def _zip(path):
    def frame(name, weather, objects):
        return {"name": name, "attributes": {"weather": weather, "timeofday": "daytime",
                                             "scene": "highway"},
                "frames": [{"timestamp": 10000, "objects": objects}]}
    car = {"category": "car", "id": 0, "attributes": {"occluded": False},
           "box2d": {"x1": 1.5, "y1": 2.0, "x2": 30.0, "y2": 40.0}}
    person = {"category": "person", "id": 1, "box2d": {"x1": 5, "y1": 5, "x2": 9, "y2": 20}}
    lane = {"category": "lane/road curb", "id": 2, "poly2d": [[0, 0, "L"]]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("100k/", "")
        z.writestr("100k/train/b.json", json.dumps(frame("b", "rainy", [person])))
        z.writestr("100k/train/a.json", json.dumps(frame("a", "clear", [car, lane])))
        z.writestr("100k/val/v.json", json.dumps(frame("v", "clear", [])))
        z.writestr("100k/test/t.json", json.dumps(frame("t", "clear", [car])))
    return path


def test_convert_2018_to_det20_shape(tmp_path):
    frames = convert_2018(_zip(tmp_path / "labels.zip"), "train")
    assert [f["name"] for f in frames] == ["a.jpg", "b.jpg"]
    a = frames[0]
    assert a["attributes"]["weather"] == "clear"
    assert a["labels"] == [{"id": "0", "category": "car",
                            "box2d": {"x1": 1.5, "y1": 2.0, "x2": 30.0, "y2": 40.0}}]
    assert convert_2018(tmp_path / "labels.zip", "val")[0]["labels"] == []


def test_cli_writes_files_load_frames_accepts(tmp_path):
    z = _zip(tmp_path / "labels.zip")
    assert main(["convert-labels-2018", "--zip", str(z), "--out", str(tmp_path / "det")]) == 0
    train, stats = load_frames(tmp_path / "det" / "det_train.json")
    assert [b.category for f in train for b in f.boxes] == ["car", "pedestrian"]
    val, _ = load_frames(tmp_path / "det" / "det_val.json")
    assert [f.name for f in val] == ["v.jpg"]
    assert not (tmp_path / "det" / "det_test.json").exists()
