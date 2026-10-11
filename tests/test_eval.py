import subprocess
import sys

import pytest
import torch
from torchvision.models import resnet50

from helpers import write_image
from snz.data.bdd import Box, Frame
from snz.eval.coco_eval import evaluate
from snz.eval.guard import ensure_split_allowed
from snz.eval.predict import predict, read_detections, write_detections
from snz.train.model import build_model


def _frame(name, *boxes):
    return Frame(name, "clear", "daytime", tuple(Box(*b) for b in boxes))


def _det(name, *items):
    return {"name": name, "boxes": [list(i[1:5]) for i in items],
            "labels": [i[0] for i in items], "scores": [i[5] for i in items]}


FRAMES = [
    _frame("a.jpg", ("car", 0, 0, 10, 10), ("pedestrian", 20, 20, 30, 40)),
    _frame("b.jpg", ("car", 5, 5, 25, 25)),
]


def test_perfect_detections_score_one():
    dets = [_det("a.jpg", ("car", 0, 0, 10, 10, 0.9), ("pedestrian", 20, 20, 30, 40, 0.8)),
            _det("b.jpg", ("car", 5, 5, 25, 25, 0.7))]
    r = evaluate(FRAMES, dets)
    assert r["mAP"] == pytest.approx(1.0) and r["AP50"] == pytest.approx(1.0)
    assert r["per_class"]["car"] == pytest.approx(1.0)
    assert r["per_class"]["bus"] is None  # no ground truth: excluded, not zero


def test_hand_computed_half_recall():
    # one class, 2 GT, one exact hit: precision 1 up to recall 0.5 -> 51/101 per IoU threshold
    frames = [_frame("a.jpg", ("car", 0, 0, 10, 10), ("car", 50, 50, 60, 60))]
    r = evaluate(frames, [_det("a.jpg", ("car", 0, 0, 10, 10, 0.9))])
    assert r["mAP"] == pytest.approx(51 / 101, abs=1e-6)


def test_train_class_and_missing_images_and_empty():
    frames = FRAMES + [_frame("c.jpg", ("train", 0, 0, 50, 50))]
    dets = [_det("a.jpg", ("car", 0, 0, 10, 10, 0.9), ("train", 0, 0, 5, 5, 0.9))]
    r = evaluate(frames, dets)  # b.jpg / c.jpg have no detection records: treated as empty
    assert "train" not in r["per_class"] and r["images"] == 3
    assert evaluate(FRAMES, [])["mAP"] == 0.0
    with pytest.raises(ValueError, match="zzz.jpg"):
        evaluate(FRAMES, [_det("zzz.jpg")])


def test_guard_requires_prereg_tag(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    ensure_split_allowed("id-val", tmp_path)
    for split in ("t-clean", "t-night", "t-texture", "footage"):
        with pytest.raises(PermissionError, match="prereg-v1"):
            ensure_split_allowed(split, tmp_path)
    subprocess.run(["git", "tag", "prereg-v1"], cwd=tmp_path, check=True)
    ensure_split_allowed("t-clean", tmp_path)


def test_predict_and_detections_roundtrip(tmp_path):
    frames = [_frame(f"f{i}.jpg", ("car", 5, 5, 30, 30)) for i in range(3)]
    for f in frames:
        write_image(tmp_path / "img" / f.name, size=(64, 48))
    torch.manual_seed(0)
    model = build_model(resnet50(weights=None).state_dict(), min_size=48, max_size=64,
                        box_score_thresh=0.0)
    records = predict(model, frames, tmp_path / "img", device="cpu", batch_size=2, num_workers=0)
    assert [r["name"] for r in records] == ["f0.jpg", "f1.jpg", "f2.jpg"]
    write_detections(tmp_path / "d.jsonl", records)
    assert read_detections(tmp_path / "d.jsonl") == records
    evaluate(frames, records)  # runs end to end on real model output


def test_guard_rejects_manifest_that_does_not_match_split(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    m = tmp_path / "manifests"
    m.mkdir()
    (m / "id-val.csv").write_text("name,sha256\na.jpg,h1\n")
    (m / "t-night.csv").write_text("name,sha256\nn.jpg,h2\n")
    ensure_split_allowed("id-val", tmp_path, manifest=m / "id-val.csv")
    with pytest.raises(ValueError, match="t-night.csv"):
        ensure_split_allowed("id-val", tmp_path, manifest=m / "t-night.csv")
    # test images inside a file named id-val.csv are still a test split
    (m / "id-val.csv").write_text("name,sha256\na.jpg,h1\nn.jpg,h2\n")
    with pytest.raises(PermissionError, match="t-night"):
        ensure_split_allowed("id-val", tmp_path, manifest=m / "id-val.csv")


def test_ignore_regions_do_not_count_as_false_detections():
    frames = [Frame("a.jpg", "clear", "daytime", (
        Box("car", 0, 0, 10, 10),
        Box("car", 50, 50, 70, 70, ignore=True),  # e.g. "other vehicle"
    ))]
    dets = [_det("a.jpg", ("car", 0, 0, 10, 10, 0.9), ("car", 50, 50, 70, 70, 0.95))]
    r = evaluate(frames, dets)
    assert r["mAP"] == pytest.approx(1.0)


def test_guard_pairs_t_texture_with_t_clean_manifest(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "prereg-v1"], cwd=tmp_path, check=True)
    m = tmp_path / "manifests"
    m.mkdir()
    (m / "t-clean.csv").write_text("name,sha256\na.jpg,h1\n")
    (m / "id-val.csv").write_text("name,sha256\nb.jpg,h2\n")
    ensure_split_allowed("t-texture", tmp_path, manifest=m / "t-clean.csv")
    with pytest.raises(ValueError, match="id-val.csv"):
        ensure_split_allowed("t-texture", tmp_path, manifest=m / "id-val.csv")


def test_predict_reads_mapped_files_but_keeps_names(tmp_path):
    frames = [_frame("f0.jpg", ("car", 5, 5, 30, 30))]
    write_image(tmp_path / "sty" / "f0_s1.jpg", size=(64, 48))
    torch.manual_seed(0)
    model = build_model(resnet50(weights=None).state_dict(), min_size=48, max_size=64)
    records = predict(model, frames, tmp_path / "sty", device="cpu", num_workers=0,
                      file_name=lambda n: n.replace(".jpg", "_s1.jpg"))
    assert [r["name"] for r in records] == ["f0.jpg"]


def test_guard_pairs_t_corrupt_splits_with_their_manifest(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    m = tmp_path / "manifests"
    m.mkdir()
    (m / "t-corrupt.csv").write_text("name,sha256\na.jpg,h1\n")
    with pytest.raises(PermissionError):  # held out before the tag
        ensure_split_allowed("t-corrupt-fog-3", tmp_path, manifest=m / "t-corrupt.csv")
    subprocess.run(["git", "tag", "prereg-v1"], cwd=tmp_path, check=True)
    ensure_split_allowed("t-corrupt-fog-3", tmp_path, manifest=m / "t-corrupt.csv")
    with pytest.raises(ValueError):
        ensure_split_allowed("t-corrupt-fog-3", tmp_path, manifest=m / "t-clean.csv")


def test_guard_checks_descriptive_and_val_all_overlap(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    m = tmp_path / "manifests"
    m.mkdir()
    (m / "id-val.csv").write_text("name,sha256\nd.jpg,h1\n")
    (m / "t-dusk.csv").write_text("name,sha256\nd.jpg,h1\n")
    with pytest.raises(PermissionError, match="t-dusk"):
        ensure_split_allowed("id-val", tmp_path, manifest=m / "id-val.csv")


def test_scores_are_not_rounded(tmp_path):
    class Fake(torch.nn.Module):
        def forward(self, images):
            return [{"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                     "labels": torch.tensor([3]), "scores": torch.tensor([0.123456789])}
                    for _ in images]

    write_image(tmp_path / "img" / "a.jpg")
    r = predict(Fake(), [_frame("a.jpg")], tmp_path / "img", device="cpu", num_workers=0)
    assert r[0]["scores"][0] == pytest.approx(0.123456789, abs=1e-8)


def test_corrupt_name_is_light():
    code = ("import sys; from snz.filenames import corrupt_name; "
            "print('imagecorruptions' in sys.modules, corrupt_name('a.jpg', 'fog', 3))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["False", "a__fog_3.jpg"]


def test_guard_outside_a_git_repo_stays_closed_without_crashing(tmp_path):
    ensure_split_allowed("id-val", tmp_path)  # no .git: training splits still usable
    with pytest.raises(PermissionError):
        ensure_split_allowed("t-clean", tmp_path)
