import hashlib
import json

import pytest

from helpers import frame_json, write_image, write_labels
from snz.cli import main
from snz.data.build import build_and_write_splits
from snz.data.manifest import (
    SplitOverlapError,
    check_disjoint,
    drop_hash_duplicates,
    hash_rows,
    read_manifest,
    write_manifest,
)
from snz.hashing import ChecksumError, sha256_file, verify_sha256


def test_sha256_and_verify(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == hashlib.sha256(b"hello").hexdigest()
    verify_sha256(p, hashlib.sha256(b"hello").hexdigest())
    with pytest.raises(ChecksumError, match="x.bin"):
        verify_sha256(p, "0" * 64)


def test_manifest_roundtrip_and_missing_image(tmp_path):
    write_image(tmp_path / "img" / "a.jpg")
    rows = hash_rows(["a.jpg"], tmp_path / "img")
    write_manifest(tmp_path / "m.csv", rows)
    assert read_manifest(tmp_path / "m.csv") == rows
    with pytest.raises(FileNotFoundError, match="b.jpg"):
        hash_rows(["b.jpg"], tmp_path / "img")


def test_check_disjoint_by_name_and_by_hash():
    check_disjoint({"train": [("a", "h1")], "t-clean": [("b", "h2")]})
    with pytest.raises(SplitOverlapError, match="name"):
        check_disjoint({"train": [("a", "h1")], "t-clean": [("a", "h2")]})
    with pytest.raises(SplitOverlapError, match="hash"):
        check_disjoint({"train": [("a", "h1")], "t-clean": [("b", "h1")]})


def test_drop_hash_duplicates():
    kept, dropped = drop_hash_duplicates([("a", "h1"), ("b", "h2")], {"h2"})
    assert kept == [("a", "h1")] and dropped == [("b", "h2")]


def test_idval_copies_of_train_images_are_dropped(tmp_path):
    train = [frame_json(f"tr{i}.jpg") for i in range(10)]
    for i in range(10):  # every image identical: id-val copies must all be dropped
        write_image(tmp_path / "train" / f"tr{i}.jpg", color=(50, 50, 50))
    write_image(tmp_path / "val" / "v.jpg", color=(0, 0, 200))
    out = tmp_path / "m"
    summary = build_and_write_splits(
        write_labels(tmp_path / "t.json", train), write_labels(tmp_path / "v.json",
                                                               [frame_json("v.jpg")]),
        tmp_path / "train", tmp_path / "val", out, seed=0, idval_fraction=0.2)
    assert summary["counts"]["id-val"] == 0 and summary["dropped_duplicates"] == 2


def _fixture(tmp_path):
    train = [frame_json(f"tr{i}.jpg", labels=[("car", 0, 0, 10, 10)]) for i in range(10)]
    val = [
        frame_json("v_clean.jpg"),
        frame_json("v_night.jpg", timeofday="night"),
        frame_json("v_rain.jpg", weather="rainy"),
    ]
    for i in range(10):  # colors far apart so JPEG bytes differ
        write_image(tmp_path / "train" / f"tr{i}.jpg", color=(20 * i, 0, 0))
    write_image(tmp_path / "val" / "v_clean.jpg", color=(0, 0, 200))
    write_image(tmp_path / "val" / "v_night.jpg", color=(0, 200, 0))
    # same pixels as tr3.jpg under another name: a cross-split duplicate
    write_image(tmp_path / "val" / "v_rain.jpg", color=(60, 0, 0))
    return (
        write_labels(tmp_path / "det_train.json", train),
        write_labels(tmp_path / "det_val.json", val),
    )


def test_build_and_write_splits_drops_cross_split_duplicates(tmp_path):
    tl, vl = _fixture(tmp_path)
    out = tmp_path / "manifests"
    summary = build_and_write_splits(
        tl, vl, tmp_path / "train", tmp_path / "val", out, seed=0, idval_fraction=0.2
    )
    train = [n for n, _ in read_manifest(out / "train.csv")]
    idval = [n for n, _ in read_manifest(out / "id-val.csv")]
    assert "tr3.jpg" not in train + idval
    assert [n for n, _ in read_manifest(out / "t-rain.csv")] == ["v_rain.jpg"]
    dropped = (out / "dropped_duplicates.csv").read_text()
    assert "tr3.jpg" in dropped and "v_rain.jpg" in dropped
    assert summary["counts"]["t-clean"] == 1
    assert json.loads((out / "split_counts.json").read_text()) == summary


def test_cli_build_splits(tmp_path):
    tl, vl = _fixture(tmp_path)
    cfg = tmp_path / "splits.toml"
    cfg.write_text("seed = 0\nidval_fraction = 0.2\n")
    code = main([
        "build-splits", "--train-labels", str(tl), "--val-labels", str(vl),
        "--train-images", str(tmp_path / "train"), "--val-images", str(tmp_path / "val"),
        "--out", str(tmp_path / "m"), "--config", str(cfg),
    ])
    assert code == 0
    assert (tmp_path / "m" / "t-night.csv").exists()


def test_val_all_manifest_lists_every_val_frame(tmp_path):
    tl, vl = _fixture(tmp_path)
    out = tmp_path / "m"
    summary = build_and_write_splits(tl, vl, tmp_path / "train", tmp_path / "val", out,
                                     seed=0, idval_fraction=0.2)
    assert sorted(n for n, _ in read_manifest(out / "val-all.csv")) == [
        "v_clean.jpg", "v_night.jpg", "v_rain.jpg"]
    assert summary["counts"]["val-all"] == 3 and (out / "t-dusk.csv").exists()


def test_descriptive_split_copies_of_test_images_are_dropped(tmp_path):
    val = [frame_json("v_rain.jpg", weather="rainy"),
           frame_json("v_over.jpg", weather="overcast"),
           frame_json("v_over2.jpg", weather="overcast")]
    write_image(tmp_path / "val" / "v_rain.jpg", color=(10, 200, 10))
    write_image(tmp_path / "val" / "v_over.jpg", color=(10, 200, 10))  # same pixels as v_rain
    write_image(tmp_path / "val" / "v_over2.jpg", color=(200, 10, 10))
    write_image(tmp_path / "train" / "tr.jpg", color=(90, 90, 90))
    out = tmp_path / "m"
    build_and_write_splits(write_labels(tmp_path / "t.json", [frame_json("tr.jpg")]),
                           write_labels(tmp_path / "v.json", val), tmp_path / "train",
                           tmp_path / "val", out, seed=0, idval_fraction=0.0)
    assert [n for n, _ in read_manifest(out / "t-rain.csv")] == ["v_rain.jpg"]
    assert [n for n, _ in read_manifest(out / "t-overcast.csv")] == ["v_over2.jpg"]
    assert "v_over.jpg" in (out / "dropped_duplicates.csv").read_text()
