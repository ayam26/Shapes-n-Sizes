import subprocess

import pytest

from gnn_helpers import outline_set_for, tiny_idval
from snz.cli import main
from snz.data.bdd import load_frames
from snz.eval.guard import ensure_split_allowed
from snz.eval.matching import match_split
from snz.gnn.outlines import load_outlines
from snz.gnn.rescore import rescore


def test_base_arm_keeps_selected_scores(tmp_path):
    o = outline_set_for("a.jpg", boxes=[[0, 0, 5, 5], [1, 1, 6, 6]], scores=[0.9, 0.2],
                        extra_names=["empty.jpg"])
    recs = rescore(o, "BASE", tmp_path, image_dir=tmp_path, device="cpu")
    assert recs[0]["scores"] == pytest.approx([0.9, 0.2])
    assert recs[1] == {"name": "empty.jpg", "boxes": [], "labels": [], "scores": []}


@pytest.mark.parametrize("arm", ["GNN", "RGB"])
def test_rescored_records_are_matchable(tmp_path, monkeypatch, arm):
    monkeypatch.setenv("SNZ_GNN_EPOCHS", "1")
    imgs, labels, manifest, outlines = tiny_idval(tmp_path, n_images=8)
    main(["gnn-train", "--outlines", str(outlines), "--labels", str(labels), "--manifest",
          str(manifest), "--images", str(imgs), "--out", str(tmp_path / "m"), "--arms", arm,
          "--device", "cpu", "--no-pretrained", "--fold-dir", str(tmp_path / "manifests")])
    o, frames = load_outlines(outlines), load_frames(labels)[0]
    recs = rescore(o, arm, tmp_path / "m", image_dir=imgs, device="cpu", chunk=3)
    m = match_split(frames, recs)
    assert len(m.det_score) == len(o.score)
    assert all(0 < s < 1 for r in recs for s in r["scores"])


def test_guard_requires_prereg_v2(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "prereg-v1"], cwd=tmp_path, check=True)
    ensure_split_allowed("val-all", repo=tmp_path)  # prereg-v1 alone opens CNN evaluation
    with pytest.raises(PermissionError, match="prereg-v2"):
        ensure_split_allowed("val-all", repo=tmp_path, tag="prereg-v2")
    ensure_split_allowed("id-val", repo=tmp_path, tag="prereg-v2")
