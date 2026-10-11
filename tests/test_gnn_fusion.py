import json

import numpy as np

from gnn_helpers import tiny_idval
from snz.cli import main
from snz.gnn.fusion import auroc, fit_logistic, fusion_features, predict_logistic


def test_fit_logistic_recovers_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 4))
    w = np.array([0.5, 2, -1, 0, 1])
    y = rng.random(5000) < 1 / (1 + np.exp(-(w[0] + X @ w[1:])))
    np.testing.assert_allclose(fit_logistic(X, y), w, atol=0.15)


def test_fit_logistic_separable_is_finite():
    X = np.r_[-np.ones((50, 4)), np.ones((50, 4))]
    y = np.r_[np.zeros(50), np.ones(50)].astype(bool)
    w = fit_logistic(X, y)
    assert np.isfinite(w).all()
    p = predict_logistic(w, X)
    assert ((p > 0) & (p < 1)).all()


def test_fusion_features_columns():
    probs = np.array([[0.7, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0.2]])
    X = fusion_features(np.array([0.5]), probs, np.array([0]))
    np.testing.assert_allclose(X[0, :3], [0.0, np.log(0.7), np.log(0.2)], atol=1e-6)
    assert X[0, 3] == np.log(0.1)


def test_auroc():
    assert auroc(np.array([0, 0, 1, 1], bool), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(np.array([0, 1], bool), np.array([0.5, 0.5])) == 0.5


def test_gnn_train_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("SNZ_GNN_EPOCHS", "1")
    imgs, labels, manifest, outlines = tiny_idval(tmp_path, n_images=8)
    out = tmp_path / "models"
    assert main(["gnn-train", "--outlines", str(outlines), "--labels", str(labels),
                 "--manifest", str(manifest), "--images", str(imgs), "--out", str(out),
                 "--arms", "GNN,GNN-SHAPE,RGB", "--device", "cpu", "--no-pretrained",
                 "--fold-dir", str(tmp_path / "manifests")]) == 0
    assert (out / "GNN.pt").exists() and (out / "GNN.fusion.json").exists()
    assert (tmp_path / "manifests" / "gnn_fold_a.csv").exists()
    summary = json.loads((out / "gnn-train.json").read_text())
    assert set(summary["fold_b_auroc"]) >= {"BASE", "GNN", "RGB"}
    assert set(summary["sha256"]) == {"GNN.pt", "GNN-SHAPE.pt", "RGB.pt"}
