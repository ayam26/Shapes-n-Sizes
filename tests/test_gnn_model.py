import numpy as np
import torch

from gnn_helpers import squares_and_circles, synthetic_outlines
from snz.gnn.model import GraphDataset, ShapeGNN, predict_probs, rgb_model, train_network


def test_shape_gnn_output_and_cyclic_invariance():
    torch.manual_seed(0)
    m = ShapeGNN().eval()
    x, s = torch.randn(3, 64, 5), torch.randn(3, 6)
    assert m(x, s).shape == (3, 10)
    torch.testing.assert_close(m(x, s), m(torch.roll(x, 7, dims=1), s), atol=1e-5, rtol=1e-5)


def test_shape_variant_ignores_size():
    m = ShapeGNN(use_size=False).eval()
    x = torch.randn(2, 64, 5)
    torch.testing.assert_close(m(x, torch.zeros(2, 6)), m(x, torch.randn(2, 6)))


def test_gnn_features_are_color_blind():
    o1, o2 = synthetic_outlines(color=(255, 0, 0)), synthetic_outlines(color=(0, 0, 255))
    t = np.zeros(len(o1.score), np.int64)
    a, b = GraphDataset(o1, t, "gnn", False)[0], GraphDataset(o2, t, "gnn", False)[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert GraphDataset(o1, t, "color", False)[0][0].shape == (64, 8)
    assert torch.count_nonzero(GraphDataset(o1, t, "shape", False)[0][1]) == 0


def test_learns_square_vs_circle():
    o, t = squares_and_circles(n=200)
    m = train_network(ShapeGNN(), GraphDataset(o, t, "shape", True), epochs=20, lr=1e-3,
                      batch_size=64, weight_decay=1e-4, device="cpu")
    acc = (predict_probs(m, GraphDataset(o, t, "shape", False), device="cpu").argmax(1)
           == t).mean()
    assert acc > 0.95


def test_rgb_model_shape():
    assert rgb_model(pretrained=False)(torch.randn(2, 3, 112, 112)).shape == (2, 10)
