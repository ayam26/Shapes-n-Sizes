import numpy as np
import pytest

from snz.gnn.graph import box_contour, mask_to_contour, neighbor_index, node_features, size_features


def _square(h=100, w=120, y0=20, x0=30, s=40):
    m = np.zeros((h, w), bool)
    m[y0:y0 + s, x0:x0 + s] = True
    return m


def test_contour_shape_order_start():
    c = mask_to_contour(_square())
    assert c.shape == (64, 2)
    assert c[0, 1] == pytest.approx(c[:, 1].min(), abs=1.0)  # starts at the top-most point
    x, y = c[:, 0], c[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0  # clockwise on screen


def test_node_features_translation_scale_invariant():
    a = node_features(mask_to_contour(_square(s=40)))
    b = node_features(mask_to_contour(_square(h=400, w=400, y0=150, x0=90, s=160)))
    np.testing.assert_allclose(a, b, atol=0.05)


def test_circle_and_square_differ():
    yy, xx = np.mgrid[:100, :100]
    circle = (yy - 50) ** 2 + (xx - 50) ** 2 < 30 ** 2
    assert (node_features(mask_to_contour(circle))[:, 2].std()
            < node_features(mask_to_contour(_square()))[:, 2].std())


def test_largest_blob_and_holes():
    m = _square(s=40)
    m[80:84, 100:104] = True
    m[30:40, 40:50] = False
    c = mask_to_contour(m)
    assert c[:, 0].min() >= 29 and c[:, 0].max() <= 70


def test_degenerate_masks():
    assert mask_to_contour(np.zeros((10, 10), bool)) is None
    one = np.zeros((10, 10), bool)
    one[5, 5] = True
    assert mask_to_contour(one) is None
    assert np.isfinite(node_features(box_contour([5, 5, 6, 5.5]))).all()
    assert np.isfinite(size_features([5, 5, 6, 5.5], box_contour([5, 5, 6, 5.5]),
                                     (1280, 720))).all()


def test_size_features_values():
    box = [100, 200, 300, 400]
    s = size_features(box, box_contour(box), (1000, 500))
    np.testing.assert_allclose(s, [0.2, 0.4, 0.8, 0.2, 0.0, 1.0], atol=1e-5)


def test_neighbor_index_ring():
    idx = neighbor_index(64)
    assert idx.shape == (64, 6) and list(idx[0]) == [1, 63, 2, 62, 4, 60]
