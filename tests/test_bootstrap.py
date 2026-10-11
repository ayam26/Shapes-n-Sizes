import numpy as np
import pytest

from snz.eval.bootstrap import bootstrap_weights, holm, one_sided_p, percentile_ci


def test_weights_are_reproducible_multinomial_counts():
    a = bootstrap_weights(50, 10, seed=1)
    assert a.shape == (10, 50) and (a.sum(axis=1) == 50).all()
    assert (a == bootstrap_weights(50, 10, seed=1)).all()
    assert not (a == bootstrap_weights(50, 10, seed=2)).all()


def _paired_diffs(effect, seed=0, n=200, b=2000):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    arm = base + effect + rng.normal(0, 0.5, n)
    w = bootstrap_weights(n, b, seed)
    return (w @ arm - w @ base) / n


def test_planted_difference_is_detected_and_null_is_not():
    assert one_sided_p(_paired_diffs(0.3), "greater") < 0.01
    assert one_sided_p(_paired_diffs(0.0), "greater") > 0.05
    assert one_sided_p(_paired_diffs(-0.3), "less") < 0.01


def test_one_sided_p_formula():
    assert one_sided_p(np.array([1.0, 2.0, -1.0, 0.0]), "greater") == pytest.approx(3 / 5)
    assert one_sided_p(np.array([1.0, 2.0, -1.0, 0.0]), "less") == pytest.approx(4 / 5)


def test_percentile_ci():
    lo, hi = percentile_ci(np.arange(1001.0))
    assert (lo, hi) == pytest.approx((25.0, 975.0))


def test_holm_hand_example():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    # sorted: d .005*4=.02, a .01*3=.03, c .03*2=.06, b .04*1=.04 -> monotone max -> .06
    assert adj == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"x": 0.6, "y": 0.9}) == {"x": 1.0, "y": 1.0}
