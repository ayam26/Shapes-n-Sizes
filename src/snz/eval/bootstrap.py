"""Paired bootstrap over test images and Holm correction (spec §7.2).

Every run in a replicate uses the same image multiset (the same weight vector), so arm
differences are paired. p-values are one-sided in the predicted direction:
p = (1 + #replicates where the difference is not in that direction) / (B + 1).
"""

from __future__ import annotations

import numpy as np


def bootstrap_weights(n_images: int, b: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_images, np.full(n_images, 1 / n_images), size=b)


def one_sided_p(diffs: np.ndarray, direction: str) -> float:
    diffs = np.asarray(diffs, float)
    if direction == "greater":
        wrong = np.count_nonzero(diffs <= 0)
    elif direction == "less":
        wrong = np.count_nonzero(diffs >= 0)
    else:
        raise ValueError(f"direction must be 'greater' or 'less', not {direction!r}")
    return (1 + wrong) / (len(diffs) + 1)


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(np.asarray(x, float), [a, 100 - a])
    return float(lo), float(hi)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    names = sorted(pvals, key=pvals.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[name]))
        adjusted[name] = running
    return adjusted
