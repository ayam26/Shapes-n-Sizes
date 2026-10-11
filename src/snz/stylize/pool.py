"""Style image pool and its fixed train/test split (spec §4.3)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

STYLE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_styles(root: Path) -> list[str]:
    root = Path(root)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in STYLE_SUFFIXES
    )


def split_style_pool(
    paths: Sequence[str], seed: int, test_fraction: float = 0.2
) -> tuple[list[str], list[str]]:
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    test, train = sorted(shuffled[:n_test]), sorted(shuffled[n_test:])
    if not train or not test:
        raise ValueError(f"style pool of {len(paths)} images is too small to split")
    return train, test
