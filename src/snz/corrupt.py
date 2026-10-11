"""T-corrupt (exploratory, spec §4.2): standard image corruptions of a T-clean subset.

Uses `imagecorruptions` (Michaelis et al., 2019). Two of its 15 corruptions call APIs that
newer libraries removed (`np.float_` in fog, `multichannel=` in skimage's gaussian for glass
blur), and newer skimage's `random_noise` ignores numpy's global seed (impulse noise); the
shims below restore the old behavior. Each corruption is seeded by
(image name, corruption, severity), so files are reproducible and a rerun skips finished ones.
"""

from __future__ import annotations

import hashlib
import random
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from snz.fsutil import atomic_replace

if not hasattr(np, "float_"):
    np.float_ = np.float64  # fog's plasma fractal

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # pkg_resources deprecation inside imagecorruptions
    import imagecorruptions.corruptions as _ic
    from imagecorruptions import corrupt, get_corruption_names

_gaussian = _ic.gaussian


def _gaussian_compat(image, sigma=1, multichannel=None, **kw):
    if multichannel is not None:
        kw["channel_axis"] = -1 if multichannel else None
    return _gaussian(image, sigma=sigma, **kw)


_ic.gaussian = _gaussian_compat
_random_noise = _ic.sk.util.random_noise


def _random_noise_seeded(image, mode="gaussian", rng=None, **kw):
    if rng is None:
        rng = int(np.random.randint(2**31 - 1))  # follows np.random.seed(), as old skimage did
    return _random_noise(image, mode=mode, rng=rng, **kw)


_ic.sk.util.random_noise = _random_noise_seeded

CORRUPTIONS = tuple(get_corruption_names("all"))[:15]


def corrupt_name(name: str, corruption: str, severity: int) -> str:
    return f"{Path(name).stem}__{corruption}_{severity}.jpg"


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "big")


def corrupt_image(img: np.ndarray, corruption: str, severity: int, key: str) -> np.ndarray:
    np.random.seed(_seed(key, corruption, severity))
    out = corrupt(img, corruption_name=corruption, severity=severity)
    return np.asarray(out).astype(np.uint8)


def choose_subset(names: Sequence[str], k: int, seed: int) -> list[str]:
    if len(names) < k:
        raise ValueError(f"cannot sample {k} images from {len(names)}")
    return sorted(random.Random(seed).sample(sorted(names), k))


def _one(job: tuple[Path, Path, str, str, int]) -> None:
    src, dst, name, corruption, severity = job
    img = np.asarray(Image.open(src).convert("RGB"))
    out = corrupt_image(img, corruption, severity, key=name)
    tmp = dst.with_name(dst.name + ".tmp")
    Image.fromarray(out).save(tmp, format="JPEG", quality=95)
    atomic_replace(tmp, dst)


def build_t_corrupt(names: Sequence[str], image_dir: Path, out_dir: Path,
                    corruptions: Sequence[str] = CORRUPTIONS,
                    severities: Sequence[int] = (1, 3, 5), workers: int = 8) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for name in names:
        for c in corruptions:
            for s in severities:
                dst = out_dir / corrupt_name(name, c, s)
                rows.append({"name": name, "corruption": c, "severity": s, "file": dst.name})
                if not dst.exists():
                    jobs.append((Path(image_dir) / name, dst, name, c, s))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as pool:
            list(pool.map(_one, jobs, chunksize=4))
    else:
        for job in jobs:
            _one(job)
    return rows
