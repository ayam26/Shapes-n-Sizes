"""Precompute stylized training copies and the T-texture test set (spec §4.3).

Style choice is a pure function of (seed, image name, copy), so an interrupted or
partial build resumes to exactly the same result.
"""

from __future__ import annotations

import csv
import hashlib
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from snz.data.manifest import hash_rows, read_manifest, write_manifest
from snz.stylize.pool import list_styles, split_style_pool

StylizeFn = Callable[[Image.Image, Image.Image], Image.Image]


@dataclass(frozen=True)
class StylizeConfig:
    seed: int
    test_fraction: float
    train_copies: int
    alpha: float
    style_size: int


def load_stylize_config(path: Path) -> StylizeConfig:
    return StylizeConfig(**tomllib.loads(Path(path).read_text()))


def style_for(name: str, copy: int, styles: Sequence[str], seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{name}:{copy}".encode()).digest()
    return styles[int.from_bytes(digest[:8], "big") % len(styles)]


def stylized_name(name: str, copy: int) -> str:
    return f"{Path(name).stem}_s{copy}.jpg"


def build_stylized(
    names: Sequence[str],
    image_dir: Path,
    style_root: Path,
    styles: Sequence[str],
    out_dir: Path,
    copies: Sequence[int],
    seed: int,
    stylizer: StylizeFn,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        for copy in copies:
            style = style_for(name, copy, styles, seed)
            out = out_dir / stylized_name(name, copy)
            if not out.exists():
                content = Image.open(Path(image_dir) / name).convert("RGB")
                result = stylizer(content, Image.open(Path(style_root) / style).convert("RGB"))
                if result.size != content.size:
                    raise ValueError(
                        f"stylizer changed size of {name}: {content.size} -> {result.size}")
                tmp = out.with_name(out.name + ".tmp")
                result.save(tmp, format="JPEG", quality=95)
                tmp.replace(out)
            rows.append({"name": name, "copy": copy, "style": style, "file": out.name})
    return rows


def run_stylize(
    set_name: str,
    manifests_dir: Path,
    image_dir: Path,
    style_root: Path,
    out_dir: Path,
    cfg: StylizeConfig,
    stylizer: StylizeFn,
) -> list[dict]:
    manifests_dir = Path(manifests_dir)
    train_styles, test_styles = split_style_pool(
        list_styles(style_root), cfg.seed, cfg.test_fraction)
    write_manifest(manifests_dir / "styles_train.csv", hash_rows(train_styles, style_root))
    write_manifest(manifests_dir / "styles_test.csv", hash_rows(test_styles, style_root))
    if set_name == "train":
        source, copies, styles = "train.csv", range(1, cfg.train_copies + 1), train_styles
    elif set_name == "t-texture":
        source, copies, styles = "t-clean.csv", (1,), test_styles
    else:
        raise ValueError(f"unknown stylized set {set_name!r}; expected 'train' or 't-texture'")
    names = [n for n, _ in read_manifest(manifests_dir / source)]
    rows = build_stylized(names, image_dir, style_root, styles, out_dir, tuple(copies), cfg.seed,
                          stylizer)
    with open(manifests_dir / f"stylized_{set_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "copy", "style", "file"])
        w.writeheader()
        w.writerows(rows)
    return rows
