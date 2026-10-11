"""Split manifests (image name + SHA-256) and the no-image-in-two-splits check (spec §4.2)."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path

from snz.hashing import sha256_file

Row = tuple[str, str]  # (image name, sha256)


class SplitOverlapError(ValueError):
    pass


def hash_rows(names: Sequence[str], image_dir: Path) -> list[Row]:
    rows = []
    for name in names:
        path = Path(image_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"image listed in labels but missing on disk: {path}")
        rows.append((name, sha256_file(path)))
    return rows


def write_manifest(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "sha256"])
        w.writerows(rows)


def read_manifest(path: Path) -> list[Row]:
    with open(path, newline="") as f:
        return [(r["name"], r["sha256"]) for r in csv.DictReader(f)]


def drop_hash_duplicates(
    rows: Sequence[Row], forbidden_hashes: set[str]
) -> tuple[list[Row], list[Row]]:
    kept = [r for r in rows if r[1] not in forbidden_hashes]
    dropped = [r for r in rows if r[1] in forbidden_hashes]
    return kept, dropped


def check_disjoint(manifests: dict[str, Sequence[Row]]) -> None:
    problems = []
    for (a, rows_a), (b, rows_b) in combinations(manifests.items(), 2):
        for kind, idx in (("name", 0), ("hash", 1)):
            shared = {r[idx] for r in rows_a} & {r[idx] for r in rows_b}
            if shared:
                problems.append(
                    f"{a} / {b}: {len(shared)} shared {kind}(s), e.g. {sorted(shared)[:3]}")
    if problems:
        raise SplitOverlapError("splits overlap:\n" + "\n".join(problems))
