"""Build all split manifests from BDD labels + images (gate G2)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from snz.data.bdd import TEST_SPLITS, TRAIN_SPLITS, LoadStats, build_splits, load_frames
from snz.data.manifest import check_disjoint, drop_hash_duplicates, hash_rows, write_manifest


def _stats(s: LoadStats) -> dict:
    return {"frames": s.frames, "boxes": s.boxes, "dropped_unknown": s.dropped_unknown,
            "dropped_degenerate": s.dropped_degenerate}


def build_and_write_splits(
    train_labels: Path,
    val_labels: Path,
    train_images: Path,
    val_images: Path,
    out_dir: Path,
    seed: int,
    idval_fraction: float,
) -> dict:
    out_dir = Path(out_dir)
    train_frames, train_stats = load_frames(train_labels)
    val_frames, val_stats = load_frames(val_labels)
    names = build_splits(train_frames, val_frames, seed, idval_fraction)

    rows = {s: hash_rows(names[s], val_images) for s in TEST_SPLITS}
    forbidden = {h for split_rows in rows.values() for _, h in split_rows}
    dropped_log = []
    # train first: its hashes are then also forbidden in id-val
    for s in TRAIN_SPLITS:
        kept, dropped = drop_hash_duplicates(hash_rows(names[s], train_images), forbidden)
        rows[s] = kept
        dropped_log += [(s, name, h) for name, h in dropped]
        forbidden |= {h for _, h in kept}
    rows = {s: rows[s] for s in (*TRAIN_SPLITS, *TEST_SPLITS)}
    check_disjoint(rows)

    for split, split_rows in rows.items():
        write_manifest(out_dir / f"{split}.csv", split_rows)
    by_hash = {h: (s, n) for s in (*TEST_SPLITS, "train") for n, h in rows[s]}
    with open(out_dir / "dropped_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "name", "sha256", "duplicate_of_split", "duplicate_of_name"])
        w.writerows((s, n, h, *by_hash[h]) for s, n, h in dropped_log)

    summary = {
        "seed": seed,
        "idval_fraction": idval_fraction,
        "counts": {s: len(r) for s, r in rows.items()},
        "dropped_duplicates": len(dropped_log),
        "labels": {"train": _stats(train_stats), "val": _stats(val_stats)},
    }
    (out_dir / "split_counts.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
