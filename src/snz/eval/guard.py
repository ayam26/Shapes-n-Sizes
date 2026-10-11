"""Gate G4: no test split is evaluated before the preregistration tag exists (spec §11)."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from snz.data.bdd import TEST_SPLITS, TRAIN_SPLITS

PREREG_TAG = "prereg-v1"


def _names(manifest: Path) -> set[str]:
    with open(manifest, newline="") as f:
        return {r["name"] for r in csv.DictReader(f)}


def ensure_split_allowed(
    split: str, repo: Path = Path("."), manifest: Path | None = None
) -> None:
    """Refuse held-out data before the prereg tag.

    With `manifest`, the file must be named `<split>.csv`, and before the tag none of its
    images may appear in a test-split manifest next to it, so a mislabeled `--split`
    cannot open a test split.
    """
    if manifest is not None and Path(manifest).stem != split:
        raise ValueError(f"manifest {manifest} does not match --split {split!r}")
    tags = subprocess.run(["git", "tag", "--list", PREREG_TAG], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.split()
    if PREREG_TAG in tags:
        return
    if split not in TRAIN_SPLITS:
        raise PermissionError(
            f"split {split!r} is held out: evaluation is blocked until the preregistration "
            f"is committed and tagged {PREREG_TAG!r} (gate G4)")
    if manifest is not None:
        names = _names(manifest)
        for test in TEST_SPLITS:
            other = Path(manifest).parent / f"{test}.csv"
            if other.exists() and names & _names(other):
                raise PermissionError(
                    f"{manifest} contains images of test split {test!r}; blocked until "
                    f"{PREREG_TAG!r} (gate G4)")
