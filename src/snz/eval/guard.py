"""Gate G4: no test split is evaluated before the preregistration tag exists (spec §11)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from snz.data.bdd import TRAIN_SPLITS

PREREG_TAG = "prereg-v1"


def ensure_split_allowed(split: str, repo: Path = Path(".")) -> None:
    if split in TRAIN_SPLITS:
        return
    tags = subprocess.run(["git", "tag", "--list", PREREG_TAG], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.split()
    if PREREG_TAG not in tags:
        raise PermissionError(
            f"split {split!r} is held out: evaluation is blocked until the preregistration "
            f"is committed and tagged {PREREG_TAG!r} (gate G4)")
