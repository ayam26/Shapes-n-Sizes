"""Crash-safe file replacement: data reaches disk before the new name appears."""

from __future__ import annotations

import os
from pathlib import Path


def atomic_replace(tmp: Path, dest: Path) -> None:
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    fd = os.open(Path(dest).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
