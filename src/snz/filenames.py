"""Derived image file names, importable without heavy dependencies."""

from __future__ import annotations

from pathlib import Path


def corrupt_name(name: str, corruption: str, severity: int) -> str:
    return f"{Path(name).stem}__{corruption}_{severity}.jpg"
