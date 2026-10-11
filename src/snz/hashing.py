"""File checksums and the known-good weight hashes (README §5.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

KNOWN_SHA256 = {
    "resnet50_train_60_epochs-c8e5653e.pth.tar":
        "c8e5653e47bbf3f6efad5da531f3acea9963acb93362b6530cf39b2ba4750da3",
    "decoder.pth": "379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472",
    "vgg_normalised.pth": "804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a",
}


class ChecksumError(ValueError):
    pass


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: Path | str, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumError(f"{path}: expected sha256 {expected}, got {actual}")
