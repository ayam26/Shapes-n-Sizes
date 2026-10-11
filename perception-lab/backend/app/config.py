"""Paths and settings. Everything is resolved relative to the project root so the
same code runs on the GX10 (~/perception-lab) and in local tests."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PL_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("PL_DATA", ROOT / "data"))
CASES_DIR = DATA / "cases"
EXPORTS_DIR = DATA / "exports"
DB_PATH = DATA / "perception.sqlite"
EXPERIMENTS = ROOT / "experiments"
FRONTEND_DIST = ROOT / "frontend" / "dist"

PROVIDER = os.environ.get("PL_PROVIDER", "ollama")  # "ollama" | "fixture"
OFFLINE_GUARD = os.environ.get("PL_OFFLINE_GUARD", "0") == "1"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DIM = 2048

LABELS = ("physical_vehicle", "flat_depiction", "other", "uncertain")
TRUTH_LABELS = ("physical_vehicle", "flat_depiction", "other")
SPLITS = ("development", "heldout", "discovery", "interactive", "fidelity")
OBSERVABILITY = ("clear", "ambiguous", "unchecked")


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(name: str) -> dict:
    return json.loads((EXPERIMENTS / name).read_text())


def ensure_dirs() -> None:
    for d in (DATA, CASES_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
