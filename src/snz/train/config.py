"""Training recipe and arm definitions (configs/recipe.toml, configs/arms.toml)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    frozen: bool
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    iterations: int
    warmup_iters: int
    lr_steps: list[float]
    min_sizes: list[int]
    max_size: int
    hflip_p: float
    p_original: float
    trainable_layers: int
    bf16: bool
    num_workers: int
    log_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class Arm:
    name: str
    backbone: str
    stylized: bool


def load_recipe(path: Path) -> Recipe:
    raw = tomllib.loads(Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(Recipe)}
    unknown, missing = set(raw) - fields, fields - set(raw)
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return Recipe(**raw)


def recipe_sha256(recipe: Recipe) -> str:
    blob = json.dumps(dataclasses.asdict(recipe), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def load_arms(path: Path) -> dict[str, Arm]:
    raw = tomllib.loads(Path(path).read_text())
    return {name: Arm(name, v["backbone"], bool(v["stylized"])) for name, v in raw.items()}
