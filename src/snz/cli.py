"""Command-line entry point: `snz <command>`."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snz", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("build-splits", help="write split manifests from BDD labels (G2)")
    p.add_argument("--train-labels", type=Path, required=True)
    p.add_argument("--val-labels", type=Path, required=True)
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--val-images", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("manifests"))
    p.add_argument("--config", type=Path, default=Path("configs/splits.toml"))
    p.set_defaults(func=_cmd_build_splits)

    return parser


def _cmd_build_splits(args) -> int:
    from snz.data.build import build_and_write_splits

    cfg = tomllib.loads(args.config.read_text())
    summary = build_and_write_splits(
        args.train_labels, args.val_labels, args.train_images, args.val_images,
        args.out, seed=cfg["seed"], idval_fraction=cfg["idval_fraction"],
    )
    print(json.dumps(summary["counts"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
