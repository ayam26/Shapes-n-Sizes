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

    p = commands.add_parser("stylize", help="precompute stylized copies (G2)")
    p.add_argument("--set", choices=["train", "t-texture"], required=True)
    p.add_argument("--manifests", type=Path, default=Path("manifests"))
    p.add_argument("--images", type=Path, required=True, help="BDD train (for train) or val dir")
    p.add_argument("--styles-root", type=Path, required=True, help="e.g. dtd/images")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vgg", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/stylize.toml"))
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_stylize)

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


def _cmd_stylize(args) -> int:
    from snz.stylize.adain import Stylizer
    from snz.stylize.build import load_stylize_config, run_stylize

    cfg = load_stylize_config(args.config)
    stylizer = Stylizer.from_weights(args.vgg, args.decoder, args.device, cfg.style_size,
                                     cfg.alpha)
    rows = run_stylize(args.set, args.manifests, args.images, args.styles_root, args.out, cfg,
                       stylizer)
    print(f"{len(rows)} stylized images in {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
