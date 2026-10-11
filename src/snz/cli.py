"""Command-line entry point: `snz <command>`."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snz", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)  # noqa: F841 (tasks add here)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
