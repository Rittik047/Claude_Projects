"""`ecgtrust` command-line entry point. Subcommands register themselves here."""

from __future__ import annotations

import argparse
import importlib

# Modules that expose `register(subparsers)`; added as tasks land.
COMMAND_MODULES: list[str] = [
    "ecgtrust.data.download",
    "ecgtrust.audit",
    "ecgtrust.train_cli",
    "ecgtrust.freeze",
    "ecgtrust.evaluate.report",
    "ecgtrust.results",
    "ecgtrust.release",
    "ecgtrust.figures",
    "ecgtrust.readme_tables",
    "ecgtrust.guide.build",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecgtrust", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for mod in COMMAND_MODULES:
        importlib.import_module(mod).register(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
