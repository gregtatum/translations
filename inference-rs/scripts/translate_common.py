#!/usr/bin/env python3
"""
Shared helpers for the translate scripts.

`translate_reference.py` (the C++ reference) and `translate.py` (the Rust engine)
take the same `<source> <target> [--text ...]` interface and resolve the same
downloaded model directory, so that plumbing lives here.
"""

import argparse
import sys
from pathlib import Path

DEFAULT_MODELS_DIR = "data/models"
DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "es"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the source/target/--text/--models-dir arguments both scripts share."""
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help=f"Source language code, e.g. en (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=DEFAULT_TARGET,
        help=f"Target language code, e.g. es (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--text",
        help="Text to translate. If omitted, text is read from stdin.",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Root directory the model was downloaded into (default: {DEFAULT_MODELS_DIR})",
    )


def config_path(models_dir: str, src: str, trg: str) -> Path:
    """The decode-config path for a downloaded `src`→`trg` model (may not exist)."""
    langs = f"{src.lower()}{trg.lower()}"
    return Path(models_dir) / langs / f"config.{langs}.yml"


def resolve_config(models_dir: str, source: str, target: str) -> tuple[str, str, str, Path]:
    """Resolve the decode config for a language pair, erroring if it is missing.

    Returns `(src, trg, langs, config_path)`.
    """
    src, trg = source.lower(), target.lower()
    langs = f"{src}{trg}"
    config = config_path(models_dir, src, trg)
    if not config.exists():
        raise SystemExit(
            f"[error] decode config not found at {config}\n"
            f"  Download the model first with: task rs:download-model -- {src} {trg}"
        )
    return src, trg, langs, config


def resolve_route(
    models_dir: str, source: str, target: str, hub: str = "en"
) -> tuple[str, list[tuple[str, str, Path]]]:
    """Resolve a language pair to a translation route against the *downloaded*
    models, mirroring `fxtranslate::route::resolve_route`: a direct model wins,
    else pivot `src`→`hub`→`trg`. This is the perf/parity harnesses' local
    equivalent — it resolves by which config files exist on disk, not Remote
    Settings, so both engines run the same route the shipped resolver would pick.

    Returns `(kind, legs)` where `kind` is `"direct"` or `"pivot"` and each leg is
    `(src, trg, config_path)`. Errors (with download hints) when neither a direct
    model nor both pivot legs are present on disk.
    """
    src, trg = source.lower(), target.lower()
    hub = hub.lower()
    direct = config_path(models_dir, src, trg)
    if direct.exists():
        return ("direct", [(src, trg, direct)])

    # A pair already touching the hub (or src == trg) has no sensible pivot — the
    # legs would be degenerate (hub→hub). Report it as a missing direct model.
    if hub in (src, trg):
        raise SystemExit(
            f"[error] decode config not found at {direct}\n"
            f"  Download the model first with: task rs:download-model -- {src} {trg}"
        )

    leg1, leg2 = config_path(models_dir, src, hub), config_path(models_dir, hub, trg)
    missing = [(a, b, p) for (a, b, p) in ((src, hub, leg1), (hub, trg, leg2)) if not p.exists()]
    if missing:
        hints = "\n".join(f"  task rs:download-model -- {a} {b}" for a, b, _ in missing)
        raise SystemExit(
            f"[error] no direct {src}→{trg} model, and the pivot through {hub} is "
            f"missing {len(missing)} leg(s):\n{hints}"
        )
    return ("pivot", [(src, hub, leg1), (hub, trg, leg2)])


def read_input_text(args: argparse.Namespace) -> str:
    """The text to translate: `--text` if given, otherwise all of stdin."""
    return args.text if args.text is not None else sys.stdin.read()


def _yaml_list(value: str) -> list[str]:
    """Parse a small inline YAML list like `[a, b]` into `['a', 'b']`."""
    return [item.strip() for item in value.strip().strip("[]").split(",") if item.strip()]


def parse_model_config(config: Path) -> dict:
    """Parse the model/vocab/shortlist paths out of a bergamot decode config.

    Only the handful of keys the Rust engine needs are read. Relative entries
    are resolved against the config's directory (the configs use
    `relative-paths: true`). Returns `{"model": Path, "vocabs": [Path, ...],
    "shortlist": Path | None}`.
    """
    base = config.parent
    fields: dict[str, list[str]] = {}
    for line in config.read_text().splitlines():
        line = line.strip()
        for key in ("models", "vocabs", "shortlist"):
            if line.startswith(key + ":"):
                fields[key] = _yaml_list(line.split(":", 1)[1])

    def resolve(name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else base / p

    models = fields.get("models", [])
    vocabs = fields.get("vocabs", [])
    shortlist_entries = [e for e in fields.get("shortlist", []) if e.lower() != "false"]

    if not models:
        raise SystemExit(f"[error] no `models:` entry in {config}")
    if not vocabs:
        raise SystemExit(f"[error] no `vocabs:` entry in {config}")

    return {
        "model": resolve(models[0]),
        "vocabs": [resolve(v) for v in vocabs],
        "shortlist": resolve(shortlist_entries[0]) if shortlist_entries else None,
    }
