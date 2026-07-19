"""The ``fxtranslate`` command-line interface — a thin shell over the compiled
native engine, held byte-identical to the Rust ``fxtranslate-cli`` oracle (same
subcommands, flags, help/usage text, error strings, exit codes, and list/models
formatters). The shell owns only the interface; discovery, routing, the verified
cache, and the engine come from the compiled ``fxtranslate`` extension.

Structure mirrors the npm ``lib/`` split:

- ``usage.py`` — USAGE / MODELS_USAGE / LIST_USAGE, byte-for-byte from cli.rs
- ``parse.py`` — the arg grammar + help routing (pure), byte-for-byte from cli.rs
- ``format.py`` — list/models view formatters, byte-for-byte from cli.rs
- ``lang.py`` — the language-tag → display-name table
- ``run.py`` — parse → dispatch → exit code, over the compiled ``Translator``/
  ``Cache``/``discovery`` surface
"""

import sys

from .run import process_io, run


def main() -> int:
    """Entry point for the ``fxtranslate`` console script and ``python -m fxtranslate``.
    A thin shim mirroring ``main.rs``: lift argv, build the real terminal ``Io``, hand
    off to :func:`run`, and exit with its code."""
    args = sys.argv[1:]
    io = process_io()
    code = run(args, io)
    sys.exit(code)
