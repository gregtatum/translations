"""Parse argv, dispatch to the matching command, and execute it against ``io`` — the
Python mirror of the Rust ``run``/``dispatch`` (crates/fxtranslate-cli/src/cli.rs).

The read-only paths (``list``, ``models list/info``) and the cache-writing / engine
paths (``translate``, ``models add``, ``models rm``) all run through the compiled
core: discovery/catalog/routing, the verified ``Cache``, and the engine come from
the native ``fxtranslate`` extension. The shell owns only the interface — arg
grammar, help/usage text, error strings, the list/models formatters, and stream
handling — the bits the npm shell also re-ports.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

from .. import discovery
from .._engine import Cache, Translator
from . import parse as p
from .format import (
    PREFERRED_HUB,
    human_bytes,
    pad_end,
    pad_start,
    palette,
    pretty_pair,
    scalar_len,
    write_languages,
    write_pairs,
)
from .usage import LIST_USAGE, MODELS_USAGE, USAGE


@dataclass
class Io:
    """The host I/O + terminal contract the CLI runs against — the mirror of the Rust
    ``Io`` struct. Injected rather than probed from the process so the stream sinks
    and TTY / ``NO_COLOR`` facts are explicit."""

    stdin: TextIO
    stdout: Callable[[str], None]
    stderr: Callable[[str], None]
    stdin_is_tty: bool
    stdout_is_tty: bool
    stderr_is_tty: bool
    no_color: bool


def process_io() -> Io:
    """An :class:`Io` bound to the real process streams and terminal facts — the
    mirror of ``main.rs`` wiring the real terminal into ``run``. ``NO_COLOR`` is
    honored by presence (any value)."""
    return Io(
        stdin=sys.stdin,
        stdout=lambda s: (sys.stdout.write(s), sys.stdout.flush()),
        stderr=lambda s: (sys.stderr.write(s), sys.stderr.flush()),
        stdin_is_tty=sys.stdin.isatty(),
        stdout_is_tty=sys.stdout.isatty(),
        stderr_is_tty=sys.stderr.isatty(),
        no_color="NO_COLOR" in os.environ,
    )


def run(args, io: Io) -> int:
    """Parse and execute ``args`` against ``io``, returning the process exit code
    (0 or 1). Errors are reported to stderr prefixed ``fxtranslate: `` — the mirror of
    the Rust ``run``'s ``Err(e) => writeln!(stderr, "fxtranslate: {e}")``."""
    try:
        cmd = p.parse(args)
    except p.CliError as e:
        io.stderr(f"fxtranslate: {e}\n")
        return 1

    try:
        return _dispatch(cmd, io)
    except Exception as e:  # noqa: BLE001 - mirror Rust's single error sink
        io.stderr(f"fxtranslate: {e}\n")
        return 1


def _dispatch(cmd: "p.Command", io: Io) -> int:
    if isinstance(cmd, p.Help):
        io.stdout(f"{USAGE}\n")
        return 0
    if isinstance(cmd, p.ListHelp):
        io.stdout(f"{LIST_USAGE}\n")
        return 0
    if isinstance(cmd, p.ModelsHelp):
        io.stdout(f"{MODELS_USAGE}\n")
        return 0
    if isinstance(cmd, p.ListCmd):
        return _run_list(io, cmd.query, cmd.all)
    if isinstance(cmd, p.ModelsList):
        return _run_models_list(io, cmd.cache_dir)
    if isinstance(cmd, p.ModelsInfo):
        return _run_models_info(io, cmd.name, cmd.cache_dir)
    if isinstance(cmd, p.ModelsAdd):
        return _run_models_add(io, cmd.src, cmd.trg, cmd.cache_dir)
    if isinstance(cmd, p.ModelsRemove):
        return _run_models_remove(io, cmd.name, cmd.all, cmd.cache_dir)
    if isinstance(cmd, p.Translate):
        return _run_translate(io, cmd.src, cmd.trg, cmd.text, cmd.cache_dir)
    raise RuntimeError("unreachable command")


def _run_list(io: Io, query: Optional[str], all: bool) -> int:
    """Fetch the Remote Settings records, then render either the language view
    (default) or the raw ``--all`` pair table via the compiled ``discovery`` surface.
    Mirrors Rust's ``Command::List`` arm — color only on a TTY stdout honoring
    ``NO_COLOR``, and the ``[N …]`` trailer on stderr."""
    body = discovery.fetch_records_body()
    color = io.stdout_is_tty and not io.no_color
    if all:
        pairs = [(s, t) for (s, t) in discovery.model_pairs(body)]
        n = write_pairs(pairs, query, color, io.stdout)
        io.stderr(f"[{n} pairs]\n")
    else:
        cat = discovery.catalog(body, PREFERRED_HUB)
        n = write_languages(cat, query, color, io.stdout)
        io.stderr(f"[{n} languages]\n")
    return 0


def _run_models_list(io: Io, cache_dir: Optional[str]) -> int:
    """``models list``: the cache location, a row per cached pair (``Source → Target``
    label, ``(dir-name)`` id, size), then the total — or a friendly note for an empty
    cache. Mirrors Rust's ``run_models_list``."""
    cache = Cache(cache_dir)
    io.stdout(f"Cache: {cache.root}\n")

    cached = cache.list_cached()
    if not cached:
        io.stdout("No models cached yet. Add one with `fxtranslate models add <src> <trg>`.\n")
        io.stderr("[0 cached]\n")
        return 0

    rows = []
    for c in cached:
        pretty = pretty_pair(c["name"])
        label = f"{pretty[0]} → {pretty[1]}" if pretty else c["name"]
        rows.append((label, f"({c['name']})", human_bytes(c["bytes"])))
    w_label = max((scalar_len(label) for label, _, _ in rows), default=0)
    w_tag = max((scalar_len(tag) for _, tag, _ in rows), default=0)

    color = io.stdout_is_tty and not io.no_color
    cyan, _green, dim, reset = palette(color)
    for label, tag, size in rows:
        label = pad_end(label, w_label)
        tag = pad_end(tag, w_tag)
        io.stdout(f"  {cyan}{label}{reset} {dim}{tag}{reset} {size}\n")
    total = sum(c["bytes"] for c in cached)
    io.stdout(f"Total: {human_bytes(total)}\n")
    io.stderr(f"[{len(cached)} cached]\n")
    return 0


def _run_models_info(io: Io, name: str, cache_dir: Optional[str]) -> int:
    """``models info <pair>``: a cached pair's files — each with its size and full
    on-disk path — plus the total, or a "not cached" note. Mirrors Rust's
    ``run_models_info``."""
    cache = Cache(cache_dir)
    files = cache.pair_files(name)
    if not files:
        io.stdout(f"{name} is not cached. Add it with `fxtranslate models add <src> <trg>`.\n")
        return 0

    io.stdout(f"{name} ({os.path.join(cache.root, name)})\n")
    w_name = max((scalar_len(fname) for fname, _, _ in files), default=0)
    sizes = [human_bytes(b) for _, b, _ in files]
    w_size = max((scalar_len(s) for s in sizes), default=0)
    for (fname, _b, path), size in zip(files, sizes):
        fname = pad_end(fname, w_name)
        size = pad_start(size, w_size)
        io.stdout(f"  {fname}  {size}  {path}\n")
    total = sum(b for _, b, _ in files)
    io.stdout(f"Total: {human_bytes(total)}\n")
    return 0


def _run_models_add(io: Io, src: str, trg: str, cache_dir: Optional[str]) -> int:
    """``models add <src> <trg>``: pre-download every file for the pair (both legs of a
    pivot) into the cache, without building an engine — the compiled
    ``discovery.add_models`` (core ``ensure_route_files``) does the resolve + verified
    download. Status goes to stderr and names the resolved hop. Mirrors Rust's
    ``run_models_add``."""
    io.stderr(f"[fxtranslate] downloading {src}→{trg} model…\n")
    route = discovery.add_models(src, trg, cache_dir, io.stderr_is_tty)
    if route["kind"] == "pivot":
        io.stderr(f"[fxtranslate] cached ({src}→{route['pivot']}→{trg}, pivot).\n")
    else:
        io.stderr(f"[fxtranslate] cached ({src}→{trg}).\n")
    return 0


def _run_models_remove(io: Io, name: Optional[str], all: bool, cache_dir: Optional[str]) -> int:
    """``models rm <pair>`` / ``--all``: delete one cached pair (or every pair), each
    removal reporting the space it reclaimed. Idempotent — removing an absent pair is a
    note, not an error. Byte-for-byte with Rust's ``run_models_remove``."""
    cache = Cache(cache_dir)

    if all:
        cached = cache.list_cached()
        if not cached:
            io.stdout("No models cached; nothing to remove.\n")
            return 0
        freed = 0
        for c in cached:
            cache.remove_pair(c["name"])
            io.stdout(f"Removed {c['name']} ({human_bytes(c['bytes'])})\n")
            freed += c["bytes"]
        io.stderr(f"[{len(cached)} removed, {human_bytes(freed)} reclaimed]\n")
        return 0

    # A specific pair: look it up first so we can report the reclaimed size.
    found = next((c for c in cache.list_cached() if c["name"] == name), None)
    if found:
        cache.remove_pair(name)
        io.stdout(f"Removed {name} ({human_bytes(found['bytes'])})\n")
    else:
        io.stdout(f"{name} is not cached.\n")
    return 0


def _run_translate(io: Io, src: str, trg: str, text: str, cache_dir: Optional[str]) -> int:
    """``translate <src> <trg> [text…]``: resolve+load the session, then translate the
    arg text (one line), piped stdin (one translation per line), or an interactive TTY
    REPL. Status lines go to stderr so piped stdout carries only translations.
    Byte-for-byte with Rust's ``run_translate`` on the interface (status/prompt/EOF);
    the translated TEXT is tolerant (per notes/13, Pass B)."""
    io.stderr(f"[fxtranslate] resolving {src}→{trg} model…\n")
    session = Translator.load(src, trg, cache_dir)
    # The native Translation reports a pivot internally; the Python surface doesn't
    # expose it, so re-derive the hop from the resolved route the same records give.
    pivot = _pivot_hop(src, trg)
    if pivot is not None:
        io.stderr(f"[fxtranslate] ready ({src}→{pivot}→{trg}, pivot).\n")
    else:
        io.stderr(f"[fxtranslate] ready ({src}→{trg}).\n")

    if text != "":
        io.stdout(f"{session.translate_long(text)}\n")
        return 0

    if io.stdin_is_tty:
        _repl(session, io, src, trg)
        return 0

    # Pipe mode: one translation per input line (marian-style).
    for line in _read_lines(io.stdin):
        io.stdout(f"{session.translate_long(line)}\n")
    return 0


def _pivot_hop(src: str, trg: str) -> Optional[str]:
    """The pivot language if ``src``→``trg`` has no direct model and is served by a
    two-leg pivot, else ``None`` — so ``translate`` can report the hop like the native
    CLI. Resolved from the same Remote Settings records the load used."""
    body = discovery.fetch_records_body()
    route = discovery.resolve_route(body, src, trg)
    return route["pivot"] if route["kind"] == "pivot" else None


def _repl(session, io: Io, src: str, trg: str) -> None:
    """Minimal interactive REPL: a prompt on stderr, a line in, its translation out,
    until EOF (Ctrl-D). Blank lines are skipped. Mirrors Rust's ``repl`` — the intro
    line, the ``src→trg» `` prompt (no newline), and the trailing newline at EOF."""
    io.stderr(f"Interactive {src}→{trg}. Type a sentence and press Enter; Ctrl-D to quit.\n")
    while True:
        io.stderr(f"{src}→{trg}» ")
        line = io.stdin.readline()
        if line == "":
            io.stderr("\n")  # EOF closes the final prompt line
            return
        text = line.strip()
        if text != "":
            io.stdout(f"{session.translate_long(text)}\n")


def _read_lines(stdin: TextIO):
    """Yield ``stdin`` one line at a time, stripping only the trailing ``\\n`` (and a
    preceding ``\\r``) — matching Rust's ``BufRead::read_line`` splitting. A final line
    with no trailing newline is still yielded."""
    for line in stdin:
        if line.endswith("\n"):
            line = line[:-1]
            if line.endswith("\r"):
                line = line[:-1]
        yield line
