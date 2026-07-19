"""The ``list`` and ``models`` view formatters — a byte-for-byte port of the Rust
CLI's rendering (crates/fxtranslate-cli/src/cli.rs). Presentation is a shell concern
(duplicated on purpose, per notes/13), but pinned against the Rust oracle: every
column width, size unit, sort order, and trailer must match. The pure decisions
these render (catalog classification, version-gated pairs) come from the compiled
``discovery`` surface, not from here.
"""

from typing import Callable, List, Optional, Tuple

from .lang import display_name

# The production hub. Mirrors `route::PREFERRED_HUB`.
PREFERRED_HUB = "en"


def scalar_len(s: str) -> int:
    """Unicode scalar (code-point) count of ``s`` — what Rust's ``chars().count()``
    and ``{:<width$}`` measure. Python ``len`` already counts code points, so this is
    just ``len``; kept named for parity with the Rust/JS intent."""
    return len(s)


def pad_end(s: str, width: int) -> str:
    """Left-justify (pad right) ``s`` to ``width`` scalars — Rust ``{:<width$}``."""
    return s.ljust(width)


def pad_start(s: str, width: int) -> str:
    """Right-justify (pad left) ``s`` to ``width`` scalars — Rust ``{:>width$}``."""
    return s.rjust(width)


def human_bytes(n: int) -> str:
    """Human-readable byte count on the base-1024 scale — raw bytes under 1 KiB, then
    one decimal of KiB/MiB/GiB. Mirrors Rust's ``human_bytes``. Python's ``%.1f``
    rounds half-to-even, matching Rust's float formatter."""
    kib = 1024.0
    mib = kib * 1024.0
    gib = mib * 1024.0
    f = float(n)
    if f >= gib:
        return f"{f / gib:.1f} GiB"
    if f >= mib:
        return f"{f / mib:.1f} MiB"
    if f >= kib:
        return f"{f / kib:.1f} KiB"
    return f"{n} B"


def palette(color: bool) -> Tuple[str, str, str, str]:
    """ANSI palette ``(cyan, green, dim, reset)``, or empty strings when color is off.
    Mirrors Rust's ``palette``."""
    if color:
        return ("\x1b[36m", "\x1b[32m", "\x1b[2m", "\x1b[0m")
    return ("", "", "", "")


def language_query_matches(lang: str, query: str) -> bool:
    """Whether a ``list`` language view surfaces ``lang`` for ``query``. Mirrors Rust's
    ``language_query_matches``."""
    dash = query.find("-")
    if dash >= 0:
        a = query[:dash]
        b = query[dash + 1 :]
        return lang.startswith(a) or lang.startswith(b)
    return lang.startswith(query)


def language_matches(src: str, trg: str, query: str) -> bool:
    """Whether a ``list --all`` query selects the pair ``src → trg``. Mirrors Rust's
    ``remote::language_matches`` — a shallow prefix match; the version gate that makes
    this list correct lives in the compiled ``model_pairs``."""
    dash = query.find("-")
    if dash >= 0:
        q_src = query[:dash]
        q_trg = query[dash + 1 :]
        return src.startswith(q_src) and trg.startswith(q_trg)
    return src.startswith(query) or trg.startswith(query)


def _render_pairs(rows: List[Tuple[str, str]], color: bool, out: Callable[[str], None]) -> None:
    """Render the aligned ``src → trg`` table — the ``--all`` view. Names and the
    source tag are padded before color-wrapping. Mirrors Rust's ``render_pairs``."""
    w_src = max((scalar_len(display_name(s)) for s, _ in rows), default=0)
    w_stag = max((scalar_len(s) + 2 for s, _ in rows), default=0)
    w_trg = max((scalar_len(display_name(t)) for _, t in rows), default=0)
    cyan, green, dim, reset = palette(color)
    for s, t in rows:
        sname = pad_end(display_name(s), w_src)
        stag = pad_end(f"({s})", w_stag)
        tname = pad_end(display_name(t), w_trg)
        out(
            f"{cyan}{sname}{reset} {dim}{stag}{reset} {dim}→{reset} "
            f"{green}{tname}{reset} {dim}({t}){reset}\n"
        )


def _render_single_direction(
    rows: List[Tuple[str, str]], color: bool, out: Callable[[str], None]
) -> None:
    """Render single-direction rows as ``source → target (src trg)``. Mirrors Rust's
    ``render_single_direction``."""
    w_src = max((scalar_len(display_name(s)) for s, _ in rows), default=0)
    w_trg = max((scalar_len(display_name(t)) for _, t in rows), default=0)
    cyan, green, dim, reset = palette(color)
    for s, t in rows:
        sname = pad_end(display_name(s), w_src)
        tname = pad_end(display_name(t), w_trg)
        out(f"{cyan}{sname}{reset} {dim}→{reset} {green}{tname}{reset} {dim}({s} {t}){reset}\n")


def write_pairs(
    all_pairs: List[Tuple[str, str]],
    query: Optional[str],
    color: bool,
    out: Callable[[str], None],
) -> int:
    """The ``list --all`` view from the version-gated model pairs. Filters by ``query``
    (via :func:`language_matches`) and renders the raw ``src → trg`` table. Returns the
    number of pairs shown, or raises with the Rust no-match message."""
    shown = [(s, t) for (s, t) in all_pairs if query is None or language_matches(s, t, query)]
    if not shown:
        raise ValueError(
            f"no model pairs match `{query or ''}` ({len(all_pairs)} pairs available; "
            "try `fxtranslate list --all`)"
        )
    _render_pairs(shown, color, out)
    return len(shown)


def write_languages(
    cat: dict,
    query: Optional[str],
    color: bool,
    out: Callable[[str], None],
) -> int:
    """The default ``list`` view: languages, not raw pairs. Renders the "Fully
    supported" and "Single-direction models" sections from the compiled catalog.
    Returns the number of languages shown, or raises with the Rust no-match message.
    Mirrors Rust's ``write_languages``."""

    def keep(l: str) -> bool:
        return query is None or language_query_matches(l, query)

    bidi = [l for l in cat["bidirectional"] if keep(l)]
    source_only = [l for l in cat["sourceOnly"] if keep(l)]
    target_only = [l for l in cat["targetOnly"] if keep(l)]
    total = len(bidi) + len(source_only) + len(target_only)
    if total == 0:
        raise ValueError(
            f"no languages match `{query or ''}` "
            "(try `fxtranslate list`, or `list --all` for raw pairs)"
        )

    cyan, _green, dim, reset = palette(color)

    if bidi:
        out("Fully supported (translate to and from any other):\n")
        w = max((scalar_len(display_name(l)) for l in bidi), default=0)
        for l in bidi:
            name = pad_end(display_name(l), w)
            out(f"  {cyan}{name}{reset} {dim}({l}){reset}\n")

    # Single-direction: hub→L (target-only) then L→hub (source-only).
    one_way: List[Tuple[str, str]] = [(PREFERRED_HUB, l) for l in target_only] + [
        (l, PREFERRED_HUB) for l in source_only
    ]
    if one_way:
        if bidi:
            out("\n")
        out("Single-direction models:\n")
        _render_single_direction(one_way, color, out)

    return total


def pretty_pair(name: str) -> Optional[Tuple[str, str]]:
    """A cached pair's ``Source → Target`` label, or ``None`` when the directory name
    doesn't carry the hub on one side. Mirrors Rust's ``pretty_pair``."""
    hub = PREFERRED_HUB
    if name.startswith(f"{hub}-"):
        return (display_name(hub), display_name(name[len(hub) + 1 :]))
    if name.endswith(f"-{hub}"):
        return (display_name(name[: len(name) - len(hub) - 1]), display_name(hub))
    return None
