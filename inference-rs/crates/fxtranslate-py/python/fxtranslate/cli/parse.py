"""The argument grammar, help routing, and error strings — a faithful port of the
Rust CLI's pure ``parse``/``parse_models`` (crates/fxtranslate-cli/src/cli.rs). No
I/O here: argv in, a :class:`Command` out (or a :class:`CliError` raised), so the
whole grammar is pinned byte-for-byte against the Rust oracle.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from .usage import LIST_USAGE, MODELS_USAGE, USAGE


class CliError(Exception):
    """A parse error carrying the exact message the Rust CLI prints after the
    ``fxtranslate: `` prefix. :func:`parse` raises this; the shell renders it and
    exits 1, matching ``run``'s ``Err(e) => writeln!(stderr, "fxtranslate: {e}")``."""


@dataclass
class Help:
    """Print top-level usage."""


@dataclass
class ListHelp:
    """Print ``list``-specific usage."""


@dataclass
class ModelsHelp:
    """Print ``models``-specific usage."""


@dataclass
class ListCmd:
    query: Optional[str]
    all: bool


@dataclass
class Translate:
    src: str
    trg: str
    text: str
    cache_dir: Optional[str]


@dataclass
class ModelsList:
    cache_dir: Optional[str]


@dataclass
class ModelsAdd:
    src: str
    trg: str
    cache_dir: Optional[str]


@dataclass
class ModelsRemove:
    name: Optional[str]
    all: bool
    cache_dir: Optional[str]


@dataclass
class ModelsInfo:
    name: str
    cache_dir: Optional[str]


Command = Union[
    Help,
    ListHelp,
    ListCmd,
    Translate,
    ModelsHelp,
    ModelsList,
    ModelsAdd,
    ModelsRemove,
    ModelsInfo,
]


def parse(args: List[str]) -> Command:
    """Parse argv (without the program name) into a :class:`Command`. Pure — no I/O.
    Raises :class:`CliError` with the Rust-identical message on a grammar error."""
    cache_dir: Optional[str] = None
    positional: List[str] = []
    help = False
    all = False

    it = iter(range(len(args)))
    for i in it:
        a = args[i]
        if a in ("-h", "--help"):
            help = True
        elif a == "--all":
            all = True
        elif a == "--cache-dir":
            if i + 1 >= len(args):
                raise CliError("--cache-dir needs a path")
            cache_dir = args[i + 1]
            next(it, None)  # consume the value
        else:
            positional.append(a)

    first = positional[0] if positional else None
    if first == "list":
        if help:
            return ListHelp()
        return ListCmd(query=positional[1] if len(positional) > 1 else None, all=all)
    if first == "models":
        return _parse_models(positional, cache_dir, all, help)

    # `--help` with a non-list (or no) command → top-level help.
    if help:
        return Help()
    if first is None:
        return Help()
    if first == "translate":
        # `translate` is explicit: `translate <src> <trg> [text…]`.
        if len(positional) < 3:
            raise CliError(
                f"`translate` needs `<src> <trg> [text…]`; got `{' '.join(positional)}`\n\n{USAGE}"
            )
        return Translate(
            src=positional[1],
            trg=positional[2],
            text=" ".join(positional[3:]),
            cache_dir=cache_dir,
        )
    raise CliError(
        f"unknown command `{first}`; expected `translate`, `list`, or `models`\n\n{USAGE}"
    )


def _parse_models(
    positional: List[str], cache_dir: Optional[str], all: bool, help: bool
) -> Command:
    """Parse a ``models <cmd> …`` invocation. ``cache_dir``/``all`` were already
    lifted out by :func:`parse`. Mirrors the Rust ``parse_models``."""
    if help:
        return ModelsHelp()
    sub = positional[1] if len(positional) > 1 else None
    if sub is None:
        return ModelsHelp()
    if sub == "list":
        return ModelsList(cache_dir=cache_dir)
    if sub == "add":
        if len(positional) < 4:
            raise CliError(
                f"`models add` needs `<src> <trg>`; got `{' '.join(positional[1:])}`\n\n{MODELS_USAGE}"
            )
        return ModelsAdd(src=positional[2], trg=positional[3], cache_dir=cache_dir)
    if sub == "rm":
        if all:
            return ModelsRemove(name=None, all=True, cache_dir=cache_dir)
        name = pair_name(positional[2:])
        if name == "":
            raise CliError(
                f"`models rm` needs a `<pair>` (e.g. `en-es` or `en es`) or `--all`\n\n{MODELS_USAGE}"
            )
        return ModelsRemove(name=name, all=False, cache_dir=cache_dir)
    if sub == "info":
        name = pair_name(positional[2:])
        if name == "":
            raise CliError(
                f"`models info` needs a `<pair>` (e.g. `en-es` or `en es`)\n\n{MODELS_USAGE}"
            )
        return ModelsInfo(name=name, cache_dir=cache_dir)
    raise CliError(
        f"unknown `models` subcommand `{sub}`; expected `list`, `add`, `rm`, or `info`\n\n{MODELS_USAGE}"
    )


def pair_name(tokens: List[str]) -> str:
    """Build the ``<src>-<trg>`` cache-directory name from a pair argument. Both the
    two-tag form (``["en", "es"]``) and the joined form (``["en-es"]``) collapse to
    ``en-es``. Mirrors the Rust ``pair_name`` (``tokens.join("-")``)."""
    return "-".join(tokens)
