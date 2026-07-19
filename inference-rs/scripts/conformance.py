#!/usr/bin/env python3
"""
Pass A of the centralized CLI conformance harness: interface conformance,
byte-exact.

Treats every fxtranslate CLI (the Rust reference, the npm binding, a future
Python one) as an opaque binary run as

    (argv, stdin, env, --cache-dir fixture, records fixture, NO_COLOR, tty) -> stdout + stderr + exit

Every case is run against every registered CLI and asserted byte-identical to a
golden generated once from the Rust CLI (the oracle). This is the *interface*
tier of the two-tier parity story in notes/13-cli-parity.md: help/usage text,
error strings, exit codes, argument grammar, the `list` tables (via a committed
records fixture, served over a local loopback HTTP server so no network is hit),
and `models list`/`info`/`rm` output (via a committed cache fixture, copied to a
throwaway dir per run so the user's real cache is never touched).

Deliberately OUT of Pass A: `translate` output (the tolerant wasm-libm tier —
Pass B) and `models add` (real downloads). Their interface bits (status lines,
REPL, pivot hop) are covered by the in-process Rust tests and join Pass B.

Usage:
    scripts/conformance.py                 # run every CLI against the goldens
    scripts/conformance.py --update-goldens   # regenerate goldens from the oracle
    scripts/conformance.py --only rust        # run just one registered CLI
    scripts/conformance.py --case list_all    # run just one case (substring match)

Exit 0 iff every registered CLI matched every golden.
"""

import argparse
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
CRATE_DIR = SCRIPTS_DIR.parent
FIXTURES_DIR = SCRIPTS_DIR / "conformance"
RECORDS_FIXTURE = FIXTURES_DIR / "records.json"
GOLDENS_DIR = FIXTURES_DIR / "goldens"

# The Rust CLI is built into a dedicated target dir (target/conformance) by the
# build-cli task, not the shared target/, so its dev-profile build doesn't thrash
# rs:test's test-profile cache when rs:check alternates between them. Anchored to
# CRATE_DIR so the path resolves the same whether invoked from the repo root or the
# inference-rs dir. Pass B (conformance_translate.py) imports this module and reuses
# the same REGISTRY, so it picks up this path too.
RUST_BIN = CRATE_DIR / "target" / "conformance" / "debug" / "fxtranslate"
NPM_BIN = CRATE_DIR / "npm" / "bin" / "fxtranslate.js"

# The Python binding is invoked as `python -m fxtranslate` (no PATH assumptions —
# it works from the editable/wheel install the maturin-develop dep produces). The
# same interpreter running this harness runs the CLI, so its editable install is the
# one exercised. `Cli.available()` special-cases this `-m` form: it probes that the
# compiled `fxtranslate` module actually imports, so a missing `maturin develop`
# SKIPs cleanly (like npm skips without its wasm build) rather than failing.
PY_MODULE = "fxtranslate"

# Placeholder substituted for the throwaway cache dir's absolute path in captured
# output, so goldens stay byte-portable across machines and CI.
CACHE_SENTINEL = b"<CACHE_DIR>"


# --- CLI registry -----------------------------------------------------------
# Adding a binding = one entry. `prefix` is prepended to each case's argv; `env`
# is merged over the hermetic base environment. `name` picks the golden set to
# check against (all bindings share the oracle's goldens). Order matters only for
# the report; the oracle should stay first so --update-goldens uses it.


@dataclass
class Cli:
    name: str
    prefix: list[str]
    is_oracle: bool = False
    env: dict[str, str] = field(default_factory=dict)

    def available(self) -> bool:
        """A CLI is runnable iff its entry point is present. Three forms:

        * `python -m <module>` — probe that `<module>` imports in the same
          interpreter running this harness (the compiled `fxtranslate` extension the
          maturin-develop dep builds), so a missing build SKIPs like npm does.
        * a `.js`/`.py` script — its file must exist.
        * the compiled Rust binary — the first argv element must exist.
        """
        if "-m" in self.prefix:
            module = self.prefix[self.prefix.index("-m") + 1]
            import importlib.util

            try:
                return importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                return False
        target = next(
            (p for p in self.prefix if p.endswith((".js", ".py"))),
            self.prefix[0],
        )
        return Path(target).exists()


REGISTRY = [
    Cli(name="rust", prefix=[str(RUST_BIN)], is_oracle=True),
    Cli(name="npm", prefix=["node", str(NPM_BIN)]),
    Cli(name="python", prefix=[sys.executable, "-m", PY_MODULE]),
]


# --- Case set ---------------------------------------------------------------
# A case is data: an id, argv, and how to supply I/O. `cache` selects a fixture
# recipe (below); `mutates` flags the destructive `models rm` cases so each CLI
# run gets its OWN fresh copy of the cache fixture. `records` requests the local
# records server (for `list`). `tty` sets stdin/stdout/stderr terminal-ness.


@dataclass
class Case:
    id: str
    argv: list[str]
    stdin: str = ""
    cache: str | None = None  # fixture recipe name, or None
    mutates: bool = False  # destructive: fresh cache copy per CLI run
    records: bool = False  # inject the local records fixture server
    tty: tuple[bool, bool, bool] = (False, False, False)  # (stdin, stdout, stderr)


# The cache fixture recipe: pair dir -> [(file name, size bytes)]. Mirrors the
# npm test/parity.js layout (already proven against the Rust cache reader): sizes
# span the raw-bytes / KiB / MiB (decimal) branches of `human_bytes`, and a
# dot-prefixed temp file proves it is skipped.
CACHE_RECIPE = {
    "en-es": [("model.enes.bin", 31), ("vocab.enes.spm", 31)],
    "en-fr": [("model.enfr.bin", 5120), (".model.enfr.bin.download", 9999)],
    "es-en": [("model.esen.bin", 1234567), ("vocab.esen.spm", 2048)],
}


def seed_cache(root: Path) -> None:
    """Materialize CACHE_RECIPE under `root` (fills bytes with 0x61 = 'a')."""
    for pair, files in CACHE_RECIPE.items():
        (root / pair).mkdir(parents=True, exist_ok=True)
        for name, size in files:
            (root / pair / name).write_bytes(b"a" * size)


CASES = [
    # grammar / help / error — pure, no I/O
    Case("help_long", ["--help"]),
    Case("help_short", ["-h"]),
    Case("no_args", []),
    Case("list_help", ["list", "--help"]),
    Case("list_help_short", ["list", "-h"]),
    Case("models_help", ["models", "--help"]),
    Case("translate_help", ["translate", "--help"]),
    Case("models_list_help", ["models", "list", "--help"]),
    Case("err_translate_no_args", ["translate"]),
    Case("err_translate_one_arg", ["translate", "en"]),
    Case("err_models_no_verb", ["models"]),
    Case("err_models_unknown_verb", ["models", "bogus"]),
    Case("err_models_add_one_arg", ["models", "add", "en"]),
    Case("err_models_rm_no_args", ["models", "rm"]),
    Case("err_models_info_no_args", ["models", "info"]),
    Case("err_unknown_command", ["bogus"]),
    Case("err_cache_dir_no_value", ["--cache-dir"]),
    # list — hermetic via the committed records fixture
    Case("list", ["list"], records=True),
    Case("list_lang", ["list", "es"], records=True),
    Case("list_all", ["list", "--all"], records=True),
    Case("list_lang_all", ["list", "es", "--all"], records=True),
    Case("list_unknown_lang", ["list", "zz"], records=True),
    # list, colored: stdout is a TTY and NO_COLOR is unset
    Case("list_tty_color", ["list"], records=True, tty=(False, True, False)),
    # models list / info — hermetic via the committed cache fixture (read-only)
    Case("models_list", ["models", "list"], cache="fixture"),
    Case("models_list_empty", ["models", "list"], cache="empty"),
    Case("models_info_enes", ["models", "info", "en-es"], cache="fixture"),
    Case("models_info_esen", ["models", "info", "es-en"], cache="fixture"),
    Case("models_info_enfr", ["models", "info", "en-fr"], cache="fixture"),
    Case("models_info_two_tag", ["models", "info", "en", "es"], cache="fixture"),
    Case("models_info_unknown", ["models", "info", "zz-zz"], cache="fixture"),
    # models rm — hermetic AND destructive: fresh cache copy per CLI run
    Case("rm_one_pair", ["models", "rm", "en-es"], cache="fixture", mutates=True),
    Case("rm_mib_pair", ["models", "rm", "es-en"], cache="fixture", mutates=True),
    Case("rm_two_tag", ["models", "rm", "en", "es"], cache="fixture", mutates=True),
    Case("rm_unknown", ["models", "rm", "zz-zz"], cache="fixture", mutates=True),
    Case("rm_all", ["models", "rm", "--all"], cache="fixture", mutates=True),
    Case("rm_all_empty", ["models", "rm", "--all"], cache="empty", mutates=True),
]


# --- Records fixture server -------------------------------------------------
# Both CLIs honor FXTRANSLATE_RECORDS_URL; point it at this loopback server so
# `list` reads the committed fixture instead of live Remote Settings. ureq (Rust)
# and Node fetch both speak plain HTTP, so no file:// support is needed.


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *_args):  # silence per-request logging
        pass


class RecordsServer:
    def __init__(self, body: bytes):
        handler = type("H", (_Handler,), {"body": body})
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/records"

    def __exit__(self, *_exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# --- Running a case ---------------------------------------------------------


def base_env(case: Case) -> dict[str, str]:
    """A hermetic, deterministic environment: no inherited FXTRANSLATE_* vars, and
    color driven only by NO_COLOR/TTY. Both NO_COLOR and FORCE_COLOR are dropped
    from the inherited environment first — an ambient FORCE_COLOR would otherwise
    make Node print a `NO_COLOR is ignored` warning to stderr and defeat the
    NO_COLOR we set below."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("FXTRANSLATE_") and k not in ("NO_COLOR", "FORCE_COLOR")
    }
    # A `_color` case exercises the colored path: it needs a stdout-TTY (supplied
    # by the pty runner) and NO_COLOR unset. Every other case is forced uncolored
    # (NO_COLOR set), so goldens never carry escape sequences by accident.
    colored = case.tty[1] and case.id.endswith("_color")
    if not colored:
        env["NO_COLOR"] = "1"
    return env


def run_case(cli: Cli, case: Case, records_url: str | None) -> tuple[bytes, bytes, int]:
    argv = list(case.argv)
    tmpdirs: list[Path] = []
    try:
        env = base_env(case)
        env.update(cli.env)
        if case.records:
            if records_url is None:
                raise RuntimeError("records case run without a server URL")
            env["FXTRANSLATE_RECORDS_URL"] = records_url
        if case.cache is not None:
            # A throwaway cache dir per run (fresh for the destructive `rm` cases,
            # never the user's real cache). Its absolute path leaks into the output,
            # so it's normalized to CACHE_SENTINEL below to keep goldens portable.
            cache_root = Path(tempfile.mkdtemp(prefix=f"fxconf-{cli.name}-"))
            tmpdirs.append(cache_root)
            if case.cache == "fixture":
                seed_cache(cache_root)
            # "empty" leaves the dir empty.
            argv = argv + ["--cache-dir", str(cache_root)]

        cmd = cli.prefix + argv
        if any(case.tty):
            out, err, code = run_with_tty(cmd, case, env)
        else:
            res = subprocess.run(cmd, input=case.stdin.encode(), capture_output=True, env=env)
            out, err, code = res.stdout, res.stderr, res.returncode

        if case.cache is not None:
            # The cache verbs echo the absolute cache path; canonicalize it to a
            # stable sentinel so goldens stay portable across machines/CI (the
            # path is a harness artifact, not part of the interface under test).
            token = str(cache_root).encode()
            out = out.replace(token, CACHE_SENTINEL)
            err = err.replace(token, CACHE_SENTINEL)
        return out, err, code
    finally:
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)


# --- TTY-backed runs --------------------------------------------------------
# Some behavior (colored `list`, the interactive REPL, the download bar) only
# fires when a stream is a real terminal — `is_terminal()` / `isatty()` probes
# the fd, which no env var can fake. To exercise those paths across opaque
# binaries we hand the child a pseudo-terminal on each flagged fd and a pipe on
# the rest, then capture stdout and stderr separately so they still diff against
# their own goldens. (stdin is only ever a read side; we feed `case.stdin`.)


def run_with_tty(cmd: list[str], case: Case, env: dict[str, str]) -> tuple[bytes, bytes, int]:
    import pty
    import select

    stdin_tty, stdout_tty, stderr_tty = case.tty

    # For each of stdout/stderr: a pty pair when the case marks it a terminal,
    # else an ordinary pipe. We read the parent-side fd either way.
    outs: dict[str, tuple[int, int, bool]] = {}
    for name, is_tty in (("stdout", stdout_tty), ("stderr", stderr_tty)):
        if is_tty:
            parent, child = pty.openpty()
        else:
            parent_r, child = os.pipe()
            parent = parent_r
        outs[name] = (parent, child, is_tty)

    if stdin_tty:
        parent_stdin, child_stdin = pty.openpty()
    else:
        child_stdin_r, parent_stdin = os.pipe()
        child_stdin = child_stdin_r

    proc = subprocess.Popen(
        cmd,
        stdin=child_stdin,
        stdout=outs["stdout"][1],
        stderr=outs["stderr"][1],
        env=env,
        close_fds=True,
    )
    # Parent closes the child ends it handed off.
    os.close(child_stdin)
    os.close(outs["stdout"][1])
    os.close(outs["stderr"][1])

    if case.stdin:
        os.write(parent_stdin, case.stdin.encode())
    os.close(parent_stdin)  # EOF for the child's stdin

    captured = {"stdout": b"", "stderr": b""}
    read_fds = {outs["stdout"][0]: "stdout", outs["stderr"][0]: "stderr"}
    open_fds = set(read_fds)
    while open_fds:
        ready, _, _ = select.select(list(open_fds), [], [], 5.0)
        if not ready:
            break
        for fd in ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:  # pty raises EIO at EOF
                chunk = b""
            if chunk:
                captured[read_fds[fd]] += chunk
            else:
                open_fds.discard(fd)
                os.close(fd)
    for fd in open_fds:
        os.close(fd)
    code = proc.wait()
    # ptys translate \n to \r\n on output; normalize so goldens stay LF-only and
    # a piped-vs-tty diff is purely about content (color), not line endings.
    out = captured["stdout"].replace(b"\r\n", b"\n") if stdout_tty else captured["stdout"]
    err = captured["stderr"].replace(b"\r\n", b"\n") if stderr_tty else captured["stderr"]
    return out, err, code


# --- Goldens ----------------------------------------------------------------


def golden_paths(case_id: str) -> tuple[Path, Path, Path]:
    base = GOLDENS_DIR / case_id
    return (
        base.with_suffix(".out"),
        base.with_suffix(".err"),
        base.with_suffix(".exit"),
    )


def write_golden(case_id: str, out: bytes, err: bytes, code: int) -> None:
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    p_out, p_err, p_exit = golden_paths(case_id)
    p_out.write_bytes(out)
    p_err.write_bytes(err)
    p_exit.write_text(f"{code}\n")


def read_golden(case_id: str) -> tuple[bytes, bytes, int]:
    p_out, p_err, p_exit = golden_paths(case_id)
    return p_out.read_bytes(), p_err.read_bytes(), int(p_exit.read_text().strip())


# --- Reporting (parity.py style) --------------------------------------------


def unified_diff(label: str, expected: bytes, actual: bytes) -> str:
    import difflib

    exp = expected.decode("utf-8", "replace").splitlines(keepends=True)
    act = actual.decode("utf-8", "replace").splitlines(keepends=True)
    diff = difflib.unified_diff(exp, act, fromfile=f"golden {label}", tofile=f"actual {label}")
    return "".join(diff)


def compare(cli: Cli, case: Case, got: tuple[bytes, bytes, int]) -> list[str]:
    out, err, code = got
    exp_out, exp_err, exp_code = read_golden(case.id)
    diffs = []
    if code != exp_code:
        diffs.append(f"exit: golden={exp_code} actual={code}")
    if out != exp_out:
        diffs.append("stdout differs\n" + unified_diff("stdout", exp_out, out))
    if err != exp_err:
        diffs.append("stderr differs\n" + unified_diff("stderr", exp_err, err))
    return diffs


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--update-goldens",
        "--regenerate",
        action="store_true",
        help="regenerate goldens from the oracle CLI, then exit",
    )
    parser.add_argument("--only", help="run just the named CLI (e.g. rust, npm)")
    parser.add_argument("--case", help="run just cases whose id contains this substring")
    args = parser.parse_args()

    cases = CASES
    if args.case:
        cases = [c for c in CASES if args.case in c.id]
        if not cases:
            print(f"no case matches {args.case!r}", file=sys.stderr)
            sys.exit(2)

    records_body = RECORDS_FIXTURE.read_bytes()

    if args.update_goldens:
        oracle = next(c for c in REGISTRY if c.is_oracle)
        if not oracle.available():
            print(
                f"oracle binary missing: {oracle.prefix}\n"
                "build it with: cargo build -p fxtranslate-cli",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"regenerating goldens from the oracle ({oracle.name}):")
        with RecordsServer(records_body) as srv:
            for case in cases:
                out, err, code = run_case(oracle, case, srv.url)
                write_golden(case.id, out, err, code)
                print(f"  wrote {case.id}")
        print(f"{len(cases)} goldens written to {GOLDENS_DIR}")
        return

    registry = REGISTRY
    if args.only:
        registry = [c for c in REGISTRY if c.name == args.only]
        if not registry:
            print(f"no CLI named {args.only!r} in the registry", file=sys.stderr)
            sys.exit(2)

    overall_fail = 0
    with RecordsServer(records_body) as srv:
        for cli in registry:
            if not cli.available():
                print(f"\n{cli.name}: SKIP (binary missing: {cli.prefix})")
                continue
            print(f"\n{cli.name} vs goldens:")
            passed = 0
            failed = 0
            for case in cases:
                got = run_case(cli, case, srv.url)
                diffs = compare(cli, case, got)
                label = f"fxtranslate {' '.join(case.argv)}".strip() + f"  [{case.id}]"
                if not diffs:
                    print(f"  ok   {label}")
                    passed += 1
                else:
                    print(f"  FAIL {label}")
                    for d in diffs:
                        for line in d.splitlines():
                            print(f"         {line}")
                    failed += 1
            print(f"  {passed}/{passed + failed} byte-identical.")
            overall_fail += failed

    sys.exit(1 if overall_fail else 0)


if __name__ == "__main__":
    main()
