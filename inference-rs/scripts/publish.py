#!/usr/bin/env python3
"""
Release the fxtranslate ecosystem as one unit: the Rust crates to crates.io, the
npm package to the npm registry, and the Python package to PyPI, on a single shared
version.

The four artifacts — the `fxtranslate` engine crate, the `fxtranslate-cli` crate, the
`fxtranslate` npm package, and the `fxtranslate` PyPI package — are all the same engine
(the CLI pins the engine exactly; npm embeds it compiled to wasm; PyPI compiles it into
a native extension). A version that shipped to one but not the others, or at different
numbers, would be meaningless, so this refuses to let them drift: everything bumps
together and publishes together, or the run stops.

The PyPI leg has one twist: wheels are per-platform. The first release ships an sdist
plus the release machine's local-platform wheel; the full binary-wheel matrix is
backfilled by CI on the tag. The actual PyPI upload is left to the human by default
(the crates.io + npm push is fully automated); pass `--pypi-upload` to include it.

The safety story is the reason this is a script and not a handful of commands. A
registry upload is effectively permanent (it can only be yanked/deprecated, never
replaced), and publishing several of them is not atomic. So the design is: verify auth
and do all the reversible work first (bump, build, test, validate packaging), then the
irreversible uploads, and only once every upload has landed create the git tag — the
one atomic step — so the tag can never mark a half-published release.

## The orchestrator: status derived from the world, never stored

The release runs as a two-pass status board of ordered `Step`s (the visual language of
`task rs:check`; see scripts/statusboard.py). The trap with a resumable multi-registry
publisher is local state — a sidecar file that says "crates done, npm not" drifts the
moment anything happens out of band. So every Step instead knows how to ask the world
whether it is already done: `probe()` is a read-only registry/git query (crates.io
sparse index, `npm view`, the PyPI JSON API, `git ls-remote`) returning DONE / PENDING /
BLOCKED. Status is *computed*, never stored, which is what makes "run it over and over
until it passes" safe and self-healing: the tool re-derives reality each time, skips
what's green, and retries what isn't.

  1. Probe pass — every Step's `probe()` runs (fanned out in parallel, since probes are
     read-only) and the board renders. A previous failed run surfaces here as red rows
     before anything executes; `--status` stops here.
  2. Execute pass — Steps run IN ORDER (the chain is dependent: the CLI crate pins the
     engine, npm/PyPI publish after crates, the tag is last). DONE steps are skipped,
     PENDING steps run with stdout hidden and a live •/✓/x, and on the FIRST failure the
     full captured log plus the step's healing hint and a "next" line are dumped and the
     run stops. Uploads stay strictly serialized.

## Resume semantics: derive the target version, refuse the double-bump

The one piece of "state" is the target version, and it lives in the manifests, not a
sidecar. A re-run of `patch` would otherwise bump AGAIN (0.4.1 -> 0.4.2 -> 0.4.3), so:

  - manifest version is fully published on crates.io -> the tree is at a clean released
    state. A bare re-run has nothing to do ("specify patch|minor|major to start a
    release"); a bump arg starts a new release.
  - manifest version is NOT yet fully published -> a release for V is in flight. A bare
    re-run (or `--resume`) resumes it at V; a bump arg is refused ("a release for V is in
    flight; re-run without a level to resume, or --set to override").

This replaces the old `--initial` flag and its double-bump footgun; resume is the
no-argument path.

`--dry-run` renders the SAME board; its execute pass actually runs only the reversible
steps (cargo build + test and the npm/maturin packaging dry-runs — the two steps flagged
`runs_in_dry_run`) and renders every irreversible step as a neutral PREVIEW row (not
executed, no green ✓, no red x — the probe detail plus a "(dry-run)" note), so it touches
no registry, file, or git ref while staying as consistent as the real thing. A blocked
gate (e.g. not logged into npm) also previews neutral in a dry-run, since a dry-run needs
no auth. Operator setup, the checklist, and first-time-publish notes live in RELEASING.md;
run with -h for flags.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from statusboard import (
    IS_INTERACTIVE,
    Group,
    InteractiveRenderer,
    Row,
    color,
    format_duration,
    status_glyph,
    styled,
)

WORKSPACE = Path(__file__).resolve().parent.parent  # inference-rs/
ROOT_MANIFEST = WORKSPACE / "Cargo.toml"
NPM_DIR = WORKSPACE / "npm"  # the npm package (wasm core copied in by build:wasm)
NPM_MANIFEST = NPM_DIR / "package.json"
PY_DIR = WORKSPACE / "crates" / "fxtranslate-py"  # the PyPI package (maturin/PyO3 wheel)
# The py crate's Cargo.toml is the single version source: pyproject declares
# `dynamic = ["version"]`, so maturin reads the crate version at build time and
# `rewrite_versions` already bumps it in lockstep (it's a workspace member). No
# separate pyproject version string exists to rewrite or drift-check.
PY_MANIFEST = PY_DIR / "Cargo.toml"
PY_DIST = PY_DIR / "dist"  # gitignored; where the dry-run/release build drops the sdist + wheel
CHANGELOG = WORKSPACE / "CHANGELOG.md"  # one workspace changelog, copied into each crate
TAG_PREFIX = "fxtranslate-v"  # bare `0.1.0` etc. are taken by old repo tags
PYPI_NAME = "fxtranslate"  # the PyPI distribution name
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_NAME}/json"

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def log(msg: str) -> None:
    print(f"[publish] {msg}", file=sys.stderr)


def die(msg: str) -> None:
    sys.exit(f"[publish] error: {msg}")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run capturing stdout+stderr as text (for inspection)."""
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


class Aborted(Exception):
    """Raised by the run helpers below to unwind out of a Step's run() body.

    The procedural code these bodies were lifted from called `die()` (which exits the
    process) on any command failure. Under the orchestrator a Step failure must instead
    surface as a StepError so the board can dump the log + hint and stop cleanly, so the
    run helpers raise this and `Step.execute` converts it. The message carries the reason.
    """


def fail(msg: str) -> None:
    raise Aborted(msg)


def run(cmd: list[str], **kw) -> None:
    """Run a command as part of a Step body, capturing its output. On failure raise
    Aborted (the orchestrator turns it into a StepError carrying the captured log). This
    replaces the old streaming `run()`/`die()` pair: under the board stdout is hidden and
    surfaced only on failure, so everything a Step runs must be captured, not streamed."""
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        fail("command failed: " + " ".join(cmd))


class Crate:
    def __init__(self, name: str, version: str, publishable: bool, manifest: Path, deps: set[str]):
        self.name = name
        self.version = version
        self.publishable = publishable
        self.manifest = manifest
        self.deps = deps  # intra-workspace dependency crate names

    def __repr__(self) -> str:
        return f"Crate({self.name} {self.version} publishable={self.publishable})"


def load_workspace() -> list[Crate]:
    root = tomllib.loads(ROOT_MANIFEST.read_text())
    members = root.get("workspace", {}).get("members", [])
    if not members:
        die(f"no [workspace].members in {ROOT_MANIFEST}")

    by_dir: dict[str, str] = {}  # member dir -> crate name (to resolve path deps)
    raw: list[tuple[dict, Path]] = []
    for m in members:
        manifest = WORKSPACE / m / "Cargo.toml"
        if not manifest.is_file():
            die(f"workspace member {m} has no Cargo.toml at {manifest}")
        data = tomllib.loads(manifest.read_text())
        by_dir[m] = data["package"]["name"]
        raw.append((data, manifest))

    crates: list[Crate] = []
    for data, manifest in raw:
        pkg = data["package"]
        # `publish = false` (or an empty registry allowlist) means "never publish".
        publish = pkg.get("publish", True)
        publishable = publish is not False and publish != []
        # Intra-workspace deps: a dependency whose table carries a `path`.
        deps: set[str] = set()
        for section in ("dependencies", "build-dependencies"):
            for dname, spec in data.get(section, {}).items():
                if isinstance(spec, dict) and "path" in spec:
                    deps.add(dname)
        crates.append(Crate(pkg["name"], pkg["version"], publishable, manifest, deps))
    return crates


def publish_order(crates: list[Crate]) -> list[Crate]:
    """Publishable crates, dependencies before dependents (topological)."""
    publishable = {c.name: c for c in crates if c.publishable}
    ordered: list[Crate] = []
    seen: set[str] = set()

    def visit(c: Crate, stack: set[str]) -> None:
        if c.name in seen:
            return
        if c.name in stack:
            die(f"dependency cycle through {c.name}")
        stack.add(c.name)
        for dep in sorted(c.deps):
            if dep in publishable:
                visit(publishable[dep], stack)
        stack.discard(c.name)
        seen.add(c.name)
        ordered.append(c)

    for c in sorted(publishable.values(), key=lambda c: c.name):
        visit(c, set())
    return ordered


NPM_VERSION = re.compile(r'^(?P<pre>\s*"version":\s*)"(?P<ver>[^"]+)"', re.MULTILINE)


def npm_version() -> str:
    """The npm package's declared version (the top-level `"version"` in package.json)."""
    m = NPM_VERSION.search(NPM_MANIFEST.read_text())
    if not m:
        die(f'could not find a top-level "version" in {NPM_MANIFEST}')
    return m.group("ver")


def current_version(crates: list[Crate]) -> str:
    """The single lockstep version, shared by every crate AND the npm package — refuse
    to proceed if they've drifted."""
    parts = {c.name: c.version for c in crates}
    parts["npm/fxtranslate"] = npm_version()
    versions = set(parts.values())
    if len(versions) != 1:
        detail = ", ".join(f"{name}={v}" for name, v in parts.items())
        die(f"versions have drifted ({detail}); ecosystem lockstep is required — reconcile first")
    return versions.pop()


def rewrite_npm_version(old: str, new: str, apply: bool) -> list[str]:
    """Set the npm package's version to `new` in package.json (the top-level `"version"`,
    anchored to the JSON key — never a dependency spec) and, if present, in
    package-lock.json (its two package-self `"version"` entries: the top-level and the
    root `""` package, which precede every dependency). Returns human-readable
    descriptions; only writes when `apply`."""
    changes: list[str] = []

    text = NPM_MANIFEST.read_text()
    new_text, n = NPM_VERSION.subn(lambda m: f'{m.group("pre")}"{new}"', text, count=1)
    if n != 1:
        die(f'{NPM_MANIFEST.relative_to(WORKSPACE)}: could not find "version": "{old}" to bump')
    if apply:
        NPM_MANIFEST.write_text(new_text)
    changes.append(f"{NPM_MANIFEST.relative_to(WORKSPACE)}: package version {old} -> {new}")

    lock = NPM_DIR / "package-lock.json"
    if lock.is_file():
        lock_re = re.compile(rf'("version":\s*)"{re.escape(old)}"')
        lock_text, n_lock = lock_re.subn(rf'\g<1>"{new}"', lock.read_text(), count=2)
        if n_lock:
            if apply:
                lock.write_text(lock_text)
            changes.append(f"npm/package-lock.json: {n_lock} package-self version {old} -> {new}")
    return changes


def bump(version: str, level: str) -> str:
    m = SEMVER.match(version)
    if not m:
        die(f"current version {version!r} is not a plain MAJOR.MINOR.PATCH; use --set")
    major, minor, patch = (int(g) for g in m.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def rewrite_versions(crates: list[Crate], old: str, new: str, apply: bool) -> list[str]:
    """Set every crate's package version to `new`, and every exact intra-workspace
    pin (`version = "=OLD"`) to `=NEW`. Returns human-readable descriptions of each
    edit; only writes when `apply`."""
    changes: list[str] = []
    pkg_re = re.compile(rf'^version = "{re.escape(old)}"$', re.MULTILINE)
    pin_old, pin_new = f'version = "={old}"', f'version = "={new}"'
    for c in crates:
        text = c.manifest.read_text()
        new_text, n_pkg = pkg_re.subn(f'version = "{new}"', text, count=1)
        n_pin = new_text.count(pin_old)
        new_text = new_text.replace(pin_old, pin_new)
        rel = c.manifest.relative_to(WORKSPACE)
        if n_pkg:
            changes.append(f"{rel}: package version {old} -> {new}")
        else:
            die(f'{rel}: could not find `version = "{old}"` to bump')
        if n_pin:
            changes.append(f"{rel}: {n_pin} exact pin(s) ={old} -> ={new}")
        if apply:
            c.manifest.write_text(new_text)
    return changes


def readme_report(crates: list[Crate], old: str) -> None:
    """Publishable crates should ship a README (cargo warns otherwise), and stale
    version strings in any README want a look before release. Report both — never
    silently rewrite README prose, since a bare version string is easy to false-match."""
    for c in crates:
        if c.publishable:
            data = tomllib.loads(c.manifest.read_text())
            has_field = "readme" in data.get("package", {})
            has_file = (c.manifest.parent / "README.md").is_file()
            if not (has_field or has_file):
                log(f"WARNING: {c.name} has no README.md / readme = (crates.io recommends one)")

    for readme in sorted(WORKSPACE.glob("**/README.md")):
        if "target" in readme.parts:
            continue
        hits = [
            f"    {readme.relative_to(WORKSPACE)}:{i}: {line.strip()}"
            for i, line in enumerate(readme.read_text().splitlines(), 1)
            if old in line
        ]
        if hits:
            log(f"README mentions the old version {old} — review before publishing:")
            for h in hits:
                print(h, file=sys.stderr)


def git(*args: str) -> str:
    r = sh(["git", "-C", str(WORKSPACE), *args])
    if r.returncode != 0:
        die(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def git_try(*args: str) -> subprocess.CompletedProcess:
    """Like `git` but returns the CompletedProcess instead of dying, for read-only
    probes that must tolerate failure (network unreachable, missing ref, etc.)."""
    return sh(["git", "-C", str(WORKSPACE), *args])


def check_npm_auth() -> None:
    """Confirm npm is logged in BEFORE the first irreversible upload. Crates publish
    first, so a logged-out release would otherwise push the crates to crates.io and only
    then fail at npm — leaving a half-published release. npm also reports a logged-out
    first publish as a bare 404 (it won't say "unauthorized" for a package you can't
    see), so check `npm whoami` here and fail fast with a clear reason."""
    if not shutil.which("npm"):
        fail("npm not found on PATH; it's needed to publish the npm package")
    r = sh(["npm", "whoami"])
    if r.returncode != 0:
        fail(
            "not authenticated to npm — run `npm login` first (crates.io is published "
            "first, so releasing while logged out would half-publish the release)"
        )
    log(f"npm authenticated as {r.stdout.strip()}")


def npm_auth_status() -> tuple[bool, str]:
    """Read-only probe for npm auth: `(ok, detail)` — `npm whoami` succeeds and returns
    the logged-in user, else not authenticated."""
    if not shutil.which("npm"):
        return False, "npm not on PATH"
    r = sh(["npm", "whoami"])
    if r.returncode != 0:
        return False, "not logged in"
    return True, r.stdout.strip()


def check_pypi_publish_readiness() -> None:
    """Confirm the PyPI toolchain is present BEFORE any irreversible upload, in the same
    up-front block as check_npm_auth. Because the operator owns the actual PyPI upload
    (default: the release does everything except the push), this is a readiness/tooling
    check, not a credential login: `maturin` builds the sdist + wheel and `twine` uploads
    them, so both must be on PATH — fail with an install hint if either is missing. A
    missing PyPI credential source is only a WARNING, not fatal: a dry-run (and the
    default no-upload release) still does all its reversible work, and the operator supplies
    creds for the actual `twine upload` (or `--pypi-upload`) themselves."""
    missing = [t for t in ("maturin", "twine") if not shutil.which(t)]
    if missing:
        fail(
            f"{' and '.join(missing)} not found on PATH; needed to build/upload the PyPI "
            "package. Install with `pipx install maturin twine`"
        )
    if not pypi_has_credentials():
        log(
            "WARNING: no PyPI credential source found (~/.pypirc or TWINE_USERNAME/"
            "TWINE_PASSWORD/TWINE_API_KEY); a `twine upload` (or --pypi-upload) will need one"
        )


def pypi_has_credentials() -> bool:
    """A credential source lets `twine upload` authenticate: `~/.pypirc` or a TWINE_* env."""
    has_pypirc = (Path.home() / ".pypirc").is_file()
    has_env = any(os.environ.get(v) for v in ("TWINE_USERNAME", "TWINE_PASSWORD", "TWINE_API_KEY"))
    return has_pypirc or has_env


def pypi_readiness_status(upload: bool) -> tuple[bool, str]:
    """Read-only probe for the PyPI toolchain + creds. `maturin`/`twine` must be on PATH
    (fatal); a missing credential source is only a concern when `--pypi-upload` is set,
    since the default release stops before the push and prints the finish commands."""
    missing = [t for t in ("maturin", "twine") if not shutil.which(t)]
    if missing:
        return False, f"{' and '.join(missing)} not on PATH"
    if not pypi_has_credentials():
        if upload:
            return False, "no ~/.pypirc / TWINE_* creds"
        return True, "maturin + twine (no creds; upload opt-in)"
    return True, "maturin + twine + creds"


def tree_branch_remote_status(allow_dirty: bool, remote: str) -> tuple[bool, str]:
    """Read-only probe mirroring `preflight`: clean working tree (unless --allow-dirty),
    on `main`, and the push remote exists. Returns `(ok, detail)`; a non-main branch or a
    dirty tree under --allow-dirty is reported but not a failure (preflight only warns)."""
    if git_try("rev-parse").returncode != 0:
        return False, "not a git repo"
    branch = git_try("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "?"
    dirty = bool(git_try("status", "--porcelain").stdout.strip())
    has_remote = git_try("remote", "get-url", remote).returncode == 0
    notes = [branch, remote if has_remote else f"{remote} (missing)"]
    if dirty and not allow_dirty:
        return False, f"dirty tree · {' · '.join(notes)}"
    if dirty:
        notes.append("dirty (allowed)")
    # Only the push remote is required; a non-main branch is reported in the detail but
    # only warns (preflight never blocks on it), so it does not enter this ok verdict.
    ok = has_remote
    return ok, " · ".join(notes)


def preflight(allow_dirty: bool) -> None:
    if (
        not (WORKSPACE / ".git").exists()
        and not sh(["git", "-C", str(WORKSPACE), "rev-parse"]).returncode == 0
    ):
        fail("not inside a git repository")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        log(f"WARNING: on branch {branch!r}, not main")
    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        fail("working tree is dirty; commit/stash first (or pass --allow-dirty)")


def crate_index_path(name: str) -> str:
    """The crates.io sparse-index path for `name` (its documented prefix layout)."""
    n = name.lower()
    if len(n) == 1:
        return f"1/{n}"
    if len(n) == 2:
        return f"2/{n}"
    if len(n) == 3:
        return f"3/{n[0]}/{n}"
    return f"{n[:2]}/{n[2:4]}/{n}"


def already_published(name: str, version: str) -> bool:
    """Whether `version` of `name` is a live (non-yanked) release on crates.io, per the
    sparse index. This lets a re-run skip a crate WITHOUT invoking `cargo publish`, which
    re-verifies by compiling the packaged crate — a step that can fail on an
    already-shipped release (e.g. against a stale target/package copy of a dependency)
    and wedge an otherwise-complete release. On any lookup error, return False so the
    normal publish path still runs (fail open, never falsely skip)."""
    url = f"https://index.crates.io/{crate_index_path(name)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = resp.read().decode()
    except Exception as e:  # noqa: BLE001 — any failure just means "don't skip"
        log(f"  (crates.io index check for {name} {version} failed: {e}; will let cargo decide)")
        return False
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("vers") == version and not rec.get("yanked", False):
            return True
    return False


def npm_published(version: str) -> bool:
    """Whether `version` of the npm package is on the registry, per `npm view @V version`.
    Fail open (return False) on any error so a real re-run still attempts the publish."""
    if not shutil.which("npm"):
        return False
    r = sh(["npm", "view", f"fxtranslate@{version}", "version"])
    return r.returncode == 0 and r.stdout.strip() == version


def _pypi_release_files(version: str) -> Optional[list[dict]]:
    """The PyPI JSON API's file list for `version`, or None if the version isn't present.
    Reads `https://pypi.org/pypi/fxtranslate/json`; a 404 (package or version absent) and
    any network error both mean "not published" from a probe's point of view."""
    try:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 — absent package, 404, or network error: not published
        return None
    return data.get("releases", {}).get(version)


def pypi_published(version: str) -> bool:
    """Whether `version` is present on PyPI (has at least one uploaded file — sdist or a
    wheel), per the JSON API. Fail open to "not published" on any error."""
    files = _pypi_release_files(version)
    return bool(files)


def validate_packaging(order: list[Crate], version: str) -> list[str]:
    """Dry-run packaging for each publishable crate (catches missing files, bad
    metadata). Uses the committed manifests — for a dependent crate this needs its
    workspace dependency already on crates.io, so a not-yet-published dep is reported,
    not treated as fatal. A crate whose `version` is already live on crates.io skips the
    verify-compile (it obviously packages, and re-verifying it can fail on a stale
    target/package copy of a dependency); its file list is still listed. Returns the
    names of crates whose `cargo publish --dry-run` did not pass cleanly, so the caller
    can flag them in the summary."""
    unclean: list[str] = []
    for c in order:
        log(f"cargo package --list -p {c.name}")
        r = sh(["cargo", "package", "--list", "-p", c.name, "--manifest-path", str(ROOT_MANIFEST)])
        if r.returncode != 0:
            log(f"  (package --list reported: {r.stderr.strip().splitlines()[-1:] })")
        else:
            log(f"  {len(r.stdout.splitlines())} files would be packaged")
        if already_published(c.name, version):
            log(f"  {c.name} {version} already on crates.io; skipping publish verify")
            continue
        log(f"cargo publish --dry-run -p {c.name}")
        # `--allow-dirty`: cleanliness is the real run's concern (preflight); a
        # dry-run must validate packaging regardless of unrelated working changes.
        r = sh(
            [
                "cargo",
                "publish",
                "--dry-run",
                "--allow-dirty",
                "-p",
                c.name,
                "--manifest-path",
                str(ROOT_MANIFEST),
            ]
        )
        if r.returncode != 0:
            tail = "\n".join(r.stderr.strip().splitlines()[-4:])
            log(f"  dry-run did not pass (fine if it needs a not-yet-published dep):\n{tail}")
            unclean.append(c.name)
        else:
            log("  dry-run OK")
    return unclean


def publish_crate(crate: str, version: str) -> None:
    """Publish one crate. Skip cleanly if `version` is already live on crates.io — both
    the belt-and-suspenders 'already uploaded' stderr check and an up-front index lookup,
    so a recovery re-run never re-verifies (and can't fail on) a crate that already
    shipped."""
    if already_published(crate, version):
        log(f"{crate} {version} is already on crates.io; skipping")
        return
    log(f"cargo publish -p {crate}")
    # --allow-dirty packages the staged CHANGELOG.md copy, an intentional untracked
    # file (see stage_changelog); the version bump is already committed.
    r = subprocess.run(
        ["cargo", "publish", "--allow-dirty", "-p", crate, "--manifest-path", str(ROOT_MANIFEST)],
        text=True,
        capture_output=True,
    )
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode == 0:
        return
    if "already uploaded" in r.stderr or "already exists" in r.stderr:
        log(f"  {crate} is already on crates.io at this version; skipping")
        return
    fail(
        f"publishing {crate} failed (see above); crates already published stay up — "
        f"fix and re-run to publish the rest, then the tag is created"
    )


def validate_npm_packaging() -> None:
    """`npm publish --dry-run` from the npm package: runs the `prepublishOnly` hook
    (rebuild wasm + typecheck + parity) and reports the exact tarball that would ship,
    without touching the registry."""
    if not shutil.which("npm"):
        fail("npm not found on PATH; needed to validate the npm package")
    log("npm publish --dry-run (runs prepublishOnly: build:wasm + typecheck + parity)")
    r = subprocess.run(
        ["npm", "publish", "--dry-run"], cwd=NPM_DIR, text=True, capture_output=True
    )
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        fail("npm publish --dry-run failed; fix packaging before releasing")
    log("  npm dry-run OK")


def publish_npm() -> None:
    """Publish the npm package. `npm publish` runs `prepublishOnly` first (rebuild wasm
    + typecheck + parity), so the tarball is always built from a fresh, verified wasm
    core. Treat 'cannot publish over previously published version' as success so a re-run
    after a partial failure is safe (mirrors publish_crate)."""
    if not shutil.which("npm"):
        fail("npm not found on PATH; install Node/npm or publish the npm package manually")
    log("npm publish (in npm/)")
    r = subprocess.run(["npm", "publish"], cwd=NPM_DIR, text=True, capture_output=True)
    sys.stderr.write(r.stderr or "")
    sys.stdout.write(r.stdout or "")
    if r.returncode == 0:
        return
    combined = r.stderr + r.stdout
    if "previously published" in combined or "cannot publish over" in combined:
        log("  npm version is already on the registry; skipping")
        return
    hint = ""
    if "E401" in combined or "ENEEDAUTH" in combined or "E404" in combined:
        # npm returns a bare 404 on a logged-out first publish; the npm-auth step should
        # have caught this earlier, but say so plainly if it slips through.
        hint = " — this looks like an auth failure; run `npm login` and re-run to resume"
    fail(
        f"npm publish failed (see above){hint}. Crates already on crates.io stay up; "
        "re-run `task rs:publish` (resumes at the committed version, then tags)"
    )


def validate_pypi_packaging() -> None:
    """Build the PyPI sdist + local-platform wheel and validate their metadata, without
    touching the registry — the PyPI analogue of validate_npm_packaging. `maturin build
    --sdist` produces both the source distribution (installs everywhere, compiling from
    source) and one binary wheel for this machine's platform (the first release ships
    exactly these; CI backfills the rest of the wheel matrix on the tag). `twine check`
    then validates the packaged metadata and README rendering. Reports the artifact
    filenames that would ship, mirroring how the npm leg reports its tarball; fails on
    error, matching validate_npm_packaging."""
    if not shutil.which("maturin") or not shutil.which("twine"):
        fail("maturin/twine not found on PATH; needed to validate the PyPI package")
    log("maturin build --sdist (builds the sdist + this platform's wheel into dist/)")
    r = subprocess.run(
        ["maturin", "build", "--sdist", "-m", str(PY_MANIFEST), "-o", str(PY_DIST)],
        text=True,
        capture_output=True,
    )
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        fail("maturin build failed; fix the PyPI packaging before releasing")
    artifacts = sorted(PY_DIST.glob("*"))
    sdists = [a.name for a in artifacts if a.suffix == ".gz"]
    wheels = [a.name for a in artifacts if a.suffix == ".whl"]
    for name in sdists:
        log(f"  sdist:  {name}")
    for name in wheels:
        log(f"  wheel:  {name} (this platform only; CI backfills the rest on the tag)")
    log("twine check dist/*")
    r = subprocess.run(
        ["twine", "check", *[str(a) for a in artifacts]], text=True, capture_output=True
    )
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        fail("twine check failed; fix the PyPI packaging before releasing")
    log("  PyPI dry-run OK")


def publish_pypi(upload: bool) -> None:
    """Build the PyPI sdist + local wheel and, only when `upload` is set, push them to
    PyPI. Ordered AFTER crates.io and npm in the real release (all uploads before the
    tag). By default (`upload` False) this does the reversible half only — build + `twine
    check` — and prints the exact two commands to finish the PyPI leg by hand, because
    "the human handles the actual upload" with their own configured credentials while the
    crates.io + npm release stays fully automated. With `--pypi-upload` a fully-authorized
    operator does it in one shot: `twine upload --skip-existing`, which treats a version
    that's already on PyPI as success (the pragmatic equivalent of publish_crate's index
    skip and publish_npm's 'previously published' handling), so a recovery re-run only
    uploads what didn't land."""
    if not shutil.which("maturin") or not shutil.which("twine"):
        fail("maturin/twine not found on PATH; install with `pipx install maturin twine`")
    log("maturin build --release --sdist (sdist + this platform's wheel into dist/)")
    r = subprocess.run(
        ["maturin", "build", "--release", "--sdist", "-m", str(PY_MANIFEST), "-o", str(PY_DIST)],
        text=True,
        capture_output=True,
    )
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        fail("maturin build failed; PyPI leg not attempted")
    artifacts = sorted(PY_DIST.glob("*"))
    log("twine check dist/*")
    r = subprocess.run(
        ["twine", "check", *[str(a) for a in artifacts]], text=True, capture_output=True
    )
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        fail("twine check failed; PyPI leg not attempted")

    if not upload:
        log("PyPI upload is opt-in (default off); the release did everything except the push.")
        log("To finish the PyPI leg, run (with your PyPI credentials configured):")
        log("    maturin build --release --sdist -m crates/fxtranslate-py/Cargo.toml -o dist")
        log("    twine upload dist/*")
        log("(or re-run this publisher with --pypi-upload to do it in one shot)")
        return

    log("twine upload --skip-existing dist/*")
    r = subprocess.run(
        ["twine", "upload", "--skip-existing", *[str(a) for a in artifacts]],
        text=True,
        capture_output=True,
    )
    sys.stderr.write(r.stderr or "")
    sys.stdout.write(r.stdout or "")
    if r.returncode == 0:
        return
    combined = r.stderr + r.stdout
    if "already exists" in combined or "File already exists" in combined:
        # --skip-existing normally exits 0 on a duplicate; belt-and-suspenders in case a
        # mirror reports it as an error, so a recovery re-run stays safe.
        log("  PyPI already has this version; treating as already published")
        return
    fail(
        "twine upload failed (see above). Crates and npm already up stay up; re-run "
        "`task rs:publish --pypi-upload` (uploads with --skip-existing, then tags)"
    )


def stage_changelog(order: list[Crate]) -> list[Path]:
    """Copy the one workspace CHANGELOG.md into each publishable crate directory so it
    ships in the `.crate` (cargo only packages files under a crate's own root, so a
    single root changelog wouldn't otherwise be included). The copies are deliberately
    left untracked — they're a publish-time artifact, not committed duplicates — which
    is why `publish_crate` packages with `--allow-dirty`. Returns the copied paths for
    `cleanup_changelog` to remove afterward."""
    if not CHANGELOG.is_file():
        log(f"WARNING: {CHANGELOG.name} not found at {CHANGELOG}; crates ship without a changelog")
        return []
    staged: list[Path] = []
    for c in order:
        dest = c.manifest.parent / CHANGELOG.name
        shutil.copyfile(CHANGELOG, dest)
        staged.append(dest)
        log(f"staged {CHANGELOG.name} into {dest.parent.relative_to(WORKSPACE)}")
    return staged


def cleanup_changelog(staged: list[Path]) -> None:
    """Remove the copies `stage_changelog` made, leaving one changelog in the workspace."""
    for dest in staged:
        dest.unlink(missing_ok=True)


# Probe statuses. DONE = the world already reflects this step (skip it). PENDING = it
# still needs to run. BLOCKED = a precondition is unmet (e.g. not logged into npm); the
# board renders it red before anything executes, but the execute pass still calls run()
# so the failure surfaces with the full log + hint.
DONE = "done"
PENDING = "pending"
BLOCKED = "blocked"

# A render-only state (never returned by a probe): in a dry-run, an irreversible step is
# not executed and is shown as a neutral preview row — no green ✓, no red x — with a
# "(dry-run)" note. See `runs_in_dry_run` on Step and `_glyph_for`/`_right_for`.
PREVIEW = "preview"


@dataclass
class Status:
    """The read-only result of a Step's probe(): a state plus a short detail string for
    the board's right column ("0.4.2 live", "not logged in", "pending")."""

    state: str
    detail: str = ""


class StepError(Exception):
    """A structured Step failure carrying the captured log and a healing hint, so the
    board can dump the log, print the hint + a "next" line, and stop (mirrors check.py's
    print_failures, plus the self-healing hint the design calls for)."""

    def __init__(self, message: str, log_text: str, hint: str):
        super().__init__(message)
        self.message = message
        self.log_text = log_text
        self.hint = hint


@dataclass
class Step:
    """One ordered unit of the release: a labelled row on the board with a read-only
    `probe` and a side-effecting `run`.

    `probe()` returns a Status by asking the world (registry/git reads only, no side
    effects) — this is what makes re-runs idempotent and the board a true dashboard.
    `run()` does the work; on failure it must raise StepError (carrying the captured log
    and this step's `hint`). Both are supplied as callables so this stays a thin wrapper
    over the existing publish functions rather than a rewrite of their logic.
    """

    label: str
    group: str
    probe_fn: Callable[[], Status]
    run_fn: Callable[[], None]
    hint: str = ""
    # Whether this step's run body is reversible enough to actually execute during a
    # dry-run. Only the two reversible steps (cargo build + test, packaging validation)
    # set this True; every other step is shown as a neutral PREVIEW row in a dry-run
    # (skipped, not executed), so a dry-run never touches a registry, file, or git ref.
    runs_in_dry_run: bool = False
    # A step whose probe is DONE has nothing to run; a step that always runs (build,
    # tests, packaging validation) reports PENDING from its probe unless the world says
    # otherwise. Populated by the scheduler.
    status: Optional[Status] = None
    result: Optional[dict] = None  # {"exit_code", "output", "duration_ms"} after execute
    running: bool = False
    failed: bool = False  # set when execute() raised, so the row renders red (x)
    preview: bool = False  # set for a dry-run irreversible step: render neutral, not run

    def probe(self) -> Status:
        try:
            self.status = self.probe_fn()
        except Exception as e:  # noqa: BLE001 — a probe must never crash the board
            self.status = Status(PENDING, f"probe error: {e}")
        return self.status

    def execute(self) -> dict:
        """Run this step with stdout/stderr captured. Returns a result dict compatible
        with the board rows; on failure raises StepError with the captured log + hint."""
        started = time.monotonic()
        capture = _Capture()
        try:
            with capture:
                self.run_fn()
        except Aborted as e:
            # The run helpers raise Aborted (was `die`) on a command failure; convert to
            # a StepError carrying everything captured plus this step's healing hint.
            output = capture.text() + f"\n[publish] error: {e}\n"
            raise StepError(str(e), output, self.hint) from e
        except StepError:
            raise
        except Exception as e:  # noqa: BLE001 — surface any unexpected failure with the log
            output = capture.text() + f"\n[publish] unexpected error: {e}\n"
            raise StepError(str(e), output, self.hint) from e
        duration_ms = (time.monotonic() - started) * 1000
        self.result = {"exit_code": 0, "output": capture.text(), "duration_ms": duration_ms}
        return self.result


class _Capture:
    """Redirect stdout+stderr into a buffer for the duration of a Step's run(), so the
    board can hide it and surface it only on failure (check.py captures its subprocess
    output the same way; here the work is in-process, so we redirect the streams)."""

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._saved: tuple = ()

    def write(self, s: str) -> int:
        self._buf.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def __enter__(self) -> "_Capture":
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = self  # type: ignore[assignment]
        sys.stderr = self  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout, sys.stderr = self._saved

    def text(self) -> str:
        return "".join(self._buf)


def always_pending(detail: str = "pending") -> Callable[[], Status]:
    """A probe for steps that always run (build, test, packaging validation): they have
    no cheap world-derived "already done" signal, so they report PENDING and let the
    execute pass do the work. (Dry-run runs exactly these reversible steps.)"""
    return lambda: Status(PENDING, detail)


def bool_probe(check: Callable[[], bool], done: str, pending: str) -> Callable[[], Status]:
    """Wrap a bool-returning world check into a DONE/PENDING probe with detail strings."""
    return lambda: Status(DONE, done) if check() else Status(PENDING, pending)


def gate_probe(check: Callable[[], tuple[bool, str]]) -> Callable[[], Status]:
    """A precondition gate (auth/tooling/tree): `(ok, detail)` -> DONE when satisfied,
    BLOCKED (red) otherwise. Gates have no "run" beyond re-verifying, so a satisfied gate
    reads as DONE and an unsatisfied one blocks the run with its detail + hint."""

    def probe() -> Status:
        ok, detail = check()
        return Status(DONE, detail) if ok else Status(BLOCKED, detail)

    return probe


def version_committed(crates: list[Crate], target: str) -> bool:
    """The "Version bumped & committed" probe: every manifest AND package.json is at
    `target`, and a `release: fxtranslate <target>` commit exists. Fresh manifests at the
    target with no commit read as PENDING (the bump hasn't been committed yet); once the
    release run commits, this flips DONE and a resume skips the bump step."""
    manifests_at_target = all(c.version == target for c in crates) and npm_version() == target
    if not manifests_at_target:
        return False
    subject = f"release: fxtranslate {target}"
    r = git_try("log", "--grep", f"^{re.escape(subject)}$", "--format=%H", "-n", "1")
    return r.returncode == 0 and bool(r.stdout.strip())


def local_tag_exists(tag: str) -> bool:
    r = git_try("tag", "--list", tag)
    return r.returncode == 0 and bool(r.stdout.strip())


def remote_tag_exists(remote: str, tag: str) -> bool:
    """Whether `tag` is present on `remote`, per `git ls-remote --tags`. Fail open to
    False (not pushed) on any error so the push step still runs."""
    r = git_try("ls-remote", "--tags", remote, f"refs/tags/{tag}")
    return r.returncode == 0 and bool(r.stdout.strip())


def build_steps(args, crates, order, old, new, target, bumping) -> list[Step]:
    """Assemble the ordered Step list for this run. Groups, in order: Preflight,
    Build & validate, Publish, Tag & push. The run bodies are the existing publish
    functions wrapped in closures; the probes are the world-derived checks above.

    Every step carries its real run body; the scheduler (execute_in_order) decides what a
    dry-run runs. Only the two reversible steps (cargo build + test, packaging validation)
    set `runs_in_dry_run=True`; in a dry-run the scheduler executes those for real and
    renders every other step as a neutral PREVIEW row (not run), so a dry-run touches no
    registry, file, or git ref while staying "as consistent as the real thing".

    Returns `(steps, cleanup, notes)`: the ordered Step list, a cleanup callable that
    removes any staged changelog copies (call it in a `finally`), and a `notes` dict the
    packaging step fills with `unclean` crate names for the dry-run verdict.
    """
    tag = f"{TAG_PREFIX}{target}"
    # Changelog copies are staged once for the packaging-validation and crates.io publish
    # steps and cleaned up at the end of the run (both live under the same `order`).
    staged: list[Path] = []
    # Packaging-validation notes surfaced to the dry-run verdict (crates whose cargo
    # dry-run didn't pass cleanly, usually a not-yet-published dependency).
    notes: dict[str, list[str]] = {"unclean": []}

    def stage() -> None:
        staged.extend(stage_changelog(order))

    steps: list[Step] = []

    # Preflight: the reversible gates checked before any irreversible upload.
    steps.append(
        Step(
            label="Clean tree · branch · remote",
            group="Preflight",
            probe_fn=gate_probe(lambda: tree_branch_remote_status(args.allow_dirty, args.remote)),
            run_fn=lambda: preflight(args.allow_dirty),
            hint="commit/stash changes (or pass --allow-dirty), and check the branch/remote",
        )
    )
    # crates.io auth: no clean whoami exists, and a token-file probe is only a proxy — the
    # user's decision is to fail at use with a hint, so there is NO preflight step here;
    # a real auth failure surfaces at the first `cargo publish` (see the publish steps).
    steps.append(
        Step(
            label="npm auth",
            group="Preflight",
            probe_fn=gate_probe(npm_auth_status),
            run_fn=check_npm_auth,
            hint="run `npm login`, confirm with `npm whoami`, then re-run `task rs:publish`",
        )
    )
    steps.append(
        Step(
            label="PyPI tooling + creds",
            group="Preflight",
            probe_fn=gate_probe(lambda: pypi_readiness_status(args.pypi_upload)),
            run_fn=check_pypi_publish_readiness,
            hint=(
                "install with `pipx install maturin twine`; for --pypi-upload also "
                "configure ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD/TWINE_API_KEY"
            ),
        )
    )

    # Build & validate: the reversible half. Ordering matters and mirrors the original
    # release sequence — rewrite manifests FIRST, then `cargo build` (which refreshes
    # Cargo.lock to the new versions), then test, then validate packaging, and only THEN
    # commit. Splitting the old "Version bumped & committed" step into a leading "Version
    # bumped" (rewrite only) and a trailing "Release committed" (git add + commit) is what
    # keeps the refreshed Cargo.lock inside the release commit: if the commit ran before
    # the build, it would capture a stale lock and the build would then dirty the tree.
    def do_bump() -> None:
        rewrite_versions(crates, old, new, apply=True)
        rewrite_npm_version(old, new, apply=True)
        log(f"bumped crate manifests + npm/package.json to {new}")

    steps.append(
        Step(
            label="Version bumped",
            group="Build & validate",
            # DONE when every manifest + package.json already sits at the target (a resume,
            # where the bump already ran), regardless of whether the commit exists yet;
            # otherwise PENDING and the run rewrites them. The commit is a later step.
            probe_fn=bool_probe(
                lambda: all(c.version == target for c in crates) and npm_version() == target,
                done=f"{target} in manifests",
                pending=(f"{old} → {new}" if bumping else "not bumped"),
            ),
            run_fn=do_bump,
            hint=f"the bump edits every manifest + package.json to {target}",
        )
    )

    def do_build() -> None:
        run(["cargo", "build", "--manifest-path", str(ROOT_MANIFEST)])
        if not args.skip_tests:
            run(["cargo", "test", "--manifest-path", str(ROOT_MANIFEST)])

    steps.append(
        Step(
            label="cargo build + test",
            group="Build & validate",
            # Always runs (unless --skip-tests skips the test half); no world-derived
            # "done", so PENDING. In dry-run it still builds+tests — reversible work. The
            # build refreshes Cargo.lock to the bumped versions BEFORE the commit step.
            probe_fn=always_pending("build" if args.skip_tests else "build + test"),
            run_fn=do_build,
            runs_in_dry_run=True,
            hint="fix the compile/test failure above, then re-run",
        )
    )

    def do_validate() -> None:
        # Stage the changelog so packaging validation (and the later crates.io publish)
        # see exactly what would ship. Cleaned up by the terminal cleanup step.
        stage()
        # Validate the committed manifests, which are at `target` after the bump step. A
        # crate whose cargo dry-run didn't pass cleanly (usually just a dependency not on
        # crates.io yet) is recorded, not fatal — it drives the dry-run "with notes" verdict.
        notes["unclean"] = validate_packaging(order, target)
        # Intentional: on EVERY real run this rebuilds the wasm core (npm dry-run) and
        # builds + twine-checks the sdist/wheel (maturin), before the first irreversible
        # crates.io upload. The original only validated npm/PyPI packaging on its dry-run
        # path; doing it here too costs a redundant wasm build + maturin build (the real
        # publishes rebuild again), but that is the point — a broken npm or PyPI package is
        # caught while everything is still reversible, never after a crate has shipped. Do
        # not "optimize" this away.
        validate_npm_packaging()
        validate_pypi_packaging()

    steps.append(
        Step(
            label="Packaging validated",
            group="Build & validate",
            probe_fn=always_pending("crates + npm + pypi"),
            run_fn=do_validate,
            runs_in_dry_run=True,
            hint="fix the packaging issue reported above (missing files/metadata), then re-run",
        )
    )

    # Release committed: LAST in Build & validate, after packaging validation and just
    # before the irreversible Publish group. It stages the bumped manifests + npm files +
    # the build-refreshed Cargo.lock and commits them, so the release commit's Cargo.lock
    # matches its Cargo.toml versions and the tree is clean afterward. On a resume the
    # probe reads DONE (the commit exists) and this is skipped.
    def do_commit() -> None:
        manifests = [str(c.manifest.relative_to(WORKSPACE)) for c in crates]
        npm_files = ["npm/package.json"]
        if (NPM_DIR / "package-lock.json").is_file():
            npm_files.append("npm/package-lock.json")
        run(["git", "-C", str(WORKSPACE), "add", *manifests, *npm_files, "Cargo.lock"])
        run(["git", "-C", str(WORKSPACE), "commit", "-m", f"release: fxtranslate {target}"])
        log(f"committed release bump for {target} (with refreshed Cargo.lock)")

    steps.append(
        Step(
            label="Release committed",
            group="Build & validate",
            probe_fn=bool_probe(
                lambda: version_committed(crates, target),
                done=f"{target} committed",
                pending="not committed",
            ),
            run_fn=do_commit,
            hint=f"stages the manifests + npm files + Cargo.lock and commits `release: fxtranslate {target}`",
        )
    )

    # Publish: the irreversible uploads. crates.io first (not atomic, not reversible),
    # dependencies before dependents.
    for c in order:
        steps.append(
            Step(
                label=f"crates.io  {c.name}",
                group="Publish",
                probe_fn=bool_probe(
                    lambda name=c.name: already_published(name, target),
                    done=f"{target} live",
                    pending="pending",
                ),
                run_fn=(lambda name=c.name: publish_crate(name, target)),
                hint=(
                    "cargo publish failed — if it's auth, run `cargo login` (crates.io has "
                    "no whoami probe, so it fails here); already-published crates stay up, "
                    "so re-run `task rs:publish` to publish the rest, then the tag"
                ),
            )
        )
    # ... then npm (also not reversible). Its prepublishOnly rebuilds the wasm core.
    steps.append(
        Step(
            label="npm        fxtranslate",
            group="Publish",
            probe_fn=bool_probe(
                lambda: npm_published(target), done=f"{target} live", pending="pending"
            ),
            run_fn=publish_npm,
            hint="if auth: run `npm login`; then re-run `task rs:publish` to resume",
        )
    )
    # ... then PyPI (also not reversible). Off by default: builds + twine-checks the
    # sdist + local wheel and prints the finish commands, unless --pypi-upload opts in.
    steps.append(
        Step(
            label="PyPI       fxtranslate",
            group="Publish",
            probe_fn=bool_probe(
                lambda: pypi_published(target),
                done=f"{target} live",
                pending="pending" if args.pypi_upload else "pending (upload opt-in)",
            ),
            run_fn=lambda: publish_pypi(args.pypi_upload),
            hint=(
                "configure ~/.pypirc or TWINE_* and re-run with --pypi-upload; or finish "
                "by hand with `maturin build --release --sdist ... && twine upload dist/*`"
            ),
        )
    )

    # Tag & push: the atomic finish, created only once every upload has landed.
    def do_tag() -> None:
        if local_tag_exists(tag):
            log(f"tag {tag} already exists; skipping")
            return
        run(["git", "-C", str(WORKSPACE), "tag", "-a", tag, "-m", f"fxtranslate {target}"])
        log(f"created tag {tag}")

    steps.append(
        Step(
            label=f"Tag {tag}",
            group="Tag & push",
            probe_fn=bool_probe(lambda: local_tag_exists(tag), done="exists", pending="pending"),
            run_fn=do_tag,
            hint="the tag is created only after every upload lands, so it never marks a half-release",
        )
    )

    def do_push() -> None:
        if args.no_push:
            log(
                f"--no-push: skipping push. Push manually: git push {args.remote} HEAD && "
                f"git push {args.remote} {tag}"
            )
            return
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        run(["git", "-C", str(WORKSPACE), "push", args.remote, branch])
        run(["git", "-C", str(WORKSPACE), "push", args.remote, tag])
        log(f"pushed {branch} and {tag} to {args.remote}")

    def push_probe() -> Status:
        if args.no_push:
            return Status(DONE, "skipped (--no-push)")
        if remote_tag_exists(args.remote, tag):
            return Status(DONE, f"on {args.remote}")
        return Status(PENDING, f"push to {args.remote}")

    steps.append(
        Step(
            label=f"Push to {args.remote}",
            group="Tag & push",
            probe_fn=push_probe,
            run_fn=do_push,
            hint=f"check your push access to `{args.remote}`, then re-run to push branch + tag",
        )
    )

    return steps, (lambda: cleanup_changelog(staged)), notes


GROUP_ORDER = ["Preflight", "Build & validate", "Publish", "Tag & push"]


def probe_all(steps: list[Step]) -> None:
    """Run every step's probe() concurrently (probes are read-only registry/git reads, so
    fanning them out — like check.py's light checks — is safe) and store the Status on
    each step for the board."""
    threads = [threading.Thread(target=s.probe) for s in steps]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _glyph_for(step: Step) -> str:
    """The board glyph for a step from its probe status / execution state."""
    if step.running:
        return color("•", "dim")  # • running
    if step.preview:
        return color("·", "dim")  # neutral dry-run preview (not executed)
    if step.failed:
        return color("x", "red")
    if step.result is not None:
        return status_glyph({"exit_code": step.result["exit_code"]})
    st = step.status
    if st is None:
        return color("·", "dim")
    if st.state == DONE:
        return color("✓", "green")
    if st.state == BLOCKED:
        return color("x", "red")
    return color("·", "dim")  # pending


def _right_for(step: Step) -> str:
    if step.running:
        return color("running...", "dim")
    if step.preview:
        detail = step.status.detail if step.status else ""
        return color(f"{detail} (dry-run)".strip(), "dim")
    if step.result is not None:
        return f"{step.status.detail} {format_duration(step.result['duration_ms'])}".strip()
    return step.status.detail if step.status else ""


def _label_width(steps: list[Step]) -> int:
    return max(len(s.label) for s in steps)


def board_snapshot(steps: list[Step], header: str, header_right: str) -> tuple:
    """Build the grouped board (a leading title row + one Group per section) for the
    shared renderer."""
    label_width = _label_width(steps)
    duration_col = 5 + label_width
    groups: list[Group] = [Group(title=header, rows=[], right=header_right)]
    for name in GROUP_ORDER:
        members = [s for s in steps if s.group == name]
        if not members:
            continue
        rows = [Row(glyph=_glyph_for(s), label=s.label, right=_right_for(s)) for s in members]
        groups.append(Group(title=name, rows=rows))
    return groups, label_width, duration_col


def render_static(steps: list[Step], header: str, header_right: str) -> None:
    """Non-interactive board: print the plan/status as plain lines (no ANSI, no redraw),
    mirroring check.py's non-TTY fallback."""
    print(header + ("  " + header_right if header_right else ""))
    for name in GROUP_ORDER:
        members = [s for s in steps if s.group == name]
        if not members:
            continue
        print(f"\n{name}")
        for s in members:
            st = s.status
            if s.preview:
                tag = "PREV"
            elif s.failed:
                tag = "FAIL"
            elif s.result is not None:
                tag = "PASS" if s.result["exit_code"] == 0 else "FAIL"
            elif st and st.state == DONE:
                tag = "DONE"
            elif st and st.state == BLOCKED:
                tag = "BLOCK"
            else:
                tag = "TODO"
            detail = f"  {st.detail}" if st and st.detail else ""
            note = "  (dry-run)" if s.preview else ""
            print(f"  {tag:5} {s.label}{detail}{note}")


def print_step_failure(step: Step, err: StepError, dry_run: bool) -> None:
    """Dump a failed step's captured log plus its healing hint and a "next" line, in
    check.py's print_failures visual language (a red header, the hidden output, then the
    self-healing guidance the design calls for)."""
    resume = "task rs:publish -- --dry-run" if dry_run else "task rs:publish"
    if IS_INTERACTIVE:
        print()
        print(styled("✖ Failures", "bold", "red"))
        print(styled("──────────", "red"))
        print()
        print(
            f"{styled('┌─', 'red')} {styled(step.label, 'bold', 'red')} failed  "
            f"{styled(err.message, 'red')}"
        )
        print(styled("│  output", "red"))
    else:
        print("\nFAILURES\n")
        print(f"FAIL {step.label} | {err.message}")
        print("output:")
    sys.stdout.write(err.log_text or "(no output)\n")
    if err.log_text and not err.log_text.endswith("\n"):
        sys.stdout.write("\n")
    if step.hint:
        marker = styled("│", "red") if IS_INTERACTIVE else "|"
        print(f"{marker} {styled('hint:', 'dim')} {step.hint}")
    nxt = styled("└─ next:", "dim") if IS_INTERACTIVE else "next:"
    print(f"{nxt} fix the above, then re-run `{resume}` (green rows are skipped)")


def execute_in_order(
    steps: list[Step], header: str, header_right: Callable[[], str], dry_run: bool
) -> int:
    """The execute pass: walk steps IN ORDER (the chain is dependent — CLI pins engine,
    npm/PyPI after crates, tag last), skip DONE, run PENDING/BLOCKED with stdout hidden
    and a live •/✓/x, and on the FIRST failure dump the log + hint + next line and stop.
    Returns a process exit code. Uploads stay strictly serialized (this loop never runs
    two steps at once). Non-TTY degrades to plain RUN/PASS/FAIL lines like check.py."""
    if IS_INTERACTIVE:
        renderer = InteractiveRenderer(lambda: board_snapshot(steps, header, header_right()))
        renderer.render()
    else:
        renderer = None
        render_static(steps, header, header_right())

    for step in steps:
        if step.status and step.status.state == DONE:
            continue
        # In a dry-run, only the reversible steps actually execute; every other step is
        # shown as a neutral PREVIEW row (not run), so no registry/file/git ref is touched
        # and a blocked gate never renders red (a dry-run needs no auth).
        if dry_run and not step.runs_in_dry_run:
            step.preview = True
            if renderer:
                renderer.render()
            else:
                st = step.status
                detail = f"  {st.detail}" if st and st.detail else ""
                print(f"PREV {step.label}{detail}  (dry-run)", flush=True)
            continue
        step.running = True
        if renderer:
            renderer.render()
        else:
            print(f"\nRUN  {step.label}", flush=True)
        try:
            step.execute()
        except StepError as err:
            step.running = False
            step.failed = True
            if renderer:
                renderer.render()
                print()
            else:
                print(f"FAIL {step.label}", flush=True)
            print_step_failure(step, err, dry_run)
            return 1
        step.running = False
        if renderer:
            renderer.render()
        else:
            print(f"PASS {step.label}", flush=True)

    if renderer:
        print()
    return 0


def crates_fully_published(order: list[Crate], version: str) -> bool:
    """Whether every publishable crate's `version` is live on crates.io. Per the design,
    crates.io is the gate for "fully published" (the released-vs-in-flight distinction
    that drives resume): the crates are the fully-automated, irreversible core, while the
    PyPI upload is opt-in and npm is derivable from them."""
    return all(already_published(c.name, version) for c in order)


def resolve_release(args, crates, order, dry_run: bool = False) -> tuple[str, str, str, bool]:
    """Resolve what this invocation should do, deriving the target version from the
    manifests (never a state file) and enforcing the resume rules:

      - manifests fully published on crates.io -> clean released state. A bump arg (or
        --set) starts a NEW release; a bare run has nothing to do.
      - manifests NOT fully published -> a release for V is in flight. A bare run (or
        --resume) resumes at V; a bump arg is REFUSED (the double-bump guard).

    Returns `(old, new, target, bumping)` where `target` is the version the board
    operates on and `bumping` is whether a bump+commit still needs to happen.

    `dry_run` relaxes the guards for previewing: a dry-run never publishes, so it always
    has something to show (validate the current or the bumped packaging). A bare
    `--dry-run` previews the current manifest version regardless of release state (like
    `--status` with a validate pass); a `patch --dry-run` previews the bumped version.
    """
    old = current_version(crates)
    wants_bump = bool(args.level) or bool(args.set_version)
    released = crates_fully_published(order, old)

    if wants_bump and not released and not dry_run:
        die(
            f"a release for {old} is in flight (not fully published on crates.io); re-run "
            f"without a level to resume it, or --set to override the target version"
        )

    if not wants_bump:
        # Bare run / --resume.
        if released and not dry_run:
            die(
                f"{old} is fully published; nothing to resume. Specify patch|minor|major "
                f"to start a new release (or --dry-run to preview)"
            )
        # Resume (or, in dry-run, preview) at the manifest version, unchanged.
        return old, old, old, False

    # A bump/--set: derive the new version (strictly greater; refuse a no-op).
    new = args.set_version if args.set_version else bump(old, args.level)
    if args.set_version and not SEMVER.match(new):
        die(f"--set {new!r} is not a plain MAJOR.MINOR.PATCH")
    if new == old:
        die(f"new version {new} equals the current version")
    return old, new, new, True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "level", nargs="?", choices=["major", "minor", "patch"], help="semver component to bump"
    )
    ap.add_argument(
        "--set",
        dest="set_version",
        metavar="X.Y.Z",
        help="set an explicit version instead of bumping",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume the in-flight release at the manifest version (the default for a bare "
        "run; replaces the old --initial)",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="probe pass only: render the board showing what's done vs pending, then exit",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="render the same board but run only the reversible validate steps; touch no "
        "registry, file, or git ref",
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty working tree")
    ap.add_argument(
        "--skip-tests", action="store_true", help="skip `cargo test` before publishing"
    )
    # Default to the `gregtatum` fork, not `origin` — in this repo `origin` is the
    # upstream mozilla/translations, and a release push belongs on the fork.
    ap.add_argument(
        "--remote", default="gregtatum", help="git remote to push to (default: gregtatum)"
    )
    ap.add_argument("--no-push", action="store_true", help="commit + tag locally but don't push")
    ap.add_argument(
        "--pypi-upload",
        action="store_true",
        help="also upload the PyPI package (default: off — the release does everything "
        "except the PyPI push and prints the finish commands; needs PyPI credentials)",
    )
    args = ap.parse_args()

    if args.level and args.set_version:
        die("give a bump level (major|minor|patch) OR --set X.Y.Z, not both")
    if args.resume and (args.level or args.set_version):
        die("--resume takes no bump level or --set (it resumes the manifest version)")

    crates = load_workspace()
    order = publish_order(crates)

    # --status is a pure dashboard: resolve the target leniently (so it can show the board
    # even in a clean released state, where resolve_release would otherwise refuse), probe
    # everything, render, and exit — no execute pass at all.
    if args.status:
        target = current_version(crates)
        released = crates_fully_published(order, target)
        old, new, bumping = target, target, False
        header = f"◆ fxtranslate {target}"
        header_right = "released" if released else "in-flight release"
        steps, cleanup, _notes = build_steps(args, crates, order, old, new, target, bumping)
        try:
            probe_all(steps)
            if IS_INTERACTIVE:
                renderer = InteractiveRenderer(lambda: board_snapshot(steps, header, header_right))
                renderer.render()
                print()
            else:
                render_static(steps, header, header_right)
        finally:
            cleanup()
        return

    old, new, target, bumping = resolve_release(args, crates, order, dry_run=args.dry_run)

    readme_report(crates, old)

    if bumping:
        edits = rewrite_versions(crates, old, new, apply=False)
        edits.extend(rewrite_npm_version(old, new, apply=False))
        log(f"planned version edits ({old} -> {new}):")
        for e in edits:
            print(f"    {e}", file=sys.stderr)

    dry = args.dry_run
    if dry:
        header = f"◆ fxtranslate {target}"
        header_right = "dry-run (validates only; touches nothing)"
    else:
        header = f"◆ fxtranslate {target}"
        header_right = "starting release" if bumping else "resuming in-flight release"

    steps, cleanup, notes = build_steps(args, crates, order, old, new, target, bumping)

    # Pass 1: probe everything (read-only) and render the current world.
    probe_all(steps)

    started = time.monotonic()

    def elapsed_right() -> str:
        return f"{header_right}   {format_duration((time.monotonic() - started) * 1000)}"

    try:
        code = execute_in_order(steps, header, elapsed_right, dry_run=dry)
    finally:
        cleanup()

    if code != 0:
        sys.exit(code)

    if dry:
        # Preserve the current dry-run's OK / OK-with-notes verdict, now derived from the
        # board: a crate whose cargo dry-run didn't pass cleanly (recorded by the packaging
        # step) is a "with notes" outcome — usually just a dependency not on crates.io yet.
        unclean = notes["unclean"]
        log("")
        if unclean:
            log(
                f"⚠  DRY RUN OK, WITH NOTES — the npm and PyPI packages validated; "
                f"crate(s) [{', '.join(unclean)}] had cargo dry-run notes above."
            )
            log("   Usually just a dependency not on crates.io yet; if they publish in-order")
            log("   this run is fine. Nothing was changed.")
        else:
            log("✓  DRY RUN PASSED — crates, npm, and PyPI packaging validated cleanly.")
            log("   Nothing was changed (no files, no crates.io, no npm, no PyPI, no git).")
        real = (
            f"--set {args.set_version}"
            if args.set_version
            else (args.level or "(bare, to resume)")
        )
        log(f"   To release {target}: task rs:publish -- {real}")
        return

    log(f"published fxtranslate {target} \U0001f389")


if __name__ == "__main__":
    main()
