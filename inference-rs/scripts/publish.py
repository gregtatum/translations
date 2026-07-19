#!/usr/bin/env python3
"""
Release the fxtranslate ecosystem as one unit: the Rust crates to crates.io and the
npm package to the npm registry, on a single shared version.

The three artifacts — the `fxtranslate` engine crate, the `fxtranslate-cli` crate, and
the `fxtranslate` npm package — are all the same engine (the CLI pins the engine
exactly; npm embeds it compiled to wasm). A version that shipped to one but not the
others, or at different numbers, would be meaningless, so this refuses to let them
drift: everything bumps together and publishes together, or the run stops.

The safety story is the reason this is a script and not a handful of commands. A
registry upload is effectively permanent (it can only be yanked/deprecated, never
replaced), and publishing several of them is not atomic. So the design is: do all the
reversible work first (bump, build, test, validate packaging), then the irreversible
uploads, and only once every upload has landed create the git tag — the one atomic step
— so the tag can never mark a half-published release. That also makes a re-run the
recovery path: an interrupted release is resumed by running the same command, because
already-published artifacts are detected and skipped.

`--dry-run` exercises the whole reversible half (including `cargo`/`npm` dry-run
packaging) and prints the plan, touching no registry, file, or git ref. Operator setup,
the checklist, and first-time-publish notes live in RELEASING.md; run with -h for flags.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent  # inference-rs/
ROOT_MANIFEST = WORKSPACE / "Cargo.toml"
NPM_DIR = WORKSPACE / "npm"  # the npm package (wasm core copied in by build:wasm)
NPM_MANIFEST = NPM_DIR / "package.json"
CHANGELOG = WORKSPACE / "CHANGELOG.md"  # one workspace changelog, copied into each crate
TAG_PREFIX = "fxtranslate-v"  # bare `0.1.0` etc. are taken by old repo tags

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def log(msg: str) -> None:
    print(f"[publish] {msg}", file=sys.stderr)


def die(msg: str) -> None:
    sys.exit(f"[publish] error: {msg}")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run capturing stdout+stderr as text (for inspection)."""
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def run(cmd: list[str], **kw) -> None:
    """Run streaming to the terminal; exit on failure."""
    log("$ " + " ".join(cmd))
    if subprocess.run(cmd, **kw).returncode != 0:
        die("command failed: " + " ".join(cmd))


# --- workspace model ---------------------------------------------------------


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


# --- versioning --------------------------------------------------------------


NPM_VERSION = re.compile(r'^(?P<pre>\s*"version":\s*)"(?P<ver>[^"]+)"', re.MULTILINE)


def npm_version() -> str:
    """The npm package's declared version (the top-level `"version"` in package.json)."""
    m = NPM_VERSION.search(NPM_MANIFEST.read_text())
    if not m:
        die(f"could not find a top-level \"version\" in {NPM_MANIFEST}")
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


# --- README hygiene ----------------------------------------------------------


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


# --- preflight ---------------------------------------------------------------


def git(*args: str) -> str:
    r = sh(["git", "-C", str(WORKSPACE), *args])
    if r.returncode != 0:
        die(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def preflight(allow_dirty: bool) -> None:
    if (
        not (WORKSPACE / ".git").exists()
        and not sh(["git", "-C", str(WORKSPACE), "rev-parse"]).returncode == 0
    ):
        die("not inside a git repository")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        log(f"WARNING: on branch {branch!r}, not main")
    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        die("working tree is dirty; commit/stash first (or pass --allow-dirty)")


# --- publish steps -----------------------------------------------------------


def validate_packaging(order: list[Crate]) -> list[str]:
    """Dry-run packaging for each publishable crate (catches missing files, bad
    metadata). Uses the committed manifests — for a dependent crate this needs its
    workspace dependency already on crates.io, so a not-yet-published dep is reported,
    not treated as fatal. Returns the names of crates whose `cargo publish --dry-run`
    did not pass cleanly, so the caller can flag them in the summary."""
    unclean: list[str] = []
    for c in order:
        log(f"cargo package --list -p {c.name}")
        r = sh(["cargo", "package", "--list", "-p", c.name, "--manifest-path", str(ROOT_MANIFEST)])
        if r.returncode != 0:
            log(f"  (package --list reported: {r.stderr.strip().splitlines()[-1:] })")
        else:
            log(f"  {len(r.stdout.splitlines())} files would be packaged")
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


def publish_crate(crate: str) -> None:
    """Publish one crate; treat 'already uploaded' as success so a re-run is safe."""
    log(f"cargo publish -p {crate}")
    # --allow-dirty packages the staged CHANGELOG.md copy, an intentional untracked
    # file (see stage_changelog); the version bump is already committed.
    r = subprocess.run(
        ["cargo", "publish", "--allow-dirty", "-p", crate, "--manifest-path", str(ROOT_MANIFEST)],
        text=True,
        capture_output=True,
    )
    sys.stderr.write(r.stderr)
    if r.returncode == 0:
        return
    if "already uploaded" in r.stderr or "already exists" in r.stderr:
        log(f"  {crate} is already on crates.io at this version; skipping")
        return
    die(
        f"publishing {crate} failed (see above); crates already published stay up — "
        f"fix and re-run to publish the rest, then the tag is created"
    )


def validate_npm_packaging() -> None:
    """`npm publish --dry-run` from the npm package: runs the `prepublishOnly` hook
    (rebuild wasm + typecheck + parity) and reports the exact tarball that would ship,
    without touching the registry."""
    if not shutil.which("npm"):
        log("WARNING: npm not found on PATH; skipping npm dry-run")
        return
    log("npm publish --dry-run (runs prepublishOnly: build:wasm + typecheck + parity)")
    r = subprocess.run(["npm", "publish", "--dry-run"], cwd=NPM_DIR, text=True)
    if r.returncode != 0:
        die("npm publish --dry-run failed; fix packaging before releasing")
    log("  npm dry-run OK")


def publish_npm() -> None:
    """Publish the npm package. `npm publish` runs `prepublishOnly` first (rebuild wasm
    + typecheck + parity), so the tarball is always built from a fresh, verified wasm
    core. Treat 'cannot publish over previously published version' as success so a re-run
    after a partial failure is safe (mirrors publish_crate)."""
    if not shutil.which("npm"):
        die("npm not found on PATH; install Node/npm or publish the npm package manually")
    log("npm publish (in npm/)")
    r = subprocess.run(["npm", "publish"], cwd=NPM_DIR, text=True, capture_output=True)
    sys.stderr.write(r.stderr)
    sys.stdout.write(r.stdout)
    if r.returncode == 0:
        return
    combined = r.stderr + r.stdout
    if "previously published" in combined or "cannot publish over" in combined:
        log("  npm version is already on the registry; skipping")
        return
    die(
        "npm publish failed (see above); crates already on crates.io stay up — "
        "fix and re-run to publish npm, then the tag is created"
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
        "--initial",
        action="store_true",
        help="first release: publish the manifest version as-is (no bump, no version commit)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + print the plan; touch nothing (no edits, publish, or git)",
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
    args = ap.parse_args()

    if sum([bool(args.level), bool(args.set_version), args.initial]) != 1:
        die("give exactly one of: a bump level (major|minor|patch), --set X.Y.Z, or --initial")

    crates = load_workspace()
    old = current_version(crates)
    # `--initial` publishes the manifest version untouched; the other modes derive a
    # strictly greater version and refuse a no-op bump.
    bumping = not args.initial
    if args.initial:
        new = old
    else:
        new = args.set_version if args.set_version else bump(old, args.level)
        if args.set_version and not SEMVER.match(new):
            die(f"--set {new!r} is not a plain MAJOR.MINOR.PATCH")
        if new == old:
            die(f"new version {new} equals the current version")
    order = publish_order(crates)
    tag = f"{TAG_PREFIX}{new}"

    log(f"initial release at {new}" if args.initial else f"workspace at {old} -> {new}")
    log("publishable crates (in order): " + ", ".join(c.name for c in order))
    guarded = [c.name for c in crates if not c.publishable]
    if guarded:
        log("guarded (publish = false): " + ", ".join(guarded))
    log(f"release tag: {tag}")

    # The version edits, previewed for everyone and applied only on a real run.
    # `--initial` changes no versions, so there's nothing to rewrite or preview.
    if bumping:
        edits = rewrite_versions(crates, old, new, apply=False)
        edits.extend(rewrite_npm_version(old, new, apply=False))
        log("version edits:")
        for e in edits:
            print(f"    {e}", file=sys.stderr)
    else:
        log("initial release: manifests already at the target version, no edits")

    readme_report(crates, old)

    if args.dry_run:
        log("dry-run: validating current packaging (no crates/git touched)")
        # Stage the changelog so packaging validation sees exactly what would ship,
        # then remove it — the tree is left as it was found.
        staged = stage_changelog(order)
        try:
            unclean = validate_packaging(order)
        finally:
            cleanup_changelog(staged)
        validate_npm_packaging()  # dies on failure, so reaching here means npm is OK

        # A clear verdict, so the dry-run ends with an unambiguous go / look-first signal
        # rather than leaving the reader to infer it from the exit code.
        real_cmd = f"task rs:publish -- {args.set_version and f'--set {args.set_version}' or args.level or '--initial'}"
        steps = f"publish [{', '.join(c.name for c in order)}] to crates.io, publish fxtranslate to npm, then tag {tag} and push to {args.remote}"
        plan = steps if args.initial else f"bump everything to {new}, commit, {steps}"
        log("")
        if unclean:
            log(f"⚠  DRY RUN OK, WITH NOTES — the npm package validated; crate(s) [{', '.join(unclean)}] had")
            log("   cargo dry-run notes above (usually just a dependency not on crates.io yet). Skim them,")
            log(f"   but if the deps are expected to publish in-order this run is fine. Nothing was changed.")
        else:
            log("✓  DRY RUN PASSED — every crate and the npm package validated cleanly. Nothing was changed")
            log("   (no files, no crates.io, no npm, no git).")
        log(f"   To release {new}: {real_cmd}")
        log(f"   That will: {plan}.")
        return

    # --- real release ---
    preflight(args.allow_dirty)
    if bumping:
        rewrite_versions(crates, old, new, apply=True)
        rewrite_npm_version(old, new, apply=True)
        log(f"bumped crate manifests + npm/package.json to {new}")
    else:
        log(f"initial release at {new}; manifests unchanged")

    # Build (refreshes Cargo.lock with the new versions) and test.
    run(["cargo", "build", "--manifest-path", str(ROOT_MANIFEST)])
    if not args.skip_tests:
        run(["cargo", "test", "--manifest-path", str(ROOT_MANIFEST)])

    # Bundle the changelog into each crate for packaging and publishing, then remove
    # the copies once every crate is up (or on failure). The copies stay untracked, so
    # the release commit below — which adds only manifests + Cargo.lock — never picks
    # them up, and the tag points at a tree with a single workspace changelog.
    staged = stage_changelog(order)
    try:
        validate_packaging(order)

        # Commit before publishing, so the published crates correspond to a committed
        # state (and the tag we create points at exactly what shipped). On `--initial`
        # there's no version bump to commit; commit only a lockfile refresh if the build
        # produced one, so the tag still lands on a clean tree.
        if bumping:
            manifests = [str(c.manifest.relative_to(WORKSPACE)) for c in crates]
            npm_files = ["npm/package.json"]
            if (NPM_DIR / "package-lock.json").is_file():
                npm_files.append("npm/package-lock.json")
            git("add", *manifests, *npm_files, "Cargo.lock")
            git("commit", "-m", f"release: fxtranslate {new}")
            log(f"committed release bump for {new}")
        elif git("status", "--porcelain", "--", "Cargo.lock"):
            git("add", "Cargo.lock")
            git("commit", "-m", f"release: fxtranslate {new} (lockfile refresh)")
            log("committed Cargo.lock refresh")

        # crates.io first (not atomic, not reversible) ...
        for c in order:
            publish_crate(c.name)
        # ... then npm (also not reversible). Its prepublishOnly rebuilds the wasm core.
        publish_npm()
    finally:
        cleanup_changelog(staged)

    # ... tag last (atomic), only now that every crate AND npm are up.
    git("tag", "-a", tag, "-m", f"fxtranslate {new}")
    log(f"created tag {tag}")

    if args.no_push:
        log(
            f"--no-push: skipping push. Push manually: git push {args.remote} HEAD && "
            f"git push {args.remote} {tag}"
        )
    else:
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        git("push", args.remote, branch)
        git("push", args.remote, tag)
        log(f"pushed {branch} and {tag} to {args.remote}")

    log(f"published fxtranslate {new} 🎉")


if __name__ == "__main__":
    main()
