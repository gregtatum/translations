# Releasing fxtranslate

The ecosystem ships on **one shared version**, in lockstep:

- `fxtranslate` — the engine — on [crates.io](https://crates.io/crates/fxtranslate)
- `fxtranslate-cli` — the native CLI — on [crates.io](https://crates.io/crates/fxtranslate-cli)
- `fxtranslate` — the Node CLI + library — on [npm](https://www.npmjs.com/package/fxtranslate)
- `fxtranslate` — the Python library — on [PyPI](https://pypi.org/project/fxtranslate/)

One command drives all four artifacts: [`scripts/publish.py`](./scripts/publish.py), wrapped as `task rs:publish`. It bumps every crate manifest and `npm/package.json` to the same version (the PyPI package's version is read from its crate `Cargo.toml`, so it moves for free), builds and tests, publishes the crates in dependency order, publishes the npm package (rebuilding its wasm core first), handles the PyPI leg (builds the sdist + local wheel, and uploads it with `--pypi-upload`), and creates + pushes a single `fxtranslate-vX.Y.Z` tag once everything is up.

## Prerequisites

- **crates.io auth** — `cargo login` (a crates.io API token).
- **npm auth** — `npm login`, with publish rights to `fxtranslate` (`npm whoami` confirms the account).
- **wasm toolchain** — `wasm-pack` and the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`); the npm publish rebuilds the wasm core.
- **PyPI toolchain** — `maturin` and `twine` (`pipx install maturin twine`); the PyPI leg builds the sdist + local wheel with maturin and validates/uploads with twine.
- **PyPI auth** — a PyPI API token in `~/.pypirc` or the `TWINE_*` env vars (`TWINE_USERNAME`/`TWINE_PASSWORD`/`TWINE_API_KEY`). That covers the operator-run `task rs:publish` upload; the CI wheel workflow uploads via Trusted Publishing (OIDC) instead, whose one-time setup is documented under **Binary wheels (CI)** below.
- **A clean tree on `main`, pushed to the `gregtatum` remote.** `origin` is upstream mozilla/translations; releases go to the fork, which is the publisher's default remote.

## Preview, then publish

Everything stays read-only until `--dry-run` is dropped.

```console
# Preview: validate crate packaging, the npm tarball, and the PyPI sdist + wheel; touch nothing.
$ task rs:publish -- patch --dry-run       # or: minor / major / --set X.Y.Z

# Release: bump → build+test → publish crates → publish npm → PyPI leg → tag → push.
$ task rs:publish -- patch

# Release, including the PyPI upload in one shot (needs PyPI credentials configured):
$ task rs:publish -- patch --pypi-upload
```

The bump level (`patch` | `minor` | `major`) or `--set X.Y.Z` sets the next shared version. The dry-run validates the crates with `cargo publish --dry-run`, the npm package with `npm publish --dry-run` (the latter runs the `prepublishOnly` hook — rebuild wasm + typecheck + parity — so it is a faithful preview and correspondingly slow), and the PyPI package by building the sdist + local wheel with `maturin build --sdist` and validating them with `twine check`.

**The PyPI upload is gated.** By default `task rs:publish` does everything *except* the PyPI push: it builds the sdist + local wheel, runs `twine check`, and prints the exact two commands to finish the leg by hand —

```console
$ maturin build --release --sdist -m crates/fxtranslate-py/Cargo.toml -o dist
$ twine upload dist/*
```

— so the crates.io + npm release stays fully automated while the actual PyPI push uses the human's configured credentials. Pass `--pypi-upload` to have the publisher do the upload too (`twine upload --skip-existing`, so a re-run is safe).

Other flags: `--no-push` (commit + tag locally, push by hand), `--allow-dirty`, `--skip-tests`, `--remote <name>`.

### Checklist

- [ ] `git status` clean, on `main`, on the `gregtatum` remote.
- [ ] `cargo login` and `npm login` done (`npm whoami` confirms).
- [ ] `maturin` and `twine` installed (`pipx install maturin twine`), and PyPI auth configured (`~/.pypirc` / `TWINE_*`).
- [ ] `CHANGELOG.md` updated for the new version.
- [ ] `task rs:publish -- <level> --dry-run` — review the version edits, the crate file lists, the **npm tarball contents** (it must include `wasm/fxtranslate_wasm_bg.wasm`), and the **PyPI artifacts** (the sdist + one local wheel, both passing `twine check`).
- [ ] `task rs:publish -- <level>` — real release (add `--pypi-upload` to include the PyPI push, or finish it by hand with the printed commands).
- [ ] Confirm: `cargo info fxtranslate` / `fxtranslate-cli`, `npm view fxtranslate version`, and `pip index versions fxtranslate` (or the [PyPI page](https://pypi.org/project/fxtranslate/)) show the new version, and the `fxtranslate-vX.Y.Z` tag is on the fork.

## Ordering and re-runs

Publishing to crates.io, then npm, then PyPI is **not atomic and not reversible** — an upload can only be yanked or deprecated. So the publisher checks auth/tooling, uploads every registry first (crates.io → npm → PyPI), and creates the git tag **last**, once everything is up; the tag therefore never points at a half-published release. (With the default no-`--pypi-upload` run, the PyPI push is deferred to the human; the tag is still created after the crates + npm uploads, and the printed commands complete the PyPI leg.)

A run interrupted *after* the version bump was committed (e.g. crates published but npm or PyPI failed) is completed with **`--initial`**, not a fresh bump:

```console
$ task rs:publish -- --initial      # publishes the already-committed version; do NOT re-run `patch`
```

`--initial` targets the version already in the manifests instead of bumping again (a re-run of `patch` would move 0.4.1 → 0.4.2 and publish a further version). Crates already on crates.io and an npm version already on the registry are detected and skipped; the PyPI leg re-runs with `twine upload --skip-existing` (add `--pypi-upload`), so a version already on PyPI is left alone — only the uploads that didn't land, plus the tag and push, happen.

## Troubleshooting

- **`npm publish` fails with `404 Not Found - PUT .../fxtranslate`** — you're not logged in. npm reports a logged-out *first* publish as a 404 (it won't reveal a package you can't see). Run `npm login`, confirm with `npm whoami`, then complete the release with `--initial`. The publisher now checks `npm whoami` up front, so a logged-out release stops before touching crates.io.
- **A crate's `cargo publish --dry-run` reports "failed to select a version for ... `=X.Y.Z`"** — expected during a bump: `fxtranslate-cli` pins the not-yet-published engine version. It resolves once the engine uploads (engine publishes first). The dry-run flags it as a note, not a failure.

## First-time publishing

A package that has never been released is published by its first lockstep run like any other — its version starts wherever the ecosystem's shared version is at that release, with no back-fill of intermediate versions. The crates used `publish.py --initial` for their very first crates.io upload (publish the manifest version without bumping); npm needs no equivalent, since a first `npm publish` simply claims the name. The npm package is public and unscoped, so no `--access public` is required unless the name is later moved under a scope.

PyPI is the same: a first `twine upload` (via `--pypi-upload`, or the printed finish commands) **claims the PyPI name**. Note the first real release ships **an sdist + the release machine's local-platform wheel only** — everyone can `pip install` from the sdist (compiling from source), and the release machine's platform also gets a binary wheel; the full binary-wheel matrix for other platforms is backfilled by CI on the tag (see below).

## Binary wheels (CI)

`task rs:publish` ships the sdist plus the release machine's own platform wheel, then creates the `fxtranslate-vX.Y.Z` tag last, once every registry is up. Pushing that tag triggers the `.github/workflows/pypi-wheels.yml` workflow, which builds and uploads the full per-platform binary wheel matrix — Linux `x86_64` + `aarch64`, macOS `x86_64` + `arm64`, and Windows `x86_64` — so most users get a fast SIMD wheel with no toolchain instead of compiling from the sdist. The crate is `abi3-py38`, so it is one wheel per platform (every CPython ≥ 3.8), not one per Python minor. The workflow also rebuilds the sdist for single provenance, uploads everything with `--skip-existing` (idempotent — it coexists with the operator's local sdist + wheel upload for the same version), and asserts every wheel is `abi3` before publishing. It can also be run manually from the Actions tab (`workflow_dispatch`) to test the matrix without cutting a release.

**One-time PyPI setup (do this once, by hand — it cannot be scripted).** The workflow uploads via **Trusted Publishing (OIDC)**, so it carries no long-lived token. On [pypi.org](https://pypi.org/manage/project/fxtranslate/settings/publishing/), add a **Trusted Publisher** for the `fxtranslate` project pointing at:

- **Owner / repository**: `gregtatum/translations`
- **Workflow filename**: `pypi-wheels.yml`
- **Environment**: `pypi`

Until that is configured, the OIDC upload will fail; the workflow documents a commented `PYPI_API_TOKEN` fallback (a repo secret) for environments without trusted publishing. Optionally add a GitHub Actions environment named `pypi` with required reviewers for a manual approval gate before the upload.

After a release, confirm the wheels land: check the [PyPI page](https://pypi.org/project/fxtranslate/) (or `pip index versions fxtranslate`) shows the new version with wheels for each platform, and glance at the workflow run under the repo's Actions tab.

## Manual npm publish

`task rs:publish` is the supported path, so the versions stay locked together. Publishing the npm package alone is a fallback only (re-releasing at an existing version is impossible on either registry — bump instead):

```console
$ cd npm
$ npm publish --dry-run     # prepublishOnly rebuilds wasm + typechecks + runs parity
$ npm publish
```

The `prepublishOnly` hook (`build:wasm && typecheck && check:parity`) runs on both, so even a manual publish cannot ship a stale or missing `wasm/`.
