# Releasing fxtranslate

The ecosystem ships on **one shared version**, in lockstep:

- `fxtranslate` — the engine — on [crates.io](https://crates.io/crates/fxtranslate)
- `fxtranslate-cli` — the native CLI — on [crates.io](https://crates.io/crates/fxtranslate-cli)
- `fxtranslate` — the Node CLI + library — on [npm](https://www.npmjs.com/package/fxtranslate)
- `fxtranslate` — the Python library — on [PyPI](https://pypi.org/project/fxtranslate/)

One command drives all four artifacts: [`scripts/publish.py`](./scripts/publish.py), wrapped as `task rs:publish`. It bumps every crate manifest and `npm/package.json` to the same version (the PyPI package's version is read from its crate `Cargo.toml`, so it moves for free), builds and tests, publishes the crates in dependency order, publishes the npm package (rebuilding its wasm core first), handles the PyPI leg (builds the sdist + local wheel, and uploads it with `--pypi-upload`), and creates + pushes a single `fxtranslate-vX.Y.Z` tag once everything is up.

It runs as a **resumable status board**, modeled on `task rs:check`: a two-pass dashboard that first *probes the world* (reads crates.io, npm, PyPI, and git — no side effects) to render what's already done vs pending, then *executes* the pending steps in order with their stdout hidden, surfacing the full captured log plus a per-step healing hint only when a step fails. There is no state file — every step re-derives its status from the registries — so the intended workflow on any failure is: fix the one red row and **re-run `task rs:publish`**; already-green steps are skipped by their probes and the run picks up where it stopped. Run it over and over until the board is all green.

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
# Status: probe every registry and render the board (what's done vs pending), then exit.
$ task rs:publish -- --status

# Preview: render the SAME board, run only the reversible validate steps; touch nothing.
$ task rs:publish -- patch --dry-run       # or: minor / major / --set X.Y.Z

# Release: bump → build+test → publish crates → publish npm → PyPI leg → tag → push.
$ task rs:publish -- patch

# Resume an in-flight release (a prior run failed part-way): re-run with no level.
$ task rs:publish                          # or: task rs:publish -- --resume

# Release, including the PyPI upload in one shot (needs PyPI credentials configured):
$ task rs:publish -- patch --pypi-upload
```

The bump level (`patch` | `minor` | `major`) or `--set X.Y.Z` sets the next shared version; a bare run (no level) **resumes** the version already in the manifests rather than bumping (see [Resume and re-runs](#resume-and-re-runs)). `--status` runs the probe pass only — the pure "where did the last run get to?" dashboard — and exits without executing anything.

`--dry-run` renders the same status board and its execute pass runs only the two reversible steps (`cargo build` + test, and packaging validation); every irreversible step is shown as a neutral `(dry-run)` preview row (not executed), so no registry, file, or git ref is touched. The validation covers the crates with `cargo publish --dry-run`, the npm package with `npm publish --dry-run` (the latter runs the `prepublishOnly` hook — rebuild wasm + typecheck + parity — so it is a faithful preview and correspondingly slow), and the PyPI package by building the sdist + local wheel with `maturin build --sdist` and validating them with `twine check`.

**The PyPI upload is gated.** By default `task rs:publish` does everything *except* the PyPI push: it builds the sdist + local wheel, runs `twine check`, and prints the exact two commands to finish the leg by hand —

```console
$ maturin build --release --sdist -m crates/fxtranslate-py/Cargo.toml -o dist
$ twine upload dist/*
```

— so the crates.io + npm release stays fully automated while the actual PyPI push uses the human's configured credentials. Pass `--pypi-upload` to have the publisher do the upload too (`twine upload --skip-existing`, so a re-run is safe).

Other flags: `--status` (probe pass only, then exit), `--resume` (explicitly resume the manifest version — the same as a bare run), `--watch-wheels` (after a successful push, poll PyPI until the 4-platform CI wheel matrix completes — see [Binary wheels (CI)](#binary-wheels-ci)), `--no-push` (commit + tag locally, push by hand), `--allow-dirty`, `--skip-tests`, `--remote <name>`.

### Checklist

- [ ] `git status` clean, on `main`, on the `gregtatum` remote.
- [ ] `cargo login` and `npm login` done (`npm whoami` confirms).
- [ ] `maturin` and `twine` installed (`pipx install maturin twine`), and PyPI auth configured (`~/.pypirc` / `TWINE_*`).
- [ ] `CHANGELOG.md` updated for the new version.
- [ ] `task rs:publish -- --status` — glance at the board; a clean released state is expected before starting a new release.
- [ ] `task rs:publish -- <level> --dry-run` — renders the board and runs only the reversible steps; review the version edits, the crate file lists, the **npm tarball contents** (it must include `wasm/fxtranslate_wasm_bg.wasm`), and the **PyPI artifacts** (the sdist + one local wheel, both passing `twine check`).
- [ ] `task rs:publish -- <level>` — real release (add `--pypi-upload` to include the PyPI push, or finish it by hand with the printed commands). If a step fails, fix the one red row and **re-run `task rs:publish`** (no level) to resume — green rows are skipped.
- [ ] Confirm: `cargo info fxtranslate` / `fxtranslate-cli`, `npm view fxtranslate version`, and `pip index versions fxtranslate` (or the [PyPI page](https://pypi.org/project/fxtranslate/)) show the new version, and the `fxtranslate-vX.Y.Z` tag is on the fork.
- [ ] Optionally `task rs:publish -- --watch-wheels` (or add it to the release run) to poll the CI wheel matrix to 4/4; a timeout is not a failure.

## Ordering and re-runs

Publishing to crates.io, then npm, then PyPI is **not atomic and not reversible** — an upload can only be yanked or deprecated. So the publisher checks auth/tooling, uploads every registry first (crates.io → npm → PyPI), and creates the git tag **last**, once everything is up; the tag therefore never points at a half-published release. (With the default no-`--pypi-upload` run, the PyPI push is deferred to the human; the tag is still created after the crates + npm uploads, and the printed commands complete the PyPI leg.)

### Resume and re-runs

There is **no `--initial` flag** and no state file. The one piece of "state" — the target version — lives in the manifests, and the publisher derives what to do from it plus the registries:

- **The manifest version is fully published on crates.io** → the tree is at a clean *released* state. A bare `task rs:publish` has nothing to do and stops (`X.Y.Z is fully published; specify patch|minor|major to start a new release`). A bump level (or `--set`) starts a **new** release.
- **The manifest version is NOT yet fully published** → a release for `V` is *in flight*. A bare `task rs:publish` (or `task rs:publish -- --resume`) **resumes** it at `V`. A bump level here is **refused** — `a release for V is in flight; re-run without a level to resume, or --set to override` — which is the double-bump guard: it prevents a re-run of `patch` from moving 0.4.1 → 0.4.2 and publishing a further version. (This replaces the old `--initial` and its footgun; resume is now the no-argument path.)

So a run interrupted *after* the version bump was committed (e.g. crates published but npm or PyPI failed) is completed by simply **re-running `task rs:publish`** with no level. Its probes read the world: crates already on crates.io and an npm version already on the registry are detected and skipped, the PyPI leg re-runs with `twine upload --skip-existing` (add `--pypi-upload`) so a version already on PyPI is left alone — only the uploads that didn't land, plus the tag and push, happen. Fix the one red row, re-run, watch the green rows get skipped.

`--dry-run` relaxes these guards so it can always show a board: a bare `--dry-run` previews the current manifest version regardless of release state, and `patch --dry-run` previews the bumped version — neither publishes anything.

## Troubleshooting

- **`npm publish` fails with `404 Not Found - PUT .../fxtranslate`** — you're not logged in. npm reports a logged-out *first* publish as a 404 (it won't reveal a package you can't see). Run `npm login`, confirm with `npm whoami`, then re-run `task rs:publish` to resume (the npm row was red; the green rows above it are skipped). The publisher checks `npm whoami` up front in Preflight, so a logged-out release shows a red **npm auth** row before touching crates.io.
- **A crate's `cargo publish --dry-run` reports "failed to select a version for ... `=X.Y.Z`"** — expected during a bump: `fxtranslate-cli` pins the not-yet-published engine version. It resolves once the engine uploads (engine publishes first). The dry-run flags it as a note, not a failure.

## First-time publishing

A package that has never been released is published by its first lockstep run like any other — its version starts wherever the ecosystem's shared version is at that release, with no back-fill of intermediate versions.

For the **very first publish** (nothing is on the registries yet, so the manifest version is *not* "fully published"), the resume model treats it as an in-flight release: run the bare command, which publishes the manifest version exactly as it stands, without bumping —

```console
$ task rs:publish                   # publishes the current manifest version as-is (the first-publish / resume path)
```

A bump level (`patch`/`minor`/`major`) or `--set` would be **refused** here with `a release for V is in flight` — because from crates.io's point of view `V` isn't published yet, so the guard reads it as an in-flight release, not a clean released state. This is the new equivalent of the old `--initial` first upload: publish the manifest version without bumping. (Set the desired first version in the manifests beforehand if it shouldn't be the current one, or use `task rs:publish -- --set X.Y.Z --dry-run` to preview a specific version — but the real first upload is the bare run.) npm needs no special handling, since a first `npm publish` simply claims the name; the npm package is public and unscoped, so no `--access public` is required unless the name is later moved under a scope.

PyPI is the same: a first `twine upload` (via `--pypi-upload`, or the printed finish commands) **claims the PyPI name**. Note the first real release ships **an sdist + the release machine's local-platform wheel only** — everyone can `pip install` from the sdist (compiling from source), and the release machine's platform also gets a binary wheel; the full binary-wheel matrix for other platforms is backfilled by CI on the tag (see below).

## Binary wheels (CI)

`task rs:publish` ships the sdist plus the release machine's own platform wheel, then creates the `fxtranslate-vX.Y.Z` tag last, once every registry is up. Pushing that tag triggers the `.github/workflows/pypi-wheels.yml` workflow, which builds and uploads the full per-platform binary wheel matrix — Linux `x86_64` + `aarch64`, macOS `arm64`, and Windows `x86_64` — so most users get a fast SIMD wheel with no toolchain instead of compiling from the sdist. (Intel macOS has no prebuilt wheel — its runner builds too slowly to gate the release on — so those users install from the sdist.) The crate is `abi3-py38`, so it is one wheel per platform (every CPython ≥ 3.8), not one per Python minor. The workflow also rebuilds the sdist for single provenance, uploads everything with `--skip-existing` (idempotent — it coexists with the operator's local sdist + wheel upload for the same version), and asserts every wheel is `abi3` before publishing. It can also be run manually from the Actions tab (`workflow_dispatch`) to test the matrix without cutting a release.

The board reflects this leg as a final **CI wheels** row — a *status-only* row that the local machine never runs (it can't build the matrix). Its probe counts how many of the four platform wheels have landed on PyPI and shows `(N/4)`, going green only at 4/4. Immediately after a release it reads `(0/4) — builds on the tag push`; it fills in as CI finishes each platform. To watch it in one sitting, add `--watch-wheels` (to the release run or on its own afterward): the publisher polls PyPI every 30s for up to 30 min, re-rendering the count until it reaches 4/4. A timeout — or Ctrl-C — is **not** a release failure (the wheels are a CI backfill; the release already succeeded); it just prints the current `(N/4)` and the Actions link, and you can re-dispatch with `gh workflow run pypi-wheels.yml`.

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
