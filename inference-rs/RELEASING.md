# Releasing fxtranslate

The ecosystem ships on **one shared version**, in lockstep:

- `fxtranslate` — the engine — on [crates.io](https://crates.io/crates/fxtranslate)
- `fxtranslate-cli` — the native CLI — on [crates.io](https://crates.io/crates/fxtranslate-cli)
- `fxtranslate` — the Node CLI + library — on [npm](https://www.npmjs.com/package/fxtranslate)

One command drives all three: [`scripts/publish.py`](./scripts/publish.py), wrapped as `task rs:publish`. It bumps every crate manifest and `npm/package.json` to the same version, builds and tests, publishes the crates in dependency order, publishes the npm package (rebuilding its wasm core first), and creates + pushes a single `fxtranslate-vX.Y.Z` tag once everything is up.

## Prerequisites

- **crates.io auth** — `cargo login` (a crates.io API token).
- **npm auth** — `npm login`, with publish rights to `fxtranslate` (`npm whoami` confirms the account).
- **wasm toolchain** — `wasm-pack` and the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`); the npm publish rebuilds the wasm core.
- **A clean tree on `main`, pushed to the `gregtatum` remote.** `origin` is upstream mozilla/translations; releases go to the fork, which is the publisher's default remote.

## Preview, then publish

Everything stays read-only until `--dry-run` is dropped.

```console
# Preview: validate crate packaging and the npm tarball, touch nothing.
$ task rs:publish -- patch --dry-run       # or: minor / major / --set X.Y.Z

# Release: bump → build+test → publish crates → publish npm → tag → push.
$ task rs:publish -- patch
```

The bump level (`patch` | `minor` | `major`) or `--set X.Y.Z` sets the next shared version. The dry-run validates the crates with `cargo publish --dry-run` and the npm package with `npm publish --dry-run` — the latter runs the `prepublishOnly` hook (rebuild wasm + typecheck + parity), so it is a faithful preview and correspondingly slow.

Other flags: `--no-push` (commit + tag locally, push by hand), `--allow-dirty`, `--skip-tests`, `--remote <name>`.

### Checklist

- [ ] `git status` clean, on `main`, on the `gregtatum` remote.
- [ ] `cargo login` and `npm login` done (`npm whoami` confirms).
- [ ] `CHANGELOG.md` updated for the new version.
- [ ] `task rs:publish -- <level> --dry-run` — review the version edits, the crate file lists, and the **npm tarball contents** (it must include `wasm/fxtranslate_wasm_bg.wasm`).
- [ ] `task rs:publish -- <level>` — real release.
- [ ] Confirm: `cargo info fxtranslate` / `fxtranslate-cli` and `npm view fxtranslate version` show the new version, and the `fxtranslate-vX.Y.Z` tag is on the fork.

## Ordering and re-runs

Publishing to crates.io then npm is **not atomic and not reversible** — an upload can only be yanked or deprecated. So the publisher uploads every registry first and creates the git tag **last**, once everything is up; the tag therefore never points at a half-published release. A run interrupted midway is recoverable: fix the cause and re-run the same command. Crates already on crates.io and an npm version already on the registry are detected and skipped, the remaining artifacts publish, then the tag lands.

## First-time publishing

A package that has never been released is published by its first lockstep run like any other — its version starts wherever the ecosystem's shared version is at that release, with no back-fill of intermediate versions. The crates used `publish.py --initial` for their very first crates.io upload (publish the manifest version without bumping); npm needs no equivalent, since a first `npm publish` simply claims the name. The npm package is public and unscoped, so no `--access public` is required unless the name is later moved under a scope.

## Manual npm publish

`task rs:publish` is the supported path, so the versions stay locked together. Publishing the npm package alone is a fallback only (re-releasing at an existing version is impossible on either registry — bump instead):

```console
$ cd npm
$ npm publish --dry-run     # prepublishOnly rebuilds wasm + typechecks + runs parity
$ npm publish
```

The `prepublishOnly` hook (`build:wasm && typecheck && check:parity`) runs on both, so even a manual publish cannot ship a stale or missing `wasm/`.
