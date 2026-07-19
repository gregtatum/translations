# Binary wheel matrix on CI (PyPI, follow-up)

**Open, follow-up. Depends on [pypi-publish-integration.md](pypi-publish-integration.md) — the
package must be on PyPI first (sdist + local wheel).** The first PyPI release ships an sdist plus
the release machine's platform wheel; everywhere else `pip install fxtranslate` compiles from
source (needs a Rust + C++ toolchain, falls back to the portable scalar kernel). This issue
backfills prebuilt binary wheels for every platform via a GitHub Actions maturin job on the
`fxtranslate-vX.Y.Z` tag, so most users get a fast SIMD wheel with no toolchain.

## Background

Unlike crates.io (source) and npm (one wasm artifact), PyPI binary distribution is per-platform:
a wheel is compiled for a specific OS + CPU architecture. `PyO3/maturin-action` (the official
action) builds manylinux/macOS/Windows wheels in CI. Because the crate is `abi3-py38`, we need
**one wheel per platform**, not one per Python minor — a small matrix.

## Design

A workflow (e.g. `.github/workflows/pypi-wheels.yml`) triggered on tags matching
`fxtranslate-v*` (the tag `publish.py` creates last, once every registry is up — so wheels build
only for a real, fully-published release):

- **Matrix** (abi3, so one wheel per target, all Python 3.8+):
  - Linux `manylinux` x86_64 and aarch64
  - macOS x86_64 and arm64
  - Windows x86_64
- **Build**: `PyO3/maturin-action` with `-m crates/fxtranslate-py/Cargo.toml --release`. The
  crate's `gemmology` feature compiles the C++ SIMD kernel where a kernel + C++17 toolchain exist
  (aarch64 i8mm, x86_64 AVX2 per the core `build.rs`); manylinux images and the GitHub macOS/
  Windows runners provide the C++ toolchain. Where no SIMD kernel exists for a target, `build.rs`
  falls back to the portable scalar kernel without failing — so every matrix leg produces a
  working wheel.
- **abi3 verification**: assert each wheel's tag is `abi3` (e.g. `cp38-abi3-<platform>`), catching
  a regression where the stable-ABI feature silently drops.
- **Upload**: **PyPI Trusted Publishing (OIDC)** preferred — no long-lived token in CI. A publish
  job with `permissions: id-token: write` uses `pypa/gh-action-pypi-publish` (or maturin's upload)
  to push all matrix wheels. Fall back to a `PYPI_API_TOKEN` secret if trusted publishing isn't
  set up.
- **Idempotency**: `--skip-existing` on upload, so re-running the workflow (or overlapping with
  the human's local sdist upload for the same version) never errors on an already-present file.

### One-time human setup (document, don't automate)

- On pypi.org, configure a **Trusted Publisher** for the `fxtranslate` project pointing at this
  repo + workflow filename + environment. This is a one-time manual step in the PyPI project
  settings and cannot be scripted from here; note it in `RELEASING.md` prerequisites.
- Optionally add a GitHub Actions environment (e.g. `pypi`) with required reviewers for a manual
  approval gate before the wheel upload.

### RELEASING.md

Add a short "Binary wheels (CI)" note: after the local release tags `fxtranslate-vX.Y.Z`, the CI
workflow builds and uploads the full wheel matrix; the human confirms the wheels appear on the
PyPI page. Cross-link this issue.

## Acceptance criteria

- A workflow triggers on `fxtranslate-v*` tags and builds abi3 wheels for the five targets above.
- Every wheel is abi3 (`cp38-abi3-*`) and installs + imports on ≥ 2 CPython minors.
- Wheels upload via trusted publishing (OIDC), `--skip-existing`, and coexist with the
  human-uploaded sdist + local wheel for the same version.
- `RELEASING.md` documents the one-time PyPI trusted-publisher setup and the CI backfill step.

## Open questions

- Windows aarch64 and musllinux — include now or defer? Recommend deferring to keep the first
  matrix small; add if users ask.
- Should CI also build the sdist (skip-existing) for single provenance, or leave the sdist to the
  human's local release? Leaning: human uploads sdist + local wheel at release time (immediate
  `pip install` everywhere), CI backfills binary wheels — building the sdist in CI too is a
  harmless belt-and-suspenders. Decide when wiring the workflow.
- Does the aarch64-Linux leg cross-compile or run under emulation in maturin-action, and does the
  gemmology C++ kernel build cleanly there? Verify on first run; scalar fallback is the safety net.
