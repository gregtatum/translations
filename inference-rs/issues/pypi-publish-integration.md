# PyPI publish integration + docs (the dry-run → publish path)

**Open. Depends on [pypi-pyo3-bindings.md](pypi-pyo3-bindings.md) (the crate must exist first).
Precedes [pypi-wheel-matrix-ci.md](pypi-wheel-matrix-ci.md).** This makes the PyPI leg a real,
lockstep part of the release, the way [closed/26-npm-publish-checklist.md](closed/26-npm-publish-checklist.md)
made npm one. It extends `scripts/publish.py` + `task rs:publish` with a third registry leg,
writes the package-focused Python README/DEVELOPMENT pair, updates cross-advertising across all
three ecosystems, and extends `RELEASING.md`. As with the npm and Rust-package work, **do not
publish** — the human runs the dry-run and the real upload themselves.

## Background

Today `scripts/publish.py` releases two registries in lockstep on one shared version: crates.io
(engine + CLI) and npm (the wasm package). It bumps every crate manifest and `npm/package.json`
together, verifies auth first, does all reversible work (bump/build/test/validate packaging),
then the irreversible uploads, then tags last. PyPI is a third registry that must move on the
same version and the same safety discipline. The twist: **wheels are per-platform.** The first
release ships an sdist + the local-platform wheel (matches the npm mental model of building on the
release machine); the full binary-wheel matrix is backfilled by CI on the tag
([pypi-wheel-matrix-ci.md](pypi-wheel-matrix-ci.md)).

## Design

### `scripts/publish.py` — add the PyPI leg

The script already discovers `fxtranslate-py` as a workspace member and marks it "guarded"
(`publish = false`), so it is correctly kept off crates.io (verified in the current loader). The
PyPI leg is separate plumbing, parallel to the existing npm leg (`validate_npm_packaging` /
`publish_npm` / `check_npm_auth`).

Add constants alongside `NPM_DIR`:

```python
PY_DIR = WORKSPACE / "crates" / "fxtranslate-py"
PY_MANIFEST = PY_DIR / "Cargo.toml"        # the version source of truth (already a Crate)
```

**Version lockstep.** `fxtranslate-py`'s `Cargo.toml` version is already rewritten by
`rewrite_versions` (it's a workspace crate), so the crate version moves in lockstep for free —
this is why `dynamic = ["version"]` from the crate is preferred (see the bindings issue). If the
pinned maturin cannot read the crate version dynamically, add a `rewrite_pyproject_version(old,
new, apply)` mirroring `rewrite_npm_version` (anchored to the `[project]`-table `version` key) and
call it in both the preview and apply paths; extend `current_version` to include the pyproject
version in its drift check.

**Auth/readiness check** — add `check_pypi_publish_readiness()`, called in the same up-front block
as `check_npm_auth`, *before* any irreversible upload. Because the human owns the real upload, this
is a **readiness/tooling** check, not a credential login: verify `maturin` and `twine` are on PATH
(fail with an install hint: `pipx install maturin twine`), and warn (not die) if no PyPI
credential source is configured (`~/.pypirc`, `TWINE_*` env, or trusted-publishing context), so
the human is reminded but a dry-run still runs.

**Dry-run validation** — add `validate_pypi_packaging()`, called from the dry-run block next to
`validate_npm_packaging()`:

- `maturin build --sdist -m crates/fxtranslate-py/Cargo.toml -o <dist>` — builds the sdist +
  the local-platform wheel into a gitignored `dist/` under the py crate.
- `twine check <dist>/*` — validates metadata/README rendering.
- Report the artifacts that would ship (sdist filename + the one local wheel), mirroring how the
  npm leg reports its tarball. Die on failure, matching `validate_npm_packaging`.

Fold the PyPI verdict into the existing dry-run summary so the final "DRY RUN PASSED / OK WITH
NOTES" covers all three registries, and update the `steps`/`plan` strings to mention "publish
fxtranslate sdist + local wheel to PyPI".

**Real upload — gated for the human.** Add `publish_pypi()` in the irreversible section, ordered
**after** crates.io and npm (all three uploads before the tag, tag still last). Because "the human
handles the actual upload," gate it behind an explicit opt-in so a normal `task rs:publish` does
everything *except* the PyPI push and prints exactly the commands to finish:

```
maturin build --release --sdist -m crates/fxtranslate-py/Cargo.toml -o dist
twine upload dist/*
```

Add a `--pypi-upload` flag (default off). When off: build the sdist + local wheel, run
`twine check`, print the "to finish the PyPI leg, run: …" instructions — reversible work only, no
upload. When on: also run `twine upload --skip-existing`, treating "File already exists" as success
(mirrors `publish_crate` / `publish_npm` idempotent re-run handling) so `--initial` recovery is
safe. This keeps the crates.io + npm release fully automated while leaving the actual PyPI push to
the human's configured credentials, per the scope decision — and still lets a fully-authorized
operator do it in one shot with the flag.

**`--initial` recovery parity.** `--initial` must handle a PyPI-already-published state:
`twine upload --skip-existing` (there is no PyPI index probe as clean as the crates.io sparse
index; `--skip-existing` is the pragmatic equivalent) so a re-run after a partial failure only
uploads what didn't land. The tag is still created only after all three legs are up.

### Python README (package-focused) — `crates/fxtranslate-py/README.md`

Match `npm/README.md`'s style: usage-first, cross-advertises the other ecosystems, includes the
"The models" hosting caveat and a "Related packages" section. Sections:

- **Title + one-paragraph pitch**: translate with Firefox Translations models from Python;
  native compiled engine (fast, CPU-only), discovers/downloads/caches models for you.
- **Install**: `pip install fxtranslate`. Note the first release ships an sdist + one platform
  wheel, so on other platforms pip compiles from source (needs a Rust + C++ toolchain; falls back
  to the portable scalar kernel automatically) until the wheel matrix lands.
- **Library, batteries-included** (lead with this — the native advantage):
  ```python
  from fxtranslate import Translator
  t = Translator.load("en", "es")          # discover → download → cache → ready
  print(t.translate("The weather is nice today."))
  print(t.translate_long(long_article))
  ```
- **Library, bring-your-own model** (mirror npm's BYO example):
  ```python
  from fxtranslate import Translator
  model = open("en-es/model.bin", "rb").read()
  vocab = open("en-es/vocab.spm", "rb").read()
  t = Translator(model, vocab, vocab)       # src vocab twice for shared-vocab pairs
  ```
- **Discovery helpers**: `from fxtranslate import discovery` — `resolve_route`, `catalog`,
  `model_pairs`, `parse_records`, `segment_sentences`, `verify_and_decompress`.
- **The models**: mirror the npm README's hosting caveat (Remote Settings + Firefox CDN, platform
  cache dir, "provisioned for Firefox, not third-party traffic — mirror the files you depend on
  for production").
- **Related packages** (now three ecosystems): crates.io `fxtranslate` (Rust engine), crates.io
  `fxtranslate-cli` (native CLI), npm `fxtranslate` (Node CLI + library). Same engine, one shared
  version.
- **License**: MPL-2.0.

### Python DEVELOPMENT doc — `crates/fxtranslate-py/DEVELOPMENT.md`

Match `npm/DEVELOPMENT.md`: what the package is, layout, how it's built (maturin/PyO3, abi3, the
`python/` source package around the `_fxtranslate` compiled module), the offline-test strategy
(hermetic vs. opt-in translate parity, fixtures, oracle-pinned translate), the `rs:build-py` /
`rs:test-py` tasks, and a "Releasing" section pointing at `RELEASING.md` ("don't `twine upload` by
hand for a release — run the workspace publisher so all three registries move together and get one
tag").

### Cross-advertising — three ecosystems now

Update the "Related packages" / cross-links in all consumer READMEs so each of the three
ecosystems advertises the other two:

- `crates/fxtranslate/README.md` and `crates/fxtranslate-cli/README.md` — add the PyPI package
  alongside the existing npm cross-link.
- `npm/README.md` "Related packages" — add the PyPI `fxtranslate` entry.
- The new `crates/fxtranslate-py/README.md` — links crates.io + npm (specified above).

### `RELEASING.md` — extend to three registries

Parallel to the existing crates/npm sections:

- **Prerequisites**: add `maturin` + `twine` (`pipx install maturin twine`), and PyPI auth — a
  PyPI API token in `~/.pypirc`/`TWINE_*`, or (preferred, once CI lands) trusted publishing. Note
  the one-time pypi.org project setup is only needed for the CI/OIDC path
  ([pypi-wheel-matrix-ci.md](pypi-wheel-matrix-ci.md)).
- **Preview → publish**: the dry-run now also validates the PyPI sdist + local wheel via
  `maturin build` + `twine check`. Show the two-command finish for the PyPI leg and note
  `task rs:publish` does everything except the PyPI push by default; add `--pypi-upload` to
  include it.
- **Checklist**: add ticks for maturin/twine installed, PyPI auth configured, PyPI dry-run
  reviewed (sdist + one wheel), and post-release confirmation (`pip index versions fxtranslate` /
  the PyPI page shows the new version).
- **First-time publishing**: a first `twine upload` claims the PyPI name, like the first
  `npm publish`. Note the first real release ships **sdist + local wheel only**; the full binary
  wheel matrix is backfilled by CI on the tag.
- **Ordering**: PyPI uploads with crates.io + npm before the tag; the tag is still created last,
  once all three are up.

## Acceptance criteria

- `task rs:publish -- <level> --dry-run` validates crates, the npm tarball, **and** the PyPI sdist
  + local wheel (`maturin build` + `twine check`), and ends with a single verdict covering all
  three; touches no registry, file, or git ref.
- A real `task rs:publish -- <level>` bumps the py crate in lockstep, builds the sdist + local
  wheel, runs `twine check`, and prints the exact `twine upload` finish command; with
  `--pypi-upload` it also uploads (idempotent on re-run via `--skip-existing`).
- `--initial` completes a PyPI leg that didn't land without re-bumping; the tag is created only
  after all three registries are up.
- `crates/fxtranslate-py/README.md` and `DEVELOPMENT.md` exist and match the npm doc pair's
  style; the models caveat and related-packages cross-links are present.
- All three ecosystems cross-advertise the other two.
- `RELEASING.md` has the PyPI prerequisites, dry-run commands, checklist ticks, and first-publish
  note.

## Open questions

- **`--pypi-upload` gate vs. always-upload:** recommendation is to gate (default off) so the
  automated release stays crates+npm and the human explicitly opts into the PyPI push with their
  configured creds — matches the scope decision that "the human handles the actual upload."
- If maturin does not honor `dynamic` crate versioning, the `rewrite_pyproject_version` path is
  required and `current_version` must include pyproject in its drift check (resolve jointly with
  the bindings issue's open question).
- Should the human upload the local wheel built at release time, or always let CI produce every
  wheel (single provenance)? Leaning: first release uploads sdist + local wheel now (immediate
  `pip install` everywhere via sdist; the release machine's platform gets a binary wheel), CI
  backfills the rest. Revisit once CI exists.
- `dist/` cleanup: prefer a gitignored `dist/` under the py crate (so the human can inspect/upload
  the exact artifacts the dry-run built) over a temp dir; confirm.
