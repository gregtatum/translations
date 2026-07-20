# Publish orchestrator

Publishing is a multi-step process that is fallible across multiple package managers. Here we want to be in lockstep on the release. The goal would be to update the publish script to be a better orchestrator. Let's do a command and control interface that hides stdout, and only surfaces stdout/err when there is a failure. Take inspiration from `task rs:check`. It should show the current status of the publish. So if a previous one failed, it should surface that so I can run it over and over again until I have a pass. So maybe we can kick off a `patch` `minor` `major` run, then re-run as much as possible with a command and control interface that suggests self-healing. That way for instance, if I'm not logged into npm, it doesn't matter, I can just re-run it following the instructions until I get a full success. On failures I get the full log, and helpful hints on what happens next.

Consider how to do this for:

 - crates.io fxtranslate
 - crates.io fxtranslate-cli
 - npm fxtranslate
 - pypi fxtranslate
    - all platform wheels

Consider ergonomics and style for how it looks based off of task rs:check.

---

## Design

### The core idea: derive status from the world, not from a state file

The trap with a resumable multi-registry publisher is local state: a `.publish-state.json`
that records "crates done, npm not" drifts from reality the moment anything happens out of
band (a manual `npm publish`, a yanked crate, a half-uploaded wheel). We already avoid this
in `publish.py` — `already_published()` reads the crates.io sparse index, `publish_npm()`
treats "previously published" as success, `publish_pypi()` uses `twine upload --skip-existing`.
The orchestrator generalizes that instinct into the whole design:

> **Every step knows how to ask the world whether it is already done.** Status is *computed*,
> never *stored*. Re-running is therefore inherently safe and self-healing — the tool re-derives
> reality each time, skips what's green, and retries what isn't.

This is what makes "run it over and over until it passes" work without bookkeeping. Not logged
into npm? The npm step stays red; fix it, re-run, and every already-green step is skipped by its
own probe. There is no stale state to reset.

### Shape: a two-pass status board, modeled on `task rs:check`

`check.py` already owns the visual language we want: hidden stdout, a compact live-updating tree
grouped into sections, `✓/x/·/•` glyphs, per-group wall-time, and a full captured-log dump only
for failures (`print_failures`). The orchestrator reuses that language, but the control flow is
different — publishing is an ordered dependency chain with a *probe* distinct from a *run*, where
`check.py` is an unordered fan-out of homogeneous tasks.

So the model is a `Step`, not a `Check`:

```python
class Step:
    label: str          # "crates.io  fxtranslate"
    group: str          # "Publish"
    def probe(self) -> Status:   # read-only: DONE / PENDING / BLOCKED(detail); no side effects
    def run(self) -> None:       # do the work; raise StepError(log, hint) on failure
    hint: str                    # self-healing instruction shown on failure
```

The run has **two passes**:

1. **Probe pass** — run every step's `probe()` (read-only registry/git queries; safe to fan out
   in parallel like `check.py`'s light checks) and render the board. *This is the "show me the
   current status" the issue asks for* — a previous failed run surfaces here as red rows before
   anything executes.
2. **Execute pass** — walk steps in order; skip `DONE`; run each `PENDING` with stdout hidden and
   live `•`/`✓`/`x`; on the first failure, dump the full captured log plus the step's healing hint
   and stop (`check.py`'s stop-at-first-failure, but ordered because the chain is dependent — the
   CLI crate pins the engine, npm/PyPI publish after crates).

### The steps

Grouped for the board. Each row's **probe** is what makes re-runs idempotent; most of the **run**
bodies already exist in `publish.py` and just get wrapped.

| Group | Step | Probe (is it already done?) | Run maps to |
|-------|------|------------------------------|-------------|
| **Preflight** | Clean tree · branch · remote | `git status`/branch/remote check | `preflight()` |
| | crates.io auth | token present (`~/.cargo/credentials` / `CARGO_REGISTRY_TOKEN`) | — |
| | npm auth | `npm whoami` succeeds | `check_npm_auth()` |
| | PyPI tooling + creds | `maturin`/`twine` on PATH; `~/.pypirc`/`TWINE_*` | `check_pypi_publish_readiness()` |
| **Build & validate** | Version bumped & committed | manifests + `package.json` at target V, and a `release: fxtranslate V` commit exists | `rewrite_versions`/`rewrite_npm_version` + commit |
| | cargo build + test | (always runs unless `--skip-tests`) | `cargo build`/`cargo test` |
| | Packaging validated | (always runs; cheap-ish, and it's the dry-run half) | `validate_packaging`/`validate_npm_packaging`/`validate_pypi_packaging` |
| **Publish** | crates.io `fxtranslate` | sparse index has V, not yanked | `publish_crate()` |
| | crates.io `fxtranslate-cli` | sparse index has V, not yanked | `publish_crate()` |
| | npm `fxtranslate` | `npm view fxtranslate@V version` | `publish_npm()` |
| | PyPI `fxtranslate` (sdist + local wheel) | PyPI JSON API lists V | `publish_pypi()` |
| **Tag & push** | Tag `fxtranslate-vV` | local tag exists | `git tag` |
| | Push branch + tag | tag present on `--remote` (`git ls-remote --tags`) | `git push` |
| **CI wheels** | Platform wheels (5 targets) | PyPI JSON API has a wheel for each of linux x86_64/aarch64, macos x86_64/arm64, win x86_64 | *status-only* (see below) |

### Resume semantics: derive the target version, refuse the double-bump

The one piece of "state" is the target version, and it lives in the manifests, not a sidecar file.
The rule that makes `patch`/`minor`/`major` then bare re-runs work — and kills the `--initial`
footgun (a re-run of `patch` bumping 0.4.1→0.4.2→0.4.3):

- **manifest version is fully published** (on crates.io) → the tree is at a clean released state.
  A bare re-run has nothing to do ("specify patch|minor|major to start a release"). A bump arg
  starts a new release.
- **manifest version is NOT yet fully published** → a release for V is *in flight*. A bare re-run
  (or `--resume`) resumes it at V. A bump arg is **refused** — "a release for V is in flight;
  re-run without a level to resume, or `--set` to override" — which is exactly the mistake
  `--initial` exists to prevent, now caught automatically instead of by documentation.

`--initial` collapses into "resume", which is just the no-argument path.

### Failure UX

On the first failing step, exactly like `check.py`'s `print_failures`: a red header with the
step label and exit code, the full captured stdout/stderr (hidden until now), and — the addition
the issue asks for — a **healing hint** and **what happens next**. Hints are per-step and concrete:

- npm auth → `run: npm login` (then `npm whoami` to confirm), then re-run `task rs:publish`.
- crates.io `fxtranslate-cli` failed after the engine published → "the engine is up; re-run to
  publish the CLI, then tag."
- PyPI upload with no creds → the exact `maturin build … && twine upload dist/*` finish commands
  (already printed by `publish_pypi`), or configure `~/.pypirc` and re-run with `--pypi-upload`.
- CI wheels incomplete → link to the `pypi-wheels.yml` Actions run; re-dispatch via
  `gh workflow run pypi-wheels.yml` or the Actions tab.

Because the board is world-derived, the loop the issue describes is literal: fix the one red row,
run `task rs:publish` again, watch it skip everything green and pick up where it stopped.

### The CI-wheels leg is status-only

`task rs:publish` cannot build the full wheel matrix — `.github/workflows/pypi-wheels.yml` does,
triggered by the tag push. But the orchestrator *can and should* report it so the board reflects
the true end-state of the whole release, not just the local half. Its `probe()` queries the PyPI
JSON API and counts how many of the 5 expected platform wheels are present (`(3/5)`), and its
`run()` does no building — it just surfaces the Actions link and, optionally under `--watch-wheels`,
polls until 5/5 or a timeout. This keeps the "all platform wheels" concern visible in the same
dashboard without pretending the local machine produces them.

### Shared rendering with `check.py`

To keep the two boards visually identical (and not fork the ANSI/tree/redraw code), extract the
presentation layer from `check.py` into a small `scripts/statusboard.py`: the color/`styled`
helpers, `status_glyph`, the group-tree renderer with in-place redraw (`\x1b[…F` / `\x1b[2K`), and
the failure-log formatter. `check.py` keeps its scheduler (parallel + heavy-semaphore); the
orchestrator supplies its own two-pass ordered scheduler. Both render through the shared module,
so `task rs:check` and `task rs:publish` read as one family.

### CLI surface

Unchanged where possible; `publish.py` keeps its flags (`--dry-run`, `--allow-dirty`,
`--skip-tests`, `--remote`, `--no-push`, `--pypi-upload`, `--set`). Additions:

- bare `task rs:publish` / `--resume` — resume the in-flight release (replaces `--initial`).
- `--status` — probe pass only; render the board and exit, executing nothing. Pure dashboard for
  "where did the last run get to?"
- `--watch-wheels` — after push, poll the PyPI JSON API until the wheel matrix is complete.

### Example board

Resuming an in-flight `0.4.2` where crates landed but npm login expired:

```
◆ fxtranslate 0.4.2   resuming in-flight release              12.3s

Preflight
├─ ✓ Clean tree · main · gregtatum
├─ ✓ crates.io auth
├─ x npm auth                       not logged in
└─ ✓ PyPI tooling + creds

Build & validate
├─ ✓ Version bumped & committed     0.4.1 → 0.4.2
├─ ✓ cargo build + test             8.1s
└─ ✓ Packaging validated            crates + npm + pypi

Publish
├─ ✓ crates.io  fxtranslate         0.4.2 live
├─ ✓ crates.io  fxtranslate-cli     0.4.2 live
├─ · npm        fxtranslate         pending
└─ · PyPI       fxtranslate         pending

Tag & push
├─ · Tag fxtranslate-v0.4.2
└─ · Push to gregtatum

CI wheels
└─ · Platform wheels (0/5)          builds on tag push

✖ Failures
──────────
┌─ npm auth failed   not authenticated to npm
│  crates.io is published first, so releasing logged out would half-publish.
└─ next: run `npm login`, confirm with `npm whoami`, then re-run `task rs:publish`
```

Fix npm, re-run, and the three `crates.io`/build rows probe green instantly and are skipped; the
run picks up at npm.

### Implementation plan

1. Extract `scripts/statusboard.py` from `check.py` (rendering + failure dump); repoint `check.py`
   at it (no behavior change — a good standalone commit).
2. Refactor `publish.py`'s procedural `main()` into `Step` objects wrapping the existing functions,
   each gaining a `probe()`. The upload steps' probes are the existing `already_published` /
   `npm view` / PyPI-JSON checks lifted out of the run bodies.
3. Add the two-pass ordered scheduler (probe-all → render → execute-in-order, stop at first fail).
4. Replace `--initial` with version-derived resume + the double-bump guard; add `--status`.
5. Add the CI-wheels status probe (PyPI JSON matrix count) and optional `--watch-wheels`.
6. Update `RELEASING.md` and the `rs:publish` task summary for the resume-by-re-run model.

### Open questions

- **crates.io auth probe** has no clean `whoami`; token-file/env presence is a proxy, and a real
  auth failure only shows at the first `cargo publish`. Acceptable (fail-at-use with a hint), or
  worth a probe request against the API?

> Fail at use with a hint.

- **Parallelizing uploads**: crates must be ordered (CLI pins engine), but npm and PyPI are
  independent of each other. Worth running them concurrently, or keep the strictly-serial chain
  for a simpler, more predictable board?

> Serialize them.

- **`--dry-run` vs the board**: fold dry-run into the same two-pass board (execute pass runs only
  the validate steps), or keep the current linear dry-run output? Folding is more consistent.

> Let's make dry-run as consistent as the real thing.
