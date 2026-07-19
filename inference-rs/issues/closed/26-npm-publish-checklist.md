# npm publish checklist

**Closed, done.** Nothing code-level blocked the publish — both crates were already
on crates.io at 0.4.0 and npm was version-aligned but unpublished. The gaps were docs
and orchestration, now resolved:

- `npm/README.md` rewritten package-focused (CLI + library + related crates.io
  packages, no implementation details); the stale dev content moved to a new
  `npm/DEVELOPMENT.md` (conformance strategy, wasm-copy build, parity harness).
- Cross-advertising added both ways: the two crate READMEs point at the npm package,
  the npm README points at the crates.
- `scripts/publish.py` now publishes the whole ecosystem in **lockstep on one shared
  version** — it bumps `npm/package.json` + `package-lock.json` alongside the crate
  manifests, publishes crates → npm, and tags last (re-runs skip anything already up).
  `prepublishOnly` (build:wasm + typecheck + parity) guards against shipping a stale
  wasm core. `task rs:publish` and the docstring updated to match.
- `RELEASING.md` added: prerequisites, preview→publish commands, a tickable checklist,
  and the first-npm-publish story (npm starts at the next lockstep bump, 0.4.1).

The actual release (`npm login` + `task rs:publish -- patch`) is left to run manually.

---

Let's figure out what's blocking an npm publish. I know I want to rewrite the README.md. For npm, we want to have a package-focused README similar to: inference-rs/crates/fxtranslate/README.md and inference-rs/crates/fxtranslate-cli/README.md. We also want to retain a developer-focused documentation as well that explains the conformance strategy. For all the packaged readmes, it would be nice to cross reference the crates.io and npm packaging to cross-advertise them.

Don't include implementation details like sentence segmentation on the CLI documentation packages.

## dry-run

I want to do dry runs before publishing on npm. Give me the commands I need and maybe a checklist. Consider what a first publish looks like, and if there is a publish script. Maybe we should consider what it looks like to publish everything in lockstep so all of the ecosystem updates at the same time on the same versioning scheme?
