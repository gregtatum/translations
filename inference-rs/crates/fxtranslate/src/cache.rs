//! Local model cache: resolve a cache dir, download a record's zstd attachment,
//! decompress it, verify its SHA-256 against the record's `decompressedHash`, and
//! store it atomically. A subsequent run with a matching hash is a cache hit and
//! skips the network.

use std::cell::Cell;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::fetch::{download_retrying, Fetch, RetryPolicy};
use crate::remote::Record;

/// Hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Pass `bytes` through iff its SHA-256 matches the record's `decompressedHash`
/// (records with no hash are trusted as-is). Returns the bytes on success so it
/// chains after `zstd_decode` with `.and_then`.
fn verify_hash(record: &Record, bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    if let Some(expected) = &record.decompressed_hash {
        let got = sha256_hex(&bytes);
        if &got != expected {
            return Err(format!(
                "hash mismatch for {} after download: got {got}, expected {expected}",
                record.name
            ));
        }
    }
    Ok(bytes)
}

/// Decompress a zstd stream fully into memory (pure-Rust decoder).
pub fn zstd_decode(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut dec = ruzstd::StreamingDecoder::new(input).map_err(|e| format!("zstd init: {e}"))?;
    let mut out = Vec::new();
    dec.read_to_end(&mut out)
        .map_err(|e| format!("zstd decode: {e}"))?;
    Ok(out)
}

/// Draw a single in-place (`\r`, no newline) download progress line to stderr,
/// e.g. `  model.enes.bin: 12.4 / 31.0 MiB (40%)`, falling back to a bytes-only
/// line when the server didn't advertise a `Content-Length`. The caller prints a
/// trailing newline once the download finishes.
fn render_progress(name: &str, done: u64, total: Option<u64>) {
    const MIB: f64 = 1024.0 * 1024.0;
    let mut err = std::io::stderr();
    let line = match total {
        Some(t) if t > 0 => {
            let pct = (done as f64 / t as f64 * 100.0).round() as u64;
            format!(
                "  {name}: {:.1} / {:.1} MiB ({pct}%)",
                done as f64 / MIB,
                t as f64 / MIB
            )
        }
        _ => format!("  {name}: {:.1} MiB", done as f64 / MIB),
    };
    // Pad to clear any longer previous line, then return to column 0.
    let _ = write!(err, "\r{line:<60}");
    let _ = err.flush();
}

/// The verified model-file paths the engine loads. For shared-vocab pairs
/// `src_vocab == trg_vocab`; split-vocab (CJK) pairs differ. `lex` is present only
/// when the pair ships a shortlist.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub model: PathBuf,
    pub src_vocab: PathBuf,
    pub trg_vocab: PathBuf,
    pub lex: Option<PathBuf>,
}

/// One locally cached pair directory and its on-disk footprint — a row in the
/// `models list` view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedPair {
    /// The `<src>-<trg>` directory name — the id `pair_files`/`remove_pair` take.
    pub name: String,
    /// The pair directory itself (`<root>/<name>`).
    pub dir: PathBuf,
    /// Sum of regular-file sizes under `dir`, excluding in-flight temp files.
    pub bytes: u64,
}

/// Recursive sum of regular-file byte lengths under `dir`, skipping dot-prefixed
/// entries — which covers the cache's own in-flight temp files (`.<name>.download`,
/// `.<name>.partial` from [`Cache::ensure`]) so a download in progress never inflates
/// the reported size. A missing `dir` is `0`, not an error.
pub fn dir_size(dir: &Path) -> Result<u64, String> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut total = 0;
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        if meta.is_dir() {
            total += dir_size(&entry.path())?;
        } else if meta.is_file() {
            total += meta.len();
        }
    }
    Ok(total)
}

/// A directory of cached, decompressed model files.
pub struct Cache {
    root: PathBuf,
    /// Download retry/backoff policy (see [`RetryPolicy`]).
    retry: RetryPolicy,
    /// Draw a `\r`-updated download progress line to stderr. The caller sets this
    /// from `stderr().is_terminal()` (policy in `main`), so pipes/CI/tests stay
    /// quiet — the same TTY split as `write_list`'s `color`.
    show_progress: bool,
}

impl Cache {
    /// The default cache root: the platform-native cache directory (`dirs`) with
    /// an `fxtranslate/models` subtree — e.g. `~/Library/Caches/fxtranslate/models`
    /// on macOS, `$XDG_CACHE_HOME` (or `~/.cache`) `/fxtranslate/models` on Linux,
    /// `%LOCALAPPDATA%\fxtranslate\models` on Windows. Falls back to a local dir if
    /// the cache directory can't be determined. Per-pair files then live under
    /// `<root>/<src>-<trg>/` (see [`Cache::pair_dir`]).
    pub fn locate() -> Cache {
        let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".fxtranslate-cache"));
        Cache::with_root(base.join("fxtranslate").join("models"))
    }

    /// A cache rooted at an explicit path (tests, or a `--cache-dir` override).
    pub fn with_root(root: impl Into<PathBuf>) -> Cache {
        Cache {
            root: root.into(),
            retry: RetryPolicy::default(),
            show_progress: false,
        }
    }

    /// Enable (or disable) the download progress line. `main` turns this on only
    /// when stderr is a TTY.
    pub fn with_progress(mut self, show: bool) -> Cache {
        self.show_progress = show;
        self
    }

    /// Override the download retry policy (tests use [`RetryPolicy::no_delay`] to
    /// exercise the retry loop without real backoff sleeps).
    pub fn with_retry(mut self, policy: RetryPolicy) -> Cache {
        self.retry = policy;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Per-pair directory: `<root>/<src>-<trg>`.
    pub fn pair_dir(&self, src: &str, trg: &str) -> PathBuf {
        self.root.join(format!("{src}-{trg}"))
    }

    /// Resolve a pair directory by its (joined) name, refusing anything that isn't a
    /// single normal path component — so a stray `..`, an absolute path, or a nested
    /// name can never make `pair_files`/`remove_pair` touch a dir outside the cache.
    fn pair_path(&self, name: &str) -> Result<PathBuf, String> {
        let mut comps = Path::new(name).components();
        match (comps.next(), comps.next()) {
            (Some(std::path::Component::Normal(_)), None) => Ok(self.root.join(name)),
            _ => Err(format!("invalid model name `{name}`")),
        }
    }

    /// Enumerate cached pairs: each immediate sub-directory of the cache root, sorted
    /// by name, with its computed size (see [`dir_size`]). A missing root means
    /// nothing has been cached yet, so this yields an empty `Vec`, not an error.
    pub fn list_cached(&self) -> Result<Vec<CachedPair>, String> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut cached = Vec::new();
        for entry in fs::read_dir(&self.root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if !entry.file_type().map_err(|e| e.to_string())?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let dir = entry.path();
            let bytes = dir_size(&dir)?;
            cached.push(CachedPair { name, dir, bytes });
        }
        cached.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(cached)
    }

    /// The cached files in the `name` pair directory as `(file name, size, path)`,
    /// sorted by name; skips the cache's own temp files. An absent directory yields an
    /// empty `Vec` (nothing cached for that pair).
    pub fn pair_files(&self, name: &str) -> Result<Vec<(String, u64, PathBuf)>, String> {
        let dir = self.pair_path(name)?;
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut files = Vec::new();
        for entry in fs::read_dir(&dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let fname = entry.file_name().to_string_lossy().into_owned();
            if fname.starts_with('.') {
                continue;
            }
            let meta = entry.metadata().map_err(|e| e.to_string())?;
            if meta.is_file() {
                files.push((fname, meta.len(), entry.path()));
            }
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(files)
    }

    /// Delete the `name` pair directory and everything in it. Idempotent: `Ok(false)`
    /// if it wasn't there, `Ok(true)` if it was removed.
    pub fn remove_pair(&self, name: &str) -> Result<bool, String> {
        let dir = self.pair_path(name)?;
        if !dir.exists() {
            return Ok(false);
        }
        fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
        Ok(true)
    }

    /// Assemble [`ModelFiles`] from whatever is already in the pair's cache directory,
    /// with no network and no Remote Settings records — the offline path. A cached
    /// file's on-disk name is its record name, whose leading dotted component is the
    /// file type (`model.*`, `vocab.*`, `srcvocab.*`, `trgvocab.*`, `lex.*`), so the
    /// type is read straight off the filename; the cache's own `.`-prefixed temp files
    /// are ignored. Returns `None` unless a model and a vocabulary (shared, or both
    /// split halves) are present. Hashes aren't re-checked — the online run that wrote
    /// these files already verified them.
    pub fn cached_model(&self, src: &str, trg: &str) -> Option<ModelFiles> {
        let (mut model, mut vocab, mut src_vocab, mut trg_vocab, mut lex) =
            (None, None, None, None, None);
        for entry in fs::read_dir(self.pair_dir(src, trg)).ok()?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let slot = if name.starts_with("model.") {
                &mut model
            } else if name.starts_with("srcvocab.") {
                &mut src_vocab
            } else if name.starts_with("trgvocab.") {
                &mut trg_vocab
            } else if name.starts_with("vocab.") {
                &mut vocab
            } else if name.starts_with("lex.") {
                &mut lex
            } else {
                continue; // temp files (`.…`) and anything unrecognized
            };
            *slot = Some(entry.path());
        }
        // Shared vocab serves both directions; otherwise both split halves are required.
        let (src_vocab, trg_vocab) = match vocab {
            Some(v) => (v.clone(), v),
            None => (src_vocab?, trg_vocab?),
        };
        Some(ModelFiles {
            model: model?,
            src_vocab,
            trg_vocab,
            lex,
        })
    }

    /// Ensure `record`'s decompressed file is present and hash-verified, fetching
    /// (and decompressing) it via `fetch` only on a miss or a hash mismatch.
    /// Returns the on-disk path.
    pub fn ensure(&self, fetch: &dyn Fetch, record: &Record) -> Result<PathBuf, String> {
        let dir = self.pair_dir(&record.src, &record.trg);
        let dest = dir.join(&record.name);

        // Cache hit: file present and (if we know the expected hash) matching.
        if dest.is_file() {
            match &record.decompressed_hash {
                Some(expected) => {
                    let got = sha256_hex(&fs::read(&dest).map_err(|e| e.to_string())?);
                    if &got == expected {
                        return Ok(dest);
                    }
                    // Corrupt/partial/stale — fall through and re-fetch.
                    eprintln!(
                        "[cache] {} hash mismatch (have {}…, want {}…); re-fetching",
                        record.name,
                        &got[..8.min(got.len())],
                        &expected[..8.min(expected.len())]
                    );
                }
                None => return Ok(dest), // no hash to check against; trust presence
            }
        }

        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

        // Stream the (zstd) attachment to a partial file with timeouts + retry +
        // Range resume (memory stays bounded — it never fully buffers the body).
        // Progress renders only when enabled; the `Cell` tracks whether the `\r` line
        // was drawn so we can close it with a newline after each download.
        let url = record.cdn_url();
        let download = dir.join(format!(".{}.download", record.name));
        let show = self.show_progress;
        let name = &record.name;
        let drew = Cell::new(false);
        let mut on_progress = |done: u64, total: Option<u64>| {
            if show {
                render_progress(name, done, total);
                drew.set(true);
            }
        };

        // Download, then decode + verify. A body that fails verification *after it
        // resumed* could be a bad splice, so wipe the partial and re-download once
        // cleanly; a clean single-attempt download that still mismatches is a
        // genuinely wrong `decompressedHash`, so we don't waste a second fetch on it.
        let mut healed = false;
        let bytes = loop {
            let stats = download_retrying(fetch, &url, &download, &self.retry, &mut on_progress);
            if drew.replace(false) {
                eprintln!(); // finish the in-place progress line
            }
            let stats = stats?;
            let assembled = fs::read(&download)
                .map_err(|e| e.to_string())
                .and_then(|compressed| zstd_decode(&compressed))
                .and_then(|bytes| verify_hash(record, bytes));
            match assembled {
                Ok(bytes) => break bytes,
                Err(e) => {
                    let _ = fs::remove_file(&download);
                    if stats.attempts > 1 && !healed {
                        healed = true;
                        eprintln!(
                            "[cache] {}: assembled download failed verification; re-fetching cleanly",
                            record.name
                        );
                        continue;
                    }
                    return Err(e);
                }
            }
        };
        let _ = fs::remove_file(&download);

        // Atomic write: temp then rename, so an interrupted run never leaves a
        // partial file that a later run would trust.
        let tmp = dir.join(format!(".{}.partial", record.name));
        fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
        Ok(dest)
    }
}

/// Resolve, download+cache, and verify all files needed to translate `src`→`trg`.
/// Picks the latest-version records; handles shared vs. split (CJK) vocab.
pub fn ensure_model(
    fetch: &dyn Fetch,
    cache: &Cache,
    records: &[Record],
    src: &str,
    trg: &str,
) -> Result<ModelFiles, String> {
    use crate::remote::pick;

    let model = pick(records, "model", src, trg)
        .ok_or_else(|| format!("no model for {src}-{trg} in Remote Settings"))?;
    let model_path = cache.ensure(fetch, model)?;

    // Shared vocab ships one `vocab`; split vocab (CJK) ships `srcvocab`/`trgvocab`.
    let (src_vocab, trg_vocab) = if let Some(v) = pick(records, "vocab", src, trg) {
        let p = cache.ensure(fetch, v)?;
        (p.clone(), p)
    } else {
        let sv = pick(records, "srcvocab", src, trg)
            .ok_or_else(|| format!("no vocab/srcvocab for {src}-{trg}"))?;
        let tv = pick(records, "trgvocab", src, trg)
            .ok_or_else(|| format!("no trgvocab for {src}-{trg}"))?;
        (cache.ensure(fetch, sv)?, cache.ensure(fetch, tv)?)
    };

    let lex = match pick(records, "lex", src, trg) {
        Some(l) => Some(cache.ensure(fetch, l)?),
        None => None,
    };

    Ok(ModelFiles {
        model: model_path,
        src_vocab,
        trg_vocab,
        lex,
    })
}
