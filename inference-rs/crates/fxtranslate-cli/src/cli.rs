//! CLI surface, split from `main.rs` so the whole thing is end-to-end testable
//! without a network or an engine. [`parse`] turns argv into a [`Command`] (no
//! I/O); [`run`] then dispatches and executes it against injected [`Deps`] (a
//! [`Fetch`] for `list`, a [`Translator`] for `translate`) and [`Io`] (captured
//! streams + explicit TTY/`NO_COLOR` facts). So `tests/` can drive
//! argv → transcript against fakes, and `main.rs` is a thin shim that wires the
//! real network, engine, and terminal into `run`.

use std::io::{BufRead, Write};
use std::process::ExitCode;

use crate::translate::{Session, Translator};
use fxtranslate::fetch::Fetch;
use fxtranslate::lang::display_name;
use fxtranslate::remote::{fetch_records, language_matches, pairs};
use fxtranslate::route::{catalog, PREFERRED_HUB};

pub const USAGE: &str = "\
fxtranslate — translate with Firefox Translations models

USAGE:
  fxtranslate list [lang] [--all]             List supported languages (or raw
                                              model pairs with --all; `list --help`)
  fxtranslate translate <src> <trg> [text…]   Translate: args if given, else stdin
                                              lines, else an interactive TTY prompt

Non-English pairs pivot through English automatically (`es → fr` runs `es → en`
then `en → fr`); see pivot-translations.md.

OPTIONS:
  --cache-dir <DIR>   Model cache directory (default: <platform cache>/fxtranslate)
  -h, --help          Show this help

EXAMPLES:
  fxtranslate list es
  echo \"Hola mundo.\" | fxtranslate translate es fr   # pivots via English
  fxtranslate translate en es \"Hello world.\"
  fxtranslate translate en es                 # interactive";

pub const LIST_USAGE: &str = "\
fxtranslate list — list supported languages and models

USAGE:
  fxtranslate list [lang] [--all]

By default, `list` shows LANGUAGES, not raw models. A language is listed under
\"Fully supported\" when it can translate both to and from other languages — every
such pair works, directly or by pivoting through English. Languages that ship a
model in only one direction (e.g. only `en → xx`) work only that way, so they
appear under \"Single-direction models\" with the direction they support.

Underneath, every Firefox Translations model is a one-way pair to or from English
(`en → es` and `es → en` are separate models). Pass --all to list those raw pairs.

A [lang] argument filters by prefix, so `zh` catches `zh-Hans` and `zh-Hant`.
Display names are the standard BCP 47 language names; a tag with no known name shows
the code. See pivot-translations.md for how non-English pairs are served.

EXAMPLES:
  fxtranslate list                    # supported languages
  fxtranslate list es                 # just Spanish
  fxtranslate list --all              # every raw src → trg model pair
  fxtranslate list es --all           # both raw directions for Spanish";

/// A parsed command line.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    /// Print top-level usage.
    Help,
    /// Print `list`-specific usage.
    ListHelp,
    /// Enumerate languages (default) or raw model pairs (`--all`), optionally
    /// filtered.
    List { query: Option<String>, all: bool },
    /// Translate `src`→`trg`; `text` empty = stdin/REPL.
    Translate {
        src: String,
        trg: String,
        text: String,
        cache_dir: Option<String>,
    },
}

/// Parse argv (without the program name) into a [`Command`]. Pure — no I/O — so
/// the whole arg grammar (subcommands, `--help` routing, `--cache-dir`) is unit
/// testable.
pub fn parse(args: &[String]) -> Result<Command, String> {
    let mut cache_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut help = false;
    let mut all = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => help = true,
            "--all" => all = true,
            "--cache-dir" => {
                cache_dir = Some(it.next().ok_or("--cache-dir needs a path")?.clone());
            }
            _ => positional.push(a.clone()),
        }
    }

    match positional.first().map(String::as_str) {
        Some("list") => {
            if help {
                Ok(Command::ListHelp)
            } else {
                Ok(Command::List {
                    query: positional.get(1).cloned(),
                    all,
                })
            }
        }
        // `--help` with a non-list (or no) command → top-level help.
        _ if help => Ok(Command::Help),
        None => Ok(Command::Help),
        Some("translate") => {
            // `translate` is explicit: `translate <src> <trg> [text…]`.
            if positional.len() < 3 {
                return Err(format!(
                    "`translate` needs `<src> <trg> [text…]`; got `{}`\n\n{USAGE}",
                    positional.join(" ")
                ));
            }
            Ok(Command::Translate {
                src: positional[1].clone(),
                trg: positional[2].clone(),
                text: positional[3..].join(" "),
                cache_dir,
            })
        }
        Some(other) => Err(format!(
            "unknown command `{other}`; expected `translate` or `list`\n\n{USAGE}"
        )),
    }
}

/// ANSI styling used by both list views: `(cyan source, green target, dim, reset)`,
/// or empty strings when `color` is off (the caller decides based on TTY /
/// `NO_COLOR`).
fn palette(color: bool) -> (&'static str, &'static str, &'static str, &'static str) {
    if color {
        ("\x1b[36m", "\x1b[32m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    }
}

/// Whether a `list` language view surfaces language `lang` for `query`. A bare
/// query prefix-matches the code (so `zh` catches `zh-Hans`/`zh-Hant`); a
/// `src-trg` query matches either half, so `en-es` surfaces both English and
/// Spanish in the language view.
fn language_query_matches(lang: &str, query: &str) -> bool {
    match query.split_once('-') {
        Some((a, b)) => lang.starts_with(a) || lang.starts_with(b),
        None => lang.starts_with(query),
    }
}

/// Write the aligned `src`→`trg` table for `rows` to `out`. Names and the source
/// tag are padded (by Unicode scalar count, matching `{:<width$}`) *before*
/// color-wrapping, so escape bytes never affect columns. Shared by `--all` and the
/// single-direction section of the default view.
fn render_pairs(rows: &[(String, String)], color: bool, out: &mut dyn Write) -> Result<(), String> {
    // Column widths (Unicode scalar counts). The source *tag* is padded too, so a
    // long source script tag like `(zh-Hans)` doesn't push the arrow out of line;
    // the target tag is the last column, so it needs no padding.
    let w_src = rows
        .iter()
        .map(|(s, _)| display_name(s).chars().count())
        .max()
        .unwrap_or(0);
    let w_stag = rows
        .iter()
        .map(|(s, _)| s.chars().count() + 2) // "(" + tag + ")"
        .max()
        .unwrap_or(0);
    let w_trg = rows
        .iter()
        .map(|(_, t)| display_name(t).chars().count())
        .max()
        .unwrap_or(0);

    let (cyan, green, dim, reset) = palette(color);
    for (s, t) in rows {
        let sname = format!("{:<w_src$}", display_name(s));
        let stag = format!("{:<w_stag$}", format!("({s})"));
        let tname = format!("{:<w_trg$}", display_name(t));
        writeln!(
            out,
            "{cyan}{sname}{reset} {dim}{stag}{reset} {dim}→{reset} {green}{tname}{reset} {dim}({t}){reset}"
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The `list --all` view: fetch records, filter by `query` ([`language_matches`]),
/// and write the aligned raw `src`→`trg` model pairs. Returns the number of pairs.
pub fn write_pairs(
    fetch: &dyn Fetch,
    query: Option<&str>,
    color: bool,
    out: &mut dyn Write,
) -> Result<usize, String> {
    let records = fetch_records(fetch)?;
    let all = pairs(&records);
    let shown: Vec<(String, String)> = all
        .iter()
        .filter(|(s, t)| query.map_or(true, |q| language_matches(s, t, q)))
        .cloned()
        .collect();
    if shown.is_empty() {
        return Err(format!(
            "no model pairs match `{}` ({} pairs available; try `fxtranslate list --all`)",
            query.unwrap_or(""),
            all.len()
        ));
    }
    render_pairs(&shown, color, out)?;
    Ok(shown.len())
}

/// The default `list` view: languages, not raw pairs. A language is *fully
/// supported* when it has a model both to and from the hub (English), so it reaches
/// every other fully-supported language — listed once, as itself. Languages with a
/// model in only one direction work only that way, so they're listed separately as
/// single-direction models. Returns the number of languages shown.
pub fn write_languages(
    fetch: &dyn Fetch,
    query: Option<&str>,
    color: bool,
    out: &mut dyn Write,
) -> Result<usize, String> {
    let records = fetch_records(fetch)?;
    let cat = catalog(&records, PREFERRED_HUB);

    let keep = |l: &&String| query.map_or(true, |q| language_query_matches(l, q));
    let bidi: Vec<&String> = cat.bidirectional.iter().filter(keep).collect();
    let source_only: Vec<&String> = cat.source_only.iter().filter(keep).collect();
    let target_only: Vec<&String> = cat.target_only.iter().filter(keep).collect();
    let total = bidi.len() + source_only.len() + target_only.len();
    if total == 0 {
        return Err(format!(
            "no languages match `{}` (try `fxtranslate list`, or `list --all` for raw pairs)",
            query.unwrap_or("")
        ));
    }

    let (cyan, _green, dim, reset) = palette(color);

    // Fully supported: one row per language, code-sorted (from `catalog`).
    if !bidi.is_empty() {
        writeln!(out, "Fully supported (translate to and from any other):")
            .map_err(|e| e.to_string())?;
        let w = bidi
            .iter()
            .map(|l| display_name(l).chars().count())
            .max()
            .unwrap_or(0);
        for l in &bidi {
            let name = format!("{:<w$}", display_name(l));
            writeln!(out, "  {cyan}{name}{reset} {dim}({l}){reset}").map_err(|e| e.to_string())?;
        }
    }

    // Single-direction models: the actual one-way pairs (hub→L for target-only,
    // L→hub for source-only), rendered compactly with a single `(src trg)` tag.
    let one_way: Vec<(String, String)> = target_only
        .iter()
        .map(|l| (PREFERRED_HUB.to_string(), (*l).clone()))
        .chain(
            source_only
                .iter()
                .map(|l| ((*l).clone(), PREFERRED_HUB.to_string())),
        )
        .collect();
    if !one_way.is_empty() {
        if !bidi.is_empty() {
            writeln!(out).map_err(|e| e.to_string())?;
        }
        writeln!(out, "Single-direction models:").map_err(|e| e.to_string())?;
        render_single_direction(&one_way, color, out)?;
    }

    Ok(total)
}

/// Write the single-direction rows as `source → target (src trg)`, names padded to
/// their columns. Unlike the `--all` table (which tags each name) this pairs the
/// two display names and closes with one compact `(src trg)` tag — the short form
/// for the one-way list in the default view.
fn render_single_direction(
    rows: &[(String, String)],
    color: bool,
    out: &mut dyn Write,
) -> Result<(), String> {
    let w_src = rows
        .iter()
        .map(|(s, _)| display_name(s).chars().count())
        .max()
        .unwrap_or(0);
    let w_trg = rows
        .iter()
        .map(|(_, t)| display_name(t).chars().count())
        .max()
        .unwrap_or(0);
    let (cyan, green, dim, reset) = palette(color);
    for (s, t) in rows {
        let sname = format!("{:<w_src$}", display_name(s));
        let tname = format!("{:<w_trg$}", display_name(t));
        writeln!(
            out,
            "{cyan}{sname}{reset} {dim}→{reset} {green}{tname}{reset} {dim}({s} {t}){reset}"
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The external dependencies [`run`] executes against: the network (for `list`
/// discovery) and the translator (for `translate`). `main.rs` passes the real
/// implementations; tests pass fakes.
pub struct Deps<'a> {
    pub fetch: &'a dyn Fetch,
    pub translator: &'a dyn Translator,
}

/// The I/O and terminal environment [`run`] executes against. Injected — rather
/// than probed from the process — so tests capture streams into buffers and set
/// the TTY / `NO_COLOR` facts explicitly.
pub struct Io<'a> {
    pub stdin: &'a mut dyn BufRead,
    pub stdout: &'a mut dyn Write,
    pub stderr: &'a mut dyn Write,
    /// stdin is a terminal → `translate` drops into the interactive REPL rather
    /// than reading piped lines.
    pub stdin_is_tty: bool,
    /// stdout is a terminal → `list` may color (also gated by `no_color`).
    pub stdout_is_tty: bool,
    /// `NO_COLOR` is set in the environment.
    pub no_color: bool,
}

/// Parse argv, dispatch to the matching command, and execute it against
/// `deps`/`io`. Errors are reported to `io.stderr` (prefixed `fxtranslate:`) so
/// they share the transcript with normal output; returns the process exit status.
pub fn run(args: &[String], deps: &Deps, io: &mut Io) -> ExitCode {
    match dispatch(args, deps, io) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(io.stderr, "fxtranslate: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: &[String], deps: &Deps, io: &mut Io) -> Result<(), String> {
    match parse(args)? {
        Command::Help => writeln!(io.stdout, "{USAGE}").map_err(|e| e.to_string()),
        Command::ListHelp => writeln!(io.stdout, "{LIST_USAGE}").map_err(|e| e.to_string()),
        Command::List { query, all } => {
            // Color only on an interactive stdout, and honor NO_COLOR.
            let color = io.stdout_is_tty && !io.no_color;
            if all {
                let n = write_pairs(deps.fetch, query.as_deref(), color, io.stdout)?;
                writeln!(io.stderr, "[{n} pairs]").map_err(|e| e.to_string())
            } else {
                let n = write_languages(deps.fetch, query.as_deref(), color, io.stdout)?;
                writeln!(io.stderr, "[{n} languages]").map_err(|e| e.to_string())
            }
        }
        Command::Translate {
            src,
            trg,
            text,
            cache_dir,
        } => run_translate(deps.translator, io, &src, &trg, &text, cache_dir.as_deref()),
    }
}

/// Load the `src`→`trg` session, then translate `text` if given, else stdin
/// lines (pipe mode), else an interactive prompt (TTY). Status lines go to
/// stderr so piped stdout carries only translations.
fn run_translate(
    translator: &dyn Translator,
    io: &mut Io,
    src: &str,
    trg: &str,
    text: &str,
    cache_dir: Option<&str>,
) -> Result<(), String> {
    writeln!(io.stderr, "[fxtranslate] resolving {src}→{trg} model…").map_err(|e| e.to_string())?;
    let session = translator.load(src, trg, cache_dir)?;
    // A non-hub pair with no direct model is served by pivoting; surface the hop
    // so the resolved route is visible (`es→en→fr`), not silently different.
    match session.pivot() {
        Some(hub) => writeln!(io.stderr, "[fxtranslate] ready ({src}→{hub}→{trg}, pivot).")
            .map_err(|e| e.to_string())?,
        None => {
            writeln!(io.stderr, "[fxtranslate] ready ({src}→{trg}).").map_err(|e| e.to_string())?
        }
    }

    if !text.is_empty() {
        writeln!(io.stdout, "{}", session.translate(text)).map_err(|e| e.to_string())?;
        return Ok(());
    }

    if io.stdin_is_tty {
        return repl(session.as_ref(), io, src, trg);
    }

    // Pipe mode: one translation per input line (marian-style).
    let mut line = String::new();
    loop {
        line.clear();
        let n = io.stdin.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(()); // EOF
        }
        // Strip only the line terminator, matching `BufRead::lines`.
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        writeln!(io.stdout, "{}", session.translate(&line)).map_err(|e| e.to_string())?;
    }
}

/// Minimal interactive prompt: a line in, its translation out, until EOF.
fn repl(session: &dyn Session, io: &mut Io, src: &str, trg: &str) -> Result<(), String> {
    writeln!(
        io.stderr,
        "Interactive {src}→{trg}. Type a sentence and press Enter; Ctrl-D to quit."
    )
    .map_err(|e| e.to_string())?;
    let mut line = String::new();
    loop {
        write!(io.stderr, "{src}→{trg}» ").map_err(|e| e.to_string())?;
        io.stderr.flush().map_err(|e| e.to_string())?;
        line.clear();
        let n = io.stdin.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            writeln!(io.stderr).map_err(|e| e.to_string())?;
            return Ok(()); // EOF
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        writeln!(io.stdout, "{}", session.translate(text)).map_err(|e| e.to_string())?;
    }
}
