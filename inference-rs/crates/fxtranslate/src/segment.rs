//! Sentence segmentation for long-input translation.
//!
//! [`crate::engine::Engine::translate`] feeds the whole input to the decoder as
//! one sequence, so a line longer than the model's context window truncates: the
//! decoder emits EOS early and silently drops the tail. This module splits input
//! into sentences so each is translated within the window, matching how the
//! reference paths handle long text — the marian `translator-cli` (C++ `ssplit`)
//! and Firefox (JS `Intl.Segmenter`), both of which split before the decoder.
//!
//! The segmenter is pluggable. [`BasicSegmenter`] is always compiled — a small
//! punctuation splitter used when the `icu-segmenter` feature is off. With that
//! feature on, [`IcuSegmenter`] provides Unicode-correct (UAX #29), CJK-capable
//! segmentation; the CLI enables it. [`Engine::translate_segmented`] wires a
//! segmenter to per-sentence translation + whitespace-preserving [`reassemble`].
//!
//! [`Engine::translate_segmented`]: crate::engine::Engine::translate_segmented

/// The model's trained source context window in tokens (marian `max-length-break`,
/// default 128). Not stored in the `.bin` model (`Config` carries only
/// architecture dims), so it is a constant. A sentence whose source tokenization
/// fits the window is translated whole; a longer one is hard-wrapped to fit.
pub const CONTEXT_WINDOW: usize = 128;

/// Maximum source *content* tokens per translated unit: the window minus one slot
/// for the EOS marker the source pipeline appends (mirrors marian's
/// `wrapStep = maxLengthBreak - 1`). 128 − 1 = 127.
pub const MAX_SOURCE_TOKENS: usize = CONTEXT_WINDOW - 1;

/// A byte range into a source string (a UTF-8 boundary-aligned `start..end`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// The slice this span covers.
    pub fn of<'a>(&self, src: &'a str) -> &'a str {
        &src[self.start..self.end]
    }
}

/// Splits a source string into sentence spans. Implementations return the
/// **trimmed content** of each sentence (leading/trailing whitespace excluded) in
/// order; empty/whitespace-only sentences are omitted. The whitespace between two
/// spans — and any leading/trailing outer whitespace — is recovered from the
/// source during [`reassemble`], so Latin (spaces) and CJK (no spaces) round-trip.
pub trait Segmenter {
    fn sentences(&self, src: &str) -> Vec<Span>;
}

/// Trim ASCII+Unicode whitespace off `src[start..end]`, returning the content span,
/// or `None` if the range is empty or all whitespace.
fn trimmed_span(src: &str, start: usize, end: usize) -> Option<Span> {
    let slice = &src[start..end];
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        return None;
    }
    // `trim()` removes a prefix/suffix, so the trimmed slice sits inside `slice`;
    // its byte offset from `slice.as_ptr()` gives the new start.
    let off = trimmed.as_ptr() as usize - slice.as_ptr() as usize;
    Some(Span {
        start: start + off,
        end: start + off + trimmed.len(),
    })
}

/// Rejoin per-sentence `outputs` (one per span in `spans`, in order) into a single
/// string, restoring the original separators from `src`: between two sentences,
/// the source bytes that lay between their content spans (a space for Latin, empty
/// for CJK); and the input's leading/trailing outer whitespace. With `spans`
/// empty (empty or whitespace-only input) the source is returned unchanged, so a
/// blank line stays blank.
pub fn reassemble(src: &str, spans: &[Span], outputs: &[String]) -> String {
    debug_assert_eq!(spans.len(), outputs.len());
    if spans.is_empty() {
        return src.to_string();
    }
    let mut out = String::new();
    // Leading outer whitespace.
    out.push_str(&src[..spans[0].start]);
    for (i, piece) in outputs.iter().enumerate() {
        if i > 0 {
            // Original inter-sentence separator.
            out.push_str(&src[spans[i - 1].end..spans[i].start]);
        }
        out.push_str(piece);
    }
    // Trailing outer whitespace.
    out.push_str(&src[spans[spans.len() - 1].end..]);
    out
}

/// True for characters that end a sentence. ASCII terminators are only boundaries
/// when followed by whitespace/end (so decimals and abbreviations mostly survive);
/// the CJK/fullwidth terminators always are (no trailing space in those scripts).
fn is_ascii_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?')
}
fn is_hard_terminator(c: char) -> bool {
    // Fullwidth/ideographic stops used in CJK, plus the ellipsis.
    matches!(c, '。' | '！' | '？' | '…' | '｡')
}

/// Dependency-free sentence splitter used when `icu-segmenter` is off. Rule of
/// thumb: break after `.?!…` when the next character is whitespace or end, and
/// after CJK/fullwidth stops unconditionally. Heuristic (abbreviations, decimals,
/// and quotes are imperfect) but robust and allocation-light; `IcuSegmenter` is
/// the Unicode-correct upgrade.
pub struct BasicSegmenter;

impl Segmenter for BasicSegmenter {
    fn sentences(&self, src: &str) -> Vec<Span> {
        let mut spans = Vec::new();
        let mut start = 0usize;
        let mut chars = src.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let boundary = if is_hard_terminator(c) {
                true
            } else if is_ascii_terminator(c) {
                // Look at the next char: boundary if whitespace or end of input.
                match chars.peek() {
                    None => true,
                    Some(&(_, next)) => next.is_whitespace(),
                }
            } else {
                false
            };
            if boundary {
                let end = i + c.len_utf8();
                if let Some(s) = trimmed_span(src, start, end) {
                    spans.push(s);
                }
                start = end;
            }
        }
        // Trailing text with no terminator.
        if let Some(s) = trimmed_span(src, start, src.len()) {
            spans.push(s);
        }
        spans
    }
}

#[cfg(feature = "icu-segmenter")]
pub use icu_impl::IcuSegmenter;

#[cfg(feature = "icu-segmenter")]
mod icu_impl {
    use super::{trimmed_span, Segmenter, Span};
    use icu_segmenter::options::SentenceBreakInvariantOptions;
    use icu_segmenter::SentenceSegmenter;

    /// Unicode (UAX #29) sentence segmenter backed by `icu_segmenter` with bundled
    /// `compiled_data` — rule-based (no dictionaries), so CJK works without the
    /// large word-segmentation data. Language-agnostic (invariant) rules.
    pub struct IcuSegmenter;

    impl IcuSegmenter {
        pub fn new() -> IcuSegmenter {
            IcuSegmenter
        }
    }

    impl Default for IcuSegmenter {
        fn default() -> IcuSegmenter {
            IcuSegmenter::new()
        }
    }

    impl Segmenter for IcuSegmenter {
        fn sentences(&self, src: &str) -> Vec<Span> {
            if src.is_empty() {
                return Vec::new();
            }
            // Constructing from compiled data is a cheap borrow of static rules.
            let seg = SentenceSegmenter::new(SentenceBreakInvariantOptions::default());
            let bounds: Vec<usize> = seg.segment_str(src).collect();
            bounds
                .windows(2)
                .filter_map(|w| trimmed_span(src, w[0], w[1]))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(src: &str, spans: &[Span]) -> Vec<String> {
        spans.iter().map(|s| s.of(src).to_string()).collect()
    }

    #[test]
    fn basic_splits_multiple_sentences() {
        let src = "Hello world. How are you? I am fine!";
        let spans = BasicSegmenter.sentences(src);
        assert_eq!(
            texts(src, &spans),
            ["Hello world.", "How are you?", "I am fine!"]
        );
    }

    #[test]
    fn basic_single_sentence_is_verbatim() {
        let src = "The cat sat on the mat.";
        let spans = BasicSegmenter.sentences(src);
        assert_eq!(texts(src, &spans), [src]);
    }

    #[test]
    fn basic_trailing_text_without_terminator() {
        let src = "First one. Then a tail with no period";
        let spans = BasicSegmenter.sentences(src);
        assert_eq!(
            texts(src, &spans),
            ["First one.", "Then a tail with no period"]
        );
    }

    #[test]
    fn basic_decimal_is_not_a_boundary() {
        // The '.' in 3.14 is followed by a digit, not whitespace → not a split.
        let src = "Pi is 3.14 today. Yes.";
        let spans = BasicSegmenter.sentences(src);
        assert_eq!(texts(src, &spans), ["Pi is 3.14 today.", "Yes."]);
    }

    #[test]
    fn basic_cjk_fullwidth_stop_splits() {
        let src = "你好世界。天气很好。";
        let spans = BasicSegmenter.sentences(src);
        assert_eq!(texts(src, &spans), ["你好世界。", "天气很好。"]);
    }

    #[test]
    fn empty_and_whitespace_yield_no_spans() {
        assert!(BasicSegmenter.sentences("").is_empty());
        assert!(BasicSegmenter.sentences("   \n\t ").is_empty());
    }

    #[test]
    fn reassemble_latin_uses_source_spaces() {
        let src = "Hello world. How are you?";
        let spans = BasicSegmenter.sentences(src);
        let outputs = vec![
            "Bonjour le monde.".to_string(),
            "Comment ça va ?".to_string(),
        ];
        assert_eq!(
            reassemble(src, &spans, &outputs),
            "Bonjour le monde. Comment ça va ?"
        );
    }

    #[test]
    fn reassemble_cjk_has_no_separator() {
        let src = "你好世界。天气很好。";
        let spans = BasicSegmenter.sentences(src);
        let outputs = vec!["Hello world.".to_string(), "Nice weather.".to_string()];
        // No gap between the fullwidth stop and the next sentence → concatenated.
        assert_eq!(
            reassemble(src, &spans, &outputs),
            "Hello world.Nice weather."
        );
    }

    #[test]
    fn reassemble_single_span_is_output_verbatim() {
        let src = "Hello world.";
        let spans = BasicSegmenter.sentences(src);
        let outputs = vec!["Bonjour le monde.".to_string()];
        assert_eq!(reassemble(src, &spans, &outputs), "Bonjour le monde.");
    }

    #[test]
    fn reassemble_preserves_outer_whitespace() {
        let src = "  Hello.  World.  ";
        let spans = BasicSegmenter.sentences(src);
        let outputs = vec!["Bonjour.".to_string(), "Monde.".to_string()];
        assert_eq!(reassemble(src, &spans, &outputs), "  Bonjour.  Monde.  ");
    }

    #[test]
    fn reassemble_empty_returns_source() {
        assert_eq!(reassemble("   ", &[], &[]), "   ");
    }

    #[cfg(feature = "icu-segmenter")]
    #[test]
    fn icu_splits_latin_and_cjk() {
        let seg = IcuSegmenter::new();
        let latin = "Hello world. How are you? Fine.";
        assert_eq!(seg.sentences(latin).len(), 3);
        let cjk = "你好世界。天气很好。";
        assert_eq!(seg.sentences(cjk).len(), 2);
    }
}
