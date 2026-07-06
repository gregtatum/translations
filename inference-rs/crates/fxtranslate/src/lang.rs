//! A small database of static language-tag → display-name mapping.

/// `(tag, display name)`, sorted by tag. The names are the standard English BCP 47
/// names from `Intl.DisplayNames`. To refresh them (or after adding tags), feed the
/// tags from the first column to this Node snippet and paste the output back:
///
/// ```js
/// const names = new Intl.DisplayNames(["en"], { type: "language" });
/// for (const tag of tags) console.log(`    ("${tag}", "${names.of(tag)}"),`);
/// ```
const NAMES: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("ar", "Arabic"),
    ("as", "Assamese"),
    ("az", "Azerbaijani"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bangla"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("eo", "Esperanto"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("ff", "Fula"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("ga", "Irish"),
    ("gd", "Scottish Gaelic"),
    ("gl", "Galician"),
    ("gn", "Guarani"),
    ("gu", "Gujarati"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("my", "Burmese"),
    ("nb", "Norwegian Bokmål"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("nn", "Norwegian Nynorsk"),
    ("no", "Norwegian"),
    ("oc", "Occitan"),
    ("or", "Odia"),
    ("pa", "Punjabi"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("sv", "Swedish"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("tl", "Filipino"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("xh", "Xhosa"),
    ("zh", "Chinese"),
    ("zh-Hans", "Simplified Chinese"),
    ("zh-Hant", "Traditional Chinese"),
];

/// The display name for a language tag, or the tag itself if unknown (e.g. `nn`).
pub fn display_name(tag: &str) -> &str {
    NAMES
        .iter()
        .find(|(t, _)| *t == tag)
        .map(|(_, name)| *name)
        .unwrap_or(tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_tags_map_to_names() {
        assert_eq!(display_name("es"), "Spanish");
        assert_eq!(display_name("en"), "English");
        assert_eq!(display_name("zh-Hans"), "Simplified Chinese");
        assert_eq!(display_name("zh-Hant"), "Traditional Chinese");
        assert_eq!(display_name("fa"), "Persian");
        assert_eq!(display_name("nn"), "Norwegian Nynorsk");
    }

    #[test]
    fn unknown_tag_falls_back_to_code() {
        assert_eq!(display_name("xx"), "xx");
        assert_eq!(display_name("qya"), "qya");
    }

    #[test]
    fn table_is_sorted_and_unique() {
        for w in NAMES.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "NAMES must be sorted/unique: {} !< {}",
                w[0].0,
                w[1].0
            );
        }
    }
}
