"""
Extract casing support information from CLDR casing XML files.

This script iterates over language files inside the CLDR `common/casing/` directory,
parses each `<casingData>` section in XML, and determines if casing rules are defined
for that language. Run this script directly to generate new data. Otherwise just import
it

CLDR:
    https://github.com/unicode-org/cldr/

Definition of casing behavior:
    https://github.com/unicode-org/cldr/blob/main/docs/ldml/tr35-general.md#case-mappings

Usage:
    python opuscleaner/scripts/build_casing_support.py --cldr-repo /path/to/cldr/common/casing
"""

import argparse
from enum import Enum
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from icu import Locale


class ScriptType(Enum):
    # A script where letters represent both consonants and vowels, written in sequence.
    # Examples: Latin, Greek, Cyrillic, Armenian, Georgian
    ALPHABETIC = "alphabetic"

    # A script that primarily represents consonants; vowels may be omitted or marked optionally.
    # Examples: Arabic, Hebrew
    ABJAD = "abjad"

    # A script where each character is a consonant-vowel unit, often with an inherent vowel.
    # Vowel signs are typically diacritics or marks that modify a base consonant and
    # combine with it visually.
    # Examples: Devanagari, Bengali, Tamil, Thai, Khmer, Ethiopic
    ABUGIDA = "abugida"

    # A script where each symbol represents an entire syllable.
    # Examples: Cherokee, Vai, Yi, Japanese Kana
    SYLLABARY = "syllabary"

    # A script where each symbol represents a word or morpheme, not individual sounds.
    # Examples: Chinese Han characters, Japanese Kanji
    LOGOGRAPHIC = "logographic"

    # A script where characters are built from shapes representing phonetic features.
    # Example: Hangul (Korean)
    FEATURAL = "featural"


scripts = {
    "Adlm": {
        # 𞤀𞤤𞤳𞤵𞤤𞤫 𞤁𞤢𞤲𞤣𞤢𞤴𞤯𞤫 𞤂𞤫𞤻𞤮𞤤 𞤃𞤵𞤤𞤵𞤺𞤮𞤤
        "name": "Adlam",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Arab": {
        # العربية
        "name": "Arabic",
        "type": "phonographic",
        "bicameral": ScriptType.ABJAD,
    },
    "Armn": {
        # Հայերեն
        "name": "Armenian",
        "type": "phonographic",
        "bicameral": ScriptType.ALPHABETIC,
    },
    "Beng": {
        # বাংলা
        "name": "Bengali",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Cans": {
        # ᓀᐦᐃᔭᐍᐏᐣ
        "name": "Canadian Aboriginal",
        "type": ScriptType.SYLLABARY,
        "bicameral": False,
    },
    "Cher": {
        # ᏣᎳᎩ
        "name": "Cherokee",
        "type": ScriptType.SYLLABARY,
        "bicameral": True,
    },
    "Cyrl": {
        # кириллица
        "name": "Cyrillic",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Deva": {
        # देवनागरी
        "name": "Devanagari",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Dsrt": {
        # 𐐓𐐲𐑌𐐮𐐻𐐰𐑉
        "name": "Deseret",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Ethi": {
        # አማርኛ
        "name": "Ethiopic",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Geor": {
        # ქართული
        "name": "Georgian",
        "type": ScriptType.ALPHABETIC,
        "bicameral": False,  # Generally unicameral, but has bicameral scripts.
    },
    "Grek": {
        # Ελληνικά
        "name": "Greek",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Gujr": {
        # ગુજરાતી
        "name": "Gujarati",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Guru": {
        # ਗੁਰਮੁਖੀ
        "name": "Gurmukhi",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Hans": {
        # 简体字
        "name": "Han (Simplified)",
        "type": ScriptType.LOGOGRAPHIC,
        "bicameral": False,
    },
    "Hant": {
        # 繁體字
        "name": "Han (Traditional)",
        "type": ScriptType.LOGOGRAPHIC,
        "bicameral": False,
    },
    "Hebr": {
        # עברית
        "name": "Hebrew",
        "type": ScriptType.ABJAD,
        "bicameral": False,
    },
    "Jpan": {
        # 日本語
        "name": "Japanese",
        "type": ScriptType.LOGOGRAPHIC,
        "bicameral": False,
    },
    "Knda": {
        # ಕನ್ನಡ
        "name": "Kannada",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Khmr": {
        # ភាសាខ្មែរ
        "name": "Khmer",
        "type": ScriptType.ABUGIDA,
        # Khmr has subscript variations, but they encode semantic meanings, so probably
        # shouldn't be augmented for casing, as it would change the meaning of the text.
        "bicameral": False,
    },
    "Kore": {
        # 한국어 / 조선말
        "name": "Korean",
        # Hangul is technically phonographic, as the character blocks encode phonemic
        # meaning by forming syllable blocks, rather than ideographic meaning, as in
        # Chinese.
        "type": ScriptType.FEATURAL,
        "bicameral": False,
    },
    "Laoo": {
        # ລາວ
        "name": "Lao",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Latn": {
        # Latin
        "name": "Latin",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Mlym": {
        # മലയാളം
        "name": "Malayalam",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Mymr": {
        # မြန်မာစာ
        "name": "Myanmar",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Orya": {
        # ଓଡ଼ିଆ
        "name": "Odia",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Osge": {
        # 𐒰𐒻𐒼𐒰𐒿𐒷
        "name": "Osage",
        "type": ScriptType.ALPHABETIC,
        "bicameral": True,
    },
    "Sinh": {
        # සිංහල අක්ෂර මාලාව
        "name": "Sinhala",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Taml": {
        # தமிழ்
        "name": "Tamil",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Telu": {
        # తెలుగు
        "name": "Telugu",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Tfng": {
        # ⵜⴰⵎⴰⵣⵉⵖⵜ
        "name": "Tifinagh",
        "type": ScriptType.ALPHABETIC,
        "bicameral": False,
    },
    "Thai": {
        # ภาษาไทย
        "name": "Thai",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Tibt": {
        # བོད་ཡིག
        "name": "Tibetan",
        "type": ScriptType.ABUGIDA,
        "bicameral": False,
    },
    "Vaii": {
        # ꕉꕜꕮ ꔔꘋ ꖸ ꔰ
        "name": "Vai",
        "type": ScriptType.SYLLABARY,
        "bicameral": False,
    },
    "Yiii": {
        # ꆈꌠꁱꂷ
        "name": "Yi",
        "type": ScriptType.SYLLABARY,
        "bicameral": False,
    },
}


def extract_casing_support(cldr_repo: Path) -> None:
    casing_dir = cldr_repo / "common/casing"
    files = list(casing_dir.iterdir())
    files.sort()
    scripts = set()

    print("CASING = {")
    for lang_file in files:
        if not lang_file.is_file() or lang_file.suffix != ".xml":
            continue

        language = lang_file.stem

        try:
            tree = ET.parse(lang_file)
        except ET.ParseError as e:
            print(f"XML parse error in {lang_file}: {e}")
            continue

        root = tree.getroot()
        has_casing = root.find(".//casingData") is not None

        try:
            icu_locale = Locale(language)
            maximized = Locale.addLikelySubtags(icu_locale)
            script = maximized.getScript() or "None"
        except Exception:
            script = "Error getting script"

        scripts.add(script)

        display_name = Locale(language).getDisplayName()

        print(f'    "{language}": {{')
        print(f'        "display_name": {json.dumps(display_name)},')
        print(f'        "script": {json.dumps(script)},')
        print(f'        "has_casing": {has_casing},')
        print("    },")

    print("}")

    print("!!! scripts", scripts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract casing support info from CLDR XML data.")
    parser.add_argument(
        "--cldr-repo", type=str, required=True, help="Path to the cldr/common/casing directory."
    )
    args = parser.parse_args()

    cldr_repo = Path(args.cldr_repo)
    if not cldr_repo.exists():
        raise FileNotFoundError(f"Could not find the CLDR repo: {cldr_repo}")

    extract_casing_support(cldr_repo)


if __name__ == "__main__":
    main()

CASING = {
    # Afar
    "aa": True,
    # Afrikaans
    "af": True,
    # Aghem
    "agq": True,
    # Akan
    "ak": True,
    # Amharic
    "am": False,
    # Aragonese
    "an": True,
    # Arabic
    "ar": False,
    # Assamese
    "as": False,
    # Asu
    "asa": True,
    # Asturian
    "ast": True,
    # Azerbaijani
    "az": True,
    # Azerbaijani (Cyrillic)
    "az_Cyrl": True,
    # Baluchi
    "bal": False,
    # Baluchi (Latin)
    "bal_Latn": True,
    # Basaa
    "bas": True,
    # Belarusian
    "be": True,
    # Bemba
    "bem": True,
    # Bena
    "bez": True,
    # Bulgarian
    "bg": True,
    # Bambara
    "bm": True,
    # Bangla
    "bn": False,
    # Tibetan
    "bo": False,
    # Breton
    "br": True,
    # Bodo
    "brx": False,
    # Bosnian
    "bs": True,
    # Bosnian (Cyrillic)
    "bs_Cyrl": True,
    # Blin
    "byn": True,
    # Catalan
    "ca": True,
    # Cebuano
    "ceb": True,
    # Chiga
    "cgg": True,
    # Cherokee
    "chr": False,
    # Chickasaw
    "cic": True,
    # Czech
    "cs": True,
    # Welsh
    "cy": True,
    # Danish
    "da": True,
    # Taita
    "dav": True,
    # German
    "de": True,
    # Zarma
    "dje": True,
    # Lower Sorbian
    "dsb": True,
    # Duala
    "dua": True,
    # Jola-Fonyi
    "dyo": True,
    # Dzongkha
    "dz": True,
    # Embu
    "ebu": True,
    # Ewe
    "ee": True,
    # Greek
    "el": True,
    # English
    "en": True,
    # English (Deseret)
    "en_Dsrt": True,
    # English (United States)
    "en_US": False,
    # English (United States, Computer)
    "en_US_POSIX": True,
    # Esperanto
    "eo": True,
    # Spanish
    "es": True,
    # Estonian
    "et": True,
    # Basque
    "eu": True,
    # Ewondo
    "ewo": True,
    # Persian
    "fa": False,
    # Fulah
    "ff": True,
    # Fulah (Adlam)
    "ff_Adlm": True,
    # Finnish
    "fi": True,
    # Filipino
    "fil": True,
    # Faroese
    "fo": True,
    # French
    "fr": True,
    # Friulian
    "fur": True,
    # Western Frisian
    "fy": True,
    # Irish
    "ga": True,
    # Scottish Gaelic
    "gd": True,
    # Galician
    "gl": True,
    # Swiss German
    "gsw": True,
    # Gujarati
    "gu": True,
    # Gusii
    "guz": True,
    # Manx
    "gv": True,
    # Hausa
    "ha": True,
    # Hausa (Ghana)
    "ha_GH": True,
    # Hausa (Niger)
    "ha_NE": True,
    # Hawaiian
    "haw": True,
    # Hebrew
    "he": False,
    # Hindi
    "hi": False,
    # Croatian
    "hr": True,
    # Upper Sorbian
    "hsb": True,
    # Hungarian
    "hu": True,
    # Armenian
    "hy": True,
    # Interlingua
    "ia": True,
    # Indonesian
    "id": True,
    # Igbo
    "ig": True,
    # Sichuan Yi
    "ii": False,
    # Icelandic
    "is": True,
    # Italian
    "it": True,
    # Inuktitut
    "iu": False,
    # Inuktitut (Latin)
    "iu_Latn": False,
    # Japanese
    "ja": True,
    # Ngomba
    "jgo": True,
    # Machame
    "jmc": True,
    # Georgian
    "ka": True,
    # Kabyle
    "kab": True,
    # Kamba
    "kam": True,
    # Makonde
    "kde": True,
    # Kabuverdianu
    "kea": True,
    # Koyra Chiini
    "khq": True,
    # Kikuyu
    "ki": True,
    # Kazakh
    "kk": True,
    # Kazakh (Arabic)
    "kk_Arab": False,
    # Kako
    "kkj": True,
    # Kalaallisut
    "kl": True,
    # Kalenjin
    "kln": True,
    # Khmer
    "km": True,
    # Kannada
    "kn": True,
    # Korean
    "ko": True,
    # Konkani
    "kok": True,
    # Kpelle
    "kpe": False,
    # Kashmiri
    "ks": False,
    # Kashmiri (Devanagari)
    "ks_Deva": False,
    # Shambala
    "ksb": True,
    # Bafia
    "ksf": True,
    # Colognian
    "ksh": True,
    # Cornish
    "kw": True,
    # Kyrgyz
    "ky": True,
    # Langi
    "lag": True,
    # Luxembourgish
    "lb": True,
    # Ganda
    "lg": True,
    # Ligurian
    "lij": True,
    # Lakota
    "lkt": True,
    # Lingala
    "ln": True,
    # Lao
    "lo": True,
    # Lithuanian
    "lt": True,
    # Luba-Katanga
    "lu": True,
    # Luo
    "luo": True,
    # Luyia
    "luy": True,
    # Latvian
    "lv": True,
    # Masai
    "mas": True,
    # Meru
    "mer": True,
    # Morisyen
    "mfe": True,
    # Malagasy
    "mg": True,
    # Makhuwa-Meetto
    "mgh": True,
    # Metaʼ
    "mgo": True,
    # Macedonian
    "mk": True,
    # Malayalam
    "ml": True,
    # Mongolian
    "mn": True,
    # Marathi
    "mr": False,
    # Malay
    "ms": True,
    # Malay (Brunei)
    "ms_BN": True,
    # Malay (Singapore)
    "ms_SG": True,
    # Maltese
    "mt": True,
    # Mundang
    "mua": True,
    # Muscogee
    "mus": True,
    # Burmese
    "my": False,
    # Erzya
    "myv": True,
    # Nama
    "naq": True,
    # Norwegian Bokmål
    "nb": False,
    # North Ndebele
    "nd": True,
    # Low German
    "nds": True,
    # Nepali
    "ne": False,
    # Dutch
    "nl": True,
    # Kwasio
    "nmg": True,
    # Norwegian Nynorsk
    "nn": True,
    # Ngiemboon
    "nnh": True,
    # Norwegian
    "no": True,
    # South Ndebele
    "nr": True,
    # Northern Sotho
    "nso": True,
    # Nuer
    "nus": True,
    # Nyankole
    "nyn": True,
    # Occitan
    "oc": True,
    # Oromo
    "om": True,
    # Odia
    "or": True,
    # Ossetic
    "os": True,
    # Osage
    "osa": True,
    # Punjabi
    "pa": True,
    # Punjabi (Arabic)
    "pa_Arab": False,
    # Polish
    "pl": True,
    # Prussian
    "prg": True,
    # Pashto
    "ps": False,
    # Portuguese
    "pt": True,
    # Portuguese (Portugal)
    "pt_PT": True,
    # Quechua
    "qu": True,
    # Romansh
    "rm": True,
    # Rundi
    "rn": True,
    # Romanian
    "ro": True,
    # Rombo
    "rof": True,
    # Russian
    "ru": True,
    # Kinyarwanda
    "rw": True,
    # Rwa
    "rwk": True,
    # Sakha
    "sah": True,
    # Samburu
    "saq": True,
    # Sangu
    "sbp": True,
    # Sardinian
    "sc": True,
    # Sicilian
    "scn": True,
    # Northern Sami
    "se": True,
    # Sena
    "seh": True,
    # Koyraboro Senni
    "ses": True,
    # Sango
    "sg": True,
    # Tachelhit
    "shi": False,
    # Tachelhit (Latin)
    "shi_Latn": True,
    # Sinhala
    "si": False,
    # Slovak
    "sk": True,
    # Slovenian
    "sl": True,
    # Southern Sami
    "sma": True,
    # Lule Sami
    "smj": True,
    # Shona
    "sn": True,
    # Somali
    "so": True,
    # Albanian
    "sq": True,
    # Serbian
    "sr": True,
    # Serbian (Cyrillic)
    "sr_Cyrl": False,
    # Serbian (Cyrillic, Bosnia & Herzegovina)
    "sr_Cyrl_BA": True,
    # Serbian (Cyrillic, Montenegro)
    "sr_Cyrl_ME": True,
    # Serbian (Cyrillic, Kosovo)
    "sr_Cyrl_XK": True,
    # Serbian (Latin)
    "sr_Latn": True,
    # Swati
    "ss": True,
    # Saho
    "ssy": True,
    # Southern Sotho
    "st": True,
    # Swedish
    "sv": True,
    # Swahili
    "sw": True,
    # Silesian
    "szl": True,
    # Tamil
    "ta": True,
    # Telugu
    "te": False,
    # Teso
    "teo": True,
    # Tajik
    "tg": True,
    # Thai
    "th": False,
    # Tigrinya
    "ti": True,
    # Tigre
    "tig": True,
    # Turkmen
    "tk": True,
    # Tswana
    "tn": True,
    # Tongan
    "to": True,
    # Tok Pisin
    "tpi": True,
    # Turkish
    "tr": True,
    # Tsonga
    "ts": True,
    # Tasawaq
    "twq": True,
    # Central Atlas Tamazight
    "tzm": True,
    # Uyghur
    "ug": False,
    # Ukrainian
    "uk": True,
    # Urdu
    "ur": True,
    # Uzbek
    "uz": True,
    # Uzbek (Arabic)
    "uz_Arab": False,
    # Uzbek (Cyrillic)
    "uz_Cyrl": True,
    # Vai
    "vai": False,
    # Vai (Latin)
    "vai_Latn": True,
    # Venda
    "ve": True,
    # Vietnamese
    "vi": True,
    # Volapük
    "vo": True,
    # Vunjo
    "vun": True,
    # Walser
    "wae": True,
    # Wolaytta
    "wal": True,
    # Xhosa
    "xh": True,
    # Soga
    "xog": True,
    # Yangben
    "yav": True,
    # Yoruba
    "yo": True,
    # Standard Moroccan Tamazight
    "zgh": False,
    # Chinese
    "zh": False,
    # Chinese (Simplified)
    "zh_Hans": False,
    # Chinese (Simplified, Hong Kong SAR China)
    "zh_Hans_HK": True,
    # Chinese (Simplified, Macao SAR China)
    "zh_Hans_MO": True,
    # Chinese (Simplified, Singapore)
    "zh_Hans_SG": True,
    # Chinese (Traditional)
    "zh_Hant": False,
    # Zulu
    "zu": True,
}
