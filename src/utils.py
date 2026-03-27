from langdetect import detect, LangDetectException

LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
}


def detect_language(text: str) -> str:
    """Return a human-readable language name for *text* (defaults to English)."""
    try:
        code = detect(text)
    except LangDetectException:
        code = "en"
    return LANGUAGE_NAMES.get(code, "English")
