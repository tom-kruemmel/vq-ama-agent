import re
import unicodedata

from langdetect import detect, LangDetectException

LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
}

# Maximum length for user questions (characters)
MAX_QUESTION_LENGTH = 2000


def detect_language(text: str) -> str:
    """Return a human-readable language name for *text* (defaults to English)."""
    try:
        code = detect(text)
    except LangDetectException:
        code = "en"
    return LANGUAGE_NAMES.get(code, "English")


def sanitize_user_input(text: str, max_length: int = MAX_QUESTION_LENGTH) -> str:
    """Sanitize user input before it enters LLM prompt templates.

    - Strips leading/trailing whitespace
    - Removes ASCII control characters (keeps newlines and tabs)
    - Normalizes Unicode to NFC (prevents homoglyph injection)
    - Truncates to *max_length* characters
    - Collapses attempts to fake prompt delimiters (e.g. lines of dashes/equals)
    """
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except \n and \t
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse sequences that look like prompt section delimiters
    # (e.g. "------", "======", "######") to a single instance
    text = re.sub(r'([=\-#~*]{5,})', '---', text)
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text
