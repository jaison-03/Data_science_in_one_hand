"""Lightweight multilingual language detection and display names."""

from langdetect import DetectorFactory, LangDetectException, detect_langs


DetectorFactory.seed = 0

LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu",
    "kn": "Kannada",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ar": "Arabic",
    "ja": "Japanese",
    "zh-cn": "Chinese",
}


def detect_language(text):
    """Return a best-effort language and detector score, or an unknown result."""
    normalized = str(text).strip()
    if len(normalized) < 4:
        return {"code": None, "name": "Unknown", "confidence": None}
    try:
        candidates = detect_langs(normalized)
    except LangDetectException:
        return {"code": None, "name": "Unknown", "confidence": None}
    if not candidates:
        return {"code": None, "name": "Unknown", "confidence": None}
    best = candidates[0]
    code = best.lang.lower()
    return {
        "code": code,
        "name": LANGUAGES.get(code, f"Other ({code})"),
        "confidence": round(float(best.prob), 3),
    }


def supported_language_names():
    return list(LANGUAGES.values())
