# app/utils/langutils.py

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# We'll load translation pipelines lazily the first time each language is requested.
EN_TO_MR = None
EN_TO_HI = None
EN_TO_ES = None

def detect_language(text: str) -> str:
    """
    Detect language code using langdetect.
    Returns short code like 'en', 'mr', 'hi', 'es', etc.
    """
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"


def _get_en_to_mr():
    global EN_TO_MR
    if EN_TO_MR is None:
        from transformers import pipeline
        EN_TO_MR = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mr")
    return EN_TO_MR


def _get_en_to_hi():
    global EN_TO_HI
    if EN_TO_HI is None:
        from transformers import pipeline
        EN_TO_HI = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
    return EN_TO_HI


def _get_en_to_es():
    global EN_TO_ES
    if EN_TO_ES is None:
        from transformers import pipeline
        EN_TO_ES = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    return EN_TO_ES


def _safe_translate(pipeline_getter, text: str) -> str:
    """
    Helper: run HF pipeline safely.
    pipeline_getter is a function that returns a pipeline (and loads it lazily).
    If anything fails, return original text.
    """
    try:
        pipe = pipeline_getter()
        out = pipe(text, max_length=512)
        return out[0]["translation_text"]
    except Exception:
        return text


def translate_text_if_needed(text: str, target_lang: str = "auto") -> str:
    """
    Translate English text to target_lang when needed.

    Assumptions:
    - The text passed in is English (suggestions / improved answer).
    - We only translate *from English* to:
        - 'mr' (Marathi)
        - 'hi' (Hindi)
        - 'es' (Spanish)
    - If translation isn't available or fails, we return the original English.
    """

    if not text:
        return text

    # If no specific target language, or English requested, just return original
    if target_lang in ("auto", "en", None, ""):
        return text

    if target_lang == "mr":
        return _safe_translate(_get_en_to_mr, text)

    if target_lang == "hi":
        return _safe_translate(_get_en_to_hi, text)

    if target_lang == "es":
        return _safe_translate(_get_en_to_es, text)

    # For any other languages, just return the original English
    return text
