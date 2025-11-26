# app/utils/textutils.py

import re

def split_sentences(text: str):
    """
    Very simple sentence splitter based on punctuation.
    This avoids NLTK data downloads and is enough for our prototype.
    """
    if not text:
        return []
    # Split on ., ?, ! followed by whitespace (or end of text)
    parts = re.split(r"[.!?]\s*", text)
    # Clean and drop empty strings
    return [p.strip() for p in parts if p.strip()]

def clean_text(s: str) -> str:
    """
    Basic text cleanup: strip and collapse whitespace.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s
