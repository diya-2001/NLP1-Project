"""Pronoun scrubbing (De-Arteaga et al., 2019): delete explicit gender words from a bio.

We remove gendered pronouns and titles. We do NOT remove first names (the dataset has no name
annotations), so some gender signal remains: this is a partial scrub, and we say so.
Note: removing "ms" also removes the degree "MS"; a small, accepted side effect.
"""
import re

GENDER_WORDS = ["he", "she", "him", "her", "his", "hers", "himself", "herself", "mr", "mrs", "ms"]
_PATTERN = re.compile(r"\b(?:" + "|".join(GENDER_WORDS) + r")\b\.?", flags=re.IGNORECASE)


def scrub(text: str) -> str:
    """'She is a nurse. Her patients…' -> ' is a nurse.  patients…'"""
    return _PATTERN.sub("", text)
