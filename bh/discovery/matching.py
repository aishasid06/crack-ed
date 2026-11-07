"""Fuzzy matching of listing text against the brand's marks (prefilter now, features in Phase 6)."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

_FOLD = str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ß": "ss", "đ": "d", "ł": "l"})
_DELEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_TOKEN = re.compile(r"[a-z0-9@$]+")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.translate(_FOLD))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"(?<=[a-z0-9])[-.·_](?=[a-z0-9])", "", t)  # "a-cme" → "acme"


def _tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(normalize(text)):
        if re.search(r"[a-z]", tok) and re.search(r"[0-9@$]", tok):
            tok = tok.translate(_DELEET)  # "acm3" → "acme", but leave "500ml" alone
        out.append(tok)
    return out


class BrandMatcher:
    def __init__(self, marks: list[str]) -> None:
        self.marks = sorted({normalize(m) for m in marks if m.strip()})

    def score(self, text: str) -> float:
        """0-100: best fuzzy ratio of any token (or adjacent pair, catching 'v aelo') against any mark."""
        toks = _tokens(text)
        grams = toks + [a + b for a, b in zip(toks, toks[1:], strict=False)]
        best = 0.0
        for m in self.marks:
            for g in grams:
                if abs(len(g) - len(m)) <= 3:
                    best = max(best, fuzz.ratio(g, m))
        return best

    def exact(self, text: str) -> bool:
        """The mark appears verbatim as a word (case-insensitive, accents folded, no leetspeak)."""
        words = set(re.findall(r"[a-z0-9]+", normalize(text)))
        return any(m in words for m in self.marks)
