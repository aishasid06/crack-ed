"""Search-query generation (SPEC §7.1 node 2).

Templates over the brand's own data: mark text, product names, deterministic misspellings of the
mark, generic product descriptors (to find copied photos sold without the mark) and
replica / inspired-by variants. An optional single small-model call can extend the list.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from bh.brand_index import BrandContext
from bh.config import AppConfig

_LEET = {"o": "0", "l": "1", "e": "3", "a": "4", "i": "1", "s": "5"}
_MAX_LLM_QUERIES = 6
_QUERY_OK = re.compile(r"^[\w][\w\s\-'&.:]{1,58}[\w.]$", re.UNICODE)


@dataclass(frozen=True)
class Query:
    text: str
    kind: str  # mark | product | misspelling | generic | replica | inspired | llm


def misspellings(word: str, n: int) -> list[str]:
    """Deterministic typo variants, round-robin over error types so a small ``n`` covers several:
    doubled consonant, transposition, leetspeak digit, hyphenation, dropped letter."""
    w = word.lower()
    vowels = set("aeiou")
    # doubling w[i]; consonants inside the word first ("vaello"), then vowels, then the first letter
    order = sorted(range(len(w)), key=lambda i: (i == 0, w[i] in vowels, i))
    doubled = [w[: i + 1] + w[i] + w[i + 1:] for i in order]
    kinds = [
        doubled,
        [w[:i] + w[i + 1] + w[i] + w[i + 2:] for i in range(1, len(w) - 1)],
        [w[:i] + _LEET[c] + w[i + 1:] for i, c in enumerate(w) if c in _LEET],
        [w[0] + "-" + w[1:]],
        [w[:i] + w[i + 1:] for i in range(1, len(w))],
    ]
    seen, out = {w}, []
    while len(out) < n and any(kinds):
        for k in kinds:
            while k:
                c = k.pop(0)
                if c not in seen and len(c) >= 3:
                    seen.add(c)
                    out.append(c)
                    break
            if len(out) >= n:
                break
    return out


def _strip_capacity(name: str) -> str:
    return re.sub(r"\b\d+(?:\.\d+)?\s*(?:ml|l|oz)\b", "", name, flags=re.I).strip().lower()


def template_queries(ctx: BrandContext, cfg: AppConfig, vocabulary: dict[str, list[str]]) -> list[Query]:
    marks = [m.lower() for m in ctx.marks] or [ctx.name.lower()]
    primary = marks[0]
    qs: list[Query] = [Query(m, "mark") for m in marks]
    if ctx.name.lower() not in marks:
        qs.append(Query(ctx.name.lower(), "mark"))
    generic = [Query(p.generic_name.lower(), "generic") for p in ctx.products if p.generic_name]
    typos = [Query(t, "misspelling") for t in misspellings(primary, cfg.discovery.misspelling_variants)]
    replica = [Query(f"{primary} {kw}", "replica") for kw in vocabulary.get("replica", [])[:2]]
    inspired = [Query(f"{primary} dupe", "inspired"), Query(f"inspired by {primary}", "inspired")]
    products = [Query(f"{primary} {_strip_capacity(p.short_name)}", "product") for p in ctx.products]
    # priority order matters: the cap drops from the end
    ordered = qs + generic + typos + replica + inspired + products
    return dedupe(ordered)[: cfg.discovery.max_queries]


def dedupe(qs: list[Query]) -> list[Query]:
    seen: set[str] = set()
    out = []
    for q in qs:
        key = re.sub(r"\s+", " ", q.text.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(Query(key, q.kind))
    return out


class _QueryList(BaseModel):
    queries: list[str] = Field(default_factory=list, description="Additional marketplace search queries")


QUERY_SYSTEM = (
    "You help a brand-protection team search online marketplaces for listings that may infringe a client's "
    "trade mark or copy its product photos. Given the brand's mark, its products and the queries already "
    "planned, propose a few additional short search queries (2-6 words) that a counterfeit, look-alike or "
    "photo-copying seller might use: plausible misspellings, generic product descriptions, and phrasing such "
    "as 'style' or 'dupe'. Do not repeat planned queries. Return JSON only.")


def sanitize(texts: list[str], existing: list[Query], limit: int = _MAX_LLM_QUERIES) -> list[Query]:
    have = {q.text for q in existing}
    out: list[Query] = []
    for raw in texts:
        if any(unicodedata.category(c).startswith("C") for c in str(raw)):
            continue  # control characters (newlines etc.) → reject rather than repair
        t = unicodedata.normalize("NFKC", str(raw)).strip().lower()
        t = re.sub(r"\s+", " ", t)
        if not _QUERY_OK.match(t) or len(t.split()) > 6 or t in have:
            continue
        have.add(t)
        out.append(Query(t, "llm"))
        if len(out) >= limit:
            break
    return out


async def generate_queries(ctx: BrandContext, cfg: AppConfig, vocabulary: dict[str, list[str]],
                           llm: Any | None = None) -> list[Query]:
    qs = template_queries(ctx, cfg, vocabulary)
    if llm is None or not cfg.llm.enable_query_extension:
        return qs
    prompt = {"mark": ctx.marks, "products": [p.short_name for p in ctx.products],
              "generic_descriptions": [p.generic_name for p in ctx.products if p.generic_name],
              "planned_queries": [q.text for q in qs]}
    import json

    res = await llm.complete_json(role="small", purpose="query_extension", schema=_QueryList, system=QUERY_SYSTEM,
                                  messages=[{"role": "user", "content": json.dumps(prompt)}])
    return qs + sanitize(res.parsed.queries, qs)
