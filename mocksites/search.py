"""Keyword relevance for the mock marketplaces (token match with light fuzziness,
like a real marketplace tolerating typos)."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from mocksites.catalog import MockListing

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def score(listing: MockListing, query: str) -> float:
    q = tokens(query)
    if not q:
        return 1.0
    title = tokens(listing.title)
    body = tokens(listing.description)
    total = 0.0
    for qt in q:
        best = 0.0
        for weight, doc in ((2.0, title), (1.0, body)):
            for dt in doc:
                if dt == qt:
                    best = max(best, weight)
                elif len(qt) >= 4 and fuzz.ratio(qt, dt) >= 80:
                    best = max(best, weight * 0.6)
        total += best
    return total


def search(listings: list[MockListing], query: str, sort: str = "relevance", *,
           min_price: float | None = None, max_price: float | None = None,
           site_category: str | None = None) -> list[MockListing]:
    scored = []
    for l in listings:
        if min_price is not None and l.price < min_price:
            continue
        if max_price is not None and l.price > max_price:
            continue
        if site_category and l.site_category != site_category:
            continue
        s = score(l, query)
        if s > 0:
            scored.append((s, l))
    if sort == "price_asc":
        scored.sort(key=lambda t: (t[1].price, t[1].id))
    elif sort == "price_desc":
        scored.sort(key=lambda t: (-t[1].price, t[1].id))
    else:
        scored.sort(key=lambda t: (-t[0], t[1].id))
    return [l for _, l in scored]
