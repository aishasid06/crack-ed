"""Site adapter registry. Sites configured with ``adapter: null`` use the generic extractor."""

from __future__ import annotations

from bh.browser.adapters.base import Card, ListingData, SiteAdapter
from bh.browser.adapters.mockbay import MockBayAdapter
from bh.browser.adapters.tradeloft import TradeLoftAdapter

ADAPTERS: dict[str, type[SiteAdapter]] = {"mockbay": MockBayAdapter, "tradeloft": TradeLoftAdapter}

__all__ = ["ADAPTERS", "Card", "ListingData", "SiteAdapter", "MockBayAdapter", "TradeLoftAdapter"]
