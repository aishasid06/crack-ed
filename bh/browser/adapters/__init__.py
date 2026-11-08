"""Site adapter registry. Sites configured with ``adapter: null`` use the generic extractor."""

from __future__ import annotations

from bh.browser.adapters.base import Card, ListingData, SiteAdapter
from bh.browser.adapters.echobazaar import EchoBazaarAdapter
from bh.browser.adapters.tradeloft import TradeLoftAdapter

ADAPTERS: dict[str, type[SiteAdapter]] = {"echobazaar": EchoBazaarAdapter, "tradeloft": TradeLoftAdapter}

__all__ = ["ADAPTERS", "Card", "ListingData", "SiteAdapter", "EchoBazaarAdapter", "TradeLoftAdapter"]
