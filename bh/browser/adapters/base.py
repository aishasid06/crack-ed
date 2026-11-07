"""Adapter interface and shared data shapes for marketplace pages."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page
from pydantic import BaseModel, Field

from bh.db import ExtractionMethod
from bh.errors import SelectorError

_CURRENCY = {"£": "GBP", "€": "EUR", "$": "USD"}
_MONEY = re.compile(r"(?P<sym>[£€$])?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<code>GBP|EUR|USD)?", re.I)


def parse_money(text: str | None) -> tuple[float | None, str | None]:
    if not text:
        return None, None
    m = _MONEY.search(text)
    if not m:
        return None, None
    value = float(m.group("num").replace(",", ""))
    cur = _CURRENCY.get(m.group("sym") or "") or (m.group("code") or "").upper() or None
    return value, cur


def first_int(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"\d[\d,]*", text)
    return int(m.group(0).replace(",", "")) if m else None


def clean(text: str | None) -> str | None:
    if text is None:
        return None
    t = re.sub(r"[ \t]+", " ", text).strip()
    return t or None


class Card(BaseModel):
    """One search-result entry."""

    site: str
    url: str
    external_id: str | None = None
    title: str
    price: float | None = None
    currency: str | None = None
    thumbnail_url: str | None = None
    seller_name: str | None = None
    seller_location: str | None = None


class ListingData(BaseModel):
    """Full listing extracted from its page."""

    site: str
    url: str
    external_id: str | None = None
    title: str
    description: str = ""
    price: float | None = None
    currency: str | None = None
    seller_name: str | None = None
    seller_location: str | None = None
    seller_age_days: int | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    quantity: int | None = None
    condition: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    extra: dict[str, str] = Field(default_factory=dict)
    extraction_method: ExtractionMethod = ExtractionMethod.adapter
    fallback_reason: str | None = None


Navigate = Callable[[Page, str], Awaitable[Any]]
Step = Callable[[str], AbstractAsyncContextManager[None]]


async def _plain_goto(page: Page, url: str) -> Any:
    return await page.goto(url, wait_until="domcontentloaded")


@asynccontextmanager
async def _no_step(name: str) -> AsyncIterator[None]:
    yield


class SiteAdapter(ABC):
    """Deterministic, selector-based access to one marketplace.

    ``goto`` and ``step`` are injected by the BrowserManager: every navigation and every in-page
    action (``async with self.step("click:load_more"): ...``) is paced per site and traced.
    """

    site: str

    def __init__(self, base_url: str, goto: Navigate | None = None, step: Step | None = None,
                 selector_timeout_ms: int = 3000) -> None:
        self.base_url = base_url.rstrip("/")
        self.goto: Navigate = goto or _plain_goto
        self.step: Step = step or _no_step
        self.timeout = selector_timeout_ms

    def owns(self, url: str) -> bool:
        b, u = urlparse(self.base_url), urlparse(url)
        return (b.scheme, b.netloc) == (u.scheme, u.netloc) and u.path.startswith(b.path)

    def absolute(self, page: Page, href: str | None) -> str | None:
        return urljoin(page.url, href) if href else None

    async def prepare(self, page: Page) -> None:
        """Make the page usable (e.g. dismiss a consent banner). Default: nothing."""

    @abstractmethod
    async def search(self, page: Page, query: str, page_no: int) -> None:
        """Navigate ``page`` to results page ``page_no`` (1-based) for ``query``."""

    @abstractmethod
    async def parse_results(self, page: Page) -> list[Card]:
        """Cards currently shown. Empty list for a genuine 'no results' page; SelectorError if unrecognised."""

    @abstractmethod
    async def parse_listing(self, page: Page) -> ListingData:
        """Parse the listing page currently open. SelectorError if required fields are missing."""

    async def has_next(self, page: Page) -> bool:
        return False

    async def iter_results(self, page: Page, query: str, max_pages: int) -> AsyncIterator[list[Card]]:
        """Yield cards page by page until there is no next page or ``max_pages`` is reached."""
        for n in range(1, max_pages + 1):
            await self.search(page, query, n)
            cards = await self.parse_results(page)
            if not cards:
                return
            yield cards
            if not await self.has_next(page):
                return

    async def wait_for_any(self, page: Page, selectors: list[str]) -> str:
        """Wait until one of ``selectors`` is attached; return it, or raise SelectorError."""
        joined = ", ".join(selectors)
        try:
            await page.wait_for_selector(joined, state="attached", timeout=self.timeout)
        except Exception as e:
            raise SelectorError(f"{self.site}: none of {selectors} found on {page.url}") from e
        for sel in selectors:
            if await page.locator(sel).count():
                return sel
        raise SelectorError(f"{self.site}: none of {selectors} found on {page.url}")
