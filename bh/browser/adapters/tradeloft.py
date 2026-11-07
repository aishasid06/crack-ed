"""TradeLoft adapter: form-based search with filters and a "Load more" button."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from playwright.async_api import Page

from bh.browser.adapters.base import Card, ListingData, SiteAdapter, clean, first_int, parse_money
from bh.errors import SelectorError

_ROWS_JS = """
() => Array.from(document.querySelectorAll('#tl-results article.tl-row')).map(r => ({
  href: r.querySelector('.tl-row-title a')?.getAttribute('href') || null,
  title: r.querySelector('.tl-row-title')?.textContent || null,
  price: r.querySelector('.tl-row-price')?.textContent || null,
  thumb: r.querySelector('.tl-row-thumb img')?.getAttribute('src') || null,
  seller: r.querySelector('.tl-row-seller')?.textContent || null,
}))
"""

_LISTING_JS = """
() => {
  const t = s => document.querySelector(s)?.textContent ?? null;
  const specs = {};
  document.querySelectorAll('table.tl-specs tr').forEach(tr => {
    const k = tr.querySelector('th')?.textContent?.trim(); const v = tr.querySelector('td')?.textContent?.trim();
    if (k) specs[k] = v ?? '';
  });
  return {
    title: t('.tl-listing-title'), price: t('.tl-sidebar .tl-price'), specs,
    seller: t('.tl-seller-card .tl-seller-name'), rating: t('.tl-seller-card .tl-seller-rating'),
    location: t('.tl-seller-card .tl-seller-loc'), tenure: t('.tl-seller-card .tl-seller-tenure'),
    about: t('.tl-about'),
    images: Array.from(document.querySelectorAll('.tl-carousel .tl-photo img')).map(i => i.getAttribute('src')),
  };
}
"""

_SELLER_ROW = re.compile(r"Sold by\s+(?P<name>.+?)\s*\((?P<loc>[^)]*)\)\s*$")


class TradeLoftAdapter(SiteAdapter):
    site = "tradeloft"
    LOAD_MORE = "#tl-load-more"

    async def submit_search(self, page: Page, query: str, category: str | None = None,
                            min_price: float | None = None, max_price: float | None = None) -> None:
        await self.goto(page, f"{self.base_url}/")
        await self.wait_for_any(page, ["#tl-search-form"])
        await page.fill("#tl-q", query)
        if category:
            await page.select_option("#tl-category", label=category)
        if min_price is not None:
            await page.fill("#tl-min", f"{min_price:g}")
        if max_price is not None:
            await page.fill("#tl-max", f"{max_price:g}")
        async with self.step("submit:search_form"), \
                page.expect_navigation(wait_until="domcontentloaded", timeout=self.timeout * 3):
            await page.click("#tl-search-submit")

    async def load_more(self, page: Page) -> None:
        before = await page.locator("#tl-results article.tl-row").count()
        async with self.step("click:load_more"):
            await page.click(self.LOAD_MORE, timeout=self.timeout)
            try:
                await page.wait_for_function(
                    "([sel, n]) => document.querySelectorAll('#tl-results article.tl-row').length > n"
                    " || document.querySelector(sel)?.hidden", arg=[self.LOAD_MORE, before],
                    timeout=self.timeout * 3)
            except Exception as e:
                raise SelectorError(f"tradeloft: 'Load more' did not add results on {page.url}") from e

    async def search(self, page: Page, query: str, page_no: int) -> None:
        await self.submit_search(page, query)
        for _ in range(page_no - 1):
            if not await self.has_next(page):
                break
            await self.load_more(page)

    async def iter_results(self, page: Page, query: str, max_pages: int) -> AsyncIterator[list[Card]]:
        """Submit once, then click 'Load more' until it is hidden or ``max_pages`` batches were read."""
        await self.submit_search(page, query)
        seen = 0
        for n in range(1, max_pages + 1):
            cards = await self.parse_results(page)
            new, seen = cards[seen:], len(cards)
            if new:
                yield new
            if n == max_pages or not new or not await self.has_next(page):
                return
            await self.load_more(page)

    async def has_next(self, page: Page) -> bool:
        btn = page.locator(self.LOAD_MORE)
        return await btn.count() > 0 and await btn.is_visible()

    async def parse_results(self, page: Page) -> list[Card]:
        found = await self.wait_for_any(page, ["#tl-results article.tl-row", ".tl-empty"])
        if found == ".tl-empty":
            return []
        cards: list[Card] = []
        for r in await page.evaluate(_ROWS_JS):
            url, title = self.absolute(page, r["href"]), clean(r["title"])
            if not url or not title:
                raise SelectorError(f"tradeloft: result row without link/title on {page.url}")
            price, cur = parse_money(r["price"])
            seller = loc = None
            if r["seller"] and (m := _SELLER_ROW.search(clean(r["seller"]) or "")):
                seller, loc = m.group("name"), m.group("loc")
            cards.append(Card(site=self.site, url=url, external_id=self._listing_id(url), title=title, price=price,
                              currency=cur, thumbnail_url=self.absolute(page, r["thumb"]), seller_name=seller,
                              seller_location=loc))
        return cards

    async def parse_listing(self, page: Page) -> ListingData:
        await self.wait_for_any(page, [".tl-listing-title"])
        d: dict[str, Any] = await page.evaluate(_LISTING_JS)
        title = clean(d["title"])
        price, cur = parse_money(d["price"])
        if not title or price is None:
            raise SelectorError(f"tradeloft: listing title/price missing on {page.url}")
        specs: dict[str, str] = {k: clean(v) or "" for k, v in d["specs"].items()}
        rating = None
        if d["rating"] and (m := re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", d["rating"])):
            rating = float(m.group(1))
        loc = clean(d["location"])
        tenure = re.search(r"(\d[\d,]*)\s+days", d["tenure"] or "")
        images = list(dict.fromkeys(self.absolute(page, s) for s in d["images"] if s))
        return ListingData(
            site=self.site, url=page.url, external_id=specs.get("Listing ID") or self._listing_id(page.url),
            title=title, description=(d["about"] or "").strip(), price=price, currency=cur,
            seller_name=clean(d["seller"]), seller_location=re.sub(r"^Based in\s+", "", loc) if loc else None,
            seller_age_days=int(tenure.group(1).replace(",", "")) if tenure else None, rating=rating,
            quantity=first_int(specs.get("Units in stock")), condition=specs.get("Condition") or None,
            image_urls=[i for i in images if i], extra=specs)

    @staticmethod
    def _listing_id(url: str) -> str | None:
        m = re.search(r"/listing/([^/?#]+)", url)
        return m.group(1) if m else None
