"""EchoBazaar adapter: URL-driven search with numbered pages, blocking cookie banner."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from playwright.async_api import Page

from bh.browser.adapters.base import Card, ListingData, SiteAdapter, clean, first_int, parse_money
from bh.errors import SelectorError

_RESULTS_JS = """
() => Array.from(document.querySelectorAll('li.s-item')).map(li => ({
  href: li.querySelector('a.s-item__link')?.getAttribute('href') || null,
  title: li.querySelector('.s-item__title')?.textContent || null,
  price: li.querySelector('.s-item__price')?.textContent || null,
  thumb: li.querySelector('.s-item__image img')?.getAttribute('src') || null,
  seller: li.querySelector('.s-item__seller')?.textContent || null,
  location: li.querySelector('.s-item__location')?.textContent || null,
}))
"""

_LISTING_JS = """
() => {
  const t = s => document.querySelector(s)?.textContent ?? null;
  const imgs = Array.from(document.querySelectorAll('.vi-gallery__thumbs img, #vi-main-image'))
                    .map(i => i.getAttribute('src')).filter(Boolean);
  return {
    title: t('.x-item-title'), price: t('.x-price-primary'), condition: t('.x-item-condition'),
    quantity: t('.x-quantity'), item_number: t('.x-item-number'), seller: t('.x-seller-name'),
    feedback: t('.x-seller-feedback'), age: t('.x-seller-age'),
    location: t('.x-seller-location span'), description: t('.x-item-description'), images: imgs,
  };
}
"""


class EchoBazaarAdapter(SiteAdapter):
    site = "echobazaar"
    CONSENT = "#cookie-consent"

    async def prepare(self, page: Page) -> None:
        """Dismiss the cookie-consent overlay (choosing 'Essential only'); it blocks all clicks."""
        if await page.locator(self.CONSENT).count() == 0:
            return
        async with self.step("click:cookie_dismiss"):
            await page.click("#consent-essential", timeout=self.timeout)
            await page.wait_for_selector(self.CONSENT, state="detached", timeout=self.timeout)

    async def search(self, page: Page, query: str, page_no: int) -> None:
        await self.goto(page, f"{self.base_url}/search?q={quote(query)}&sort=relevance&page={page_no}")
        await self.prepare(page)

    async def parse_results(self, page: Page) -> list[Card]:
        found = await self.wait_for_any(page, ["li.s-item", ".srp-empty"])
        if found == ".srp-empty":
            return []
        cards: list[Card] = []
        for r in await page.evaluate(_RESULTS_JS):
            url, title = self.absolute(page, r["href"]), clean(r["title"])
            if not url or not title:
                raise SelectorError(f"echobazaar: result card without link/title on {page.url}")
            price, cur = parse_money(r["price"])
            loc = clean(r["location"])
            cards.append(Card(site=self.site, url=url, external_id=self._item_id(url), title=title, price=price,
                              currency=cur, thumbnail_url=self.absolute(page, r["thumb"]),
                              seller_name=clean(r["seller"]),
                              seller_location=re.sub(r"^From\s+", "", loc) if loc else None))
        return cards

    async def has_next(self, page: Page) -> bool:
        return await page.locator(".pagination__next").count() > 0

    async def parse_listing(self, page: Page) -> ListingData:
        await self.prepare(page)
        await self.wait_for_any(page, [".x-item-title"])
        d: dict[str, Any] = await page.evaluate(_LISTING_JS)
        title = clean(d["title"])
        price, cur = parse_money(d["price"])
        if not title or price is None:
            raise SelectorError(f"echobazaar: listing title/price missing on {page.url}")
        rating = None
        if d["feedback"] and (m := re.search(r"(\d+(?:\.\d+)?)\s*%", d["feedback"])):
            rating = round(float(m.group(1)) / 20.0, 2)  # % positive → 0-5 scale
        images = list(dict.fromkeys(self.absolute(page, s) for s in d["images"] if s))
        return ListingData(
            site=self.site, url=page.url, external_id=clean(d["item_number"]) or self._item_id(page.url),
            title=title, description=(d["description"] or "").strip(), price=price, currency=cur,
            seller_name=clean(d["seller"]), seller_location=clean(d["location"]),
            seller_age_days=first_int(d["age"]), rating=rating, quantity=first_int(d["quantity"]),
            condition=clean(d["condition"]), image_urls=[i for i in images if i],
            extra={"feedback": clean(d["feedback"]) or ""})

    @staticmethod
    def _item_id(url: str) -> str | None:
        m = re.search(r"/item/([^/?#]+)", url)
        return m.group(1) if m else None
