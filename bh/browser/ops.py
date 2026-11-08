"""Browser operations with the SPEC §7.2 failure policy:

    adapter → 1 retry with a fresh page → generic LLM extractor → ExtractionFailed

``HumanRequired`` is never retried or worked around; it propagates so the task pauses.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any
from urllib.parse import quote

from playwright.async_api import Page

from bh.browser.adapters.base import Card, ListingData
from bh.browser.generic_extract import extract_listing, extract_results
from bh.browser.session import BrowserManager
from bh.errors import ExtractionFailed, SelectorError

log = logging.getLogger(__name__)


async def _screenshot(browser: BrowserManager, site: str, page: Page, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    async with browser.action(site, "screenshot"):
        await page.screenshot(path=str(path), full_page=browser.cfg.discovery.screenshot_full_page)


async def capture_listing(browser: BrowserManager, site: str, url: str, llm: Any | None,
                          screenshot_path: Path | None = None) -> ListingData:
    """Extract a listing; optionally save a screenshot of the page the data came from."""
    adapter = browser.adapter(site)
    if adapter is None and browser.cfg.site(site).navigator:
        return await _capture_with_navigator(browser, site, url, llm, screenshot_path)
    attempts = 1 + browser.cfg.browser.retries if adapter is not None else 1
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        async with browser.page(site) as page:
            await browser.goto(site, page, url)
            if adapter is not None:
                try:
                    await adapter.prepare(page)
                    data = await adapter.parse_listing(page)
                    await _screenshot(browser, site, page, screenshot_path)
                    return data
                except SelectorError as e:
                    last = e
                    log.warning("adapter %s failed on %s (attempt %d/%d): %s", site, url, attempt, attempts, e)
                    if attempt < attempts:
                        continue
            if llm is None:
                raise ExtractionFailed(f"{site}: adapter failed and no LLM for fallback on {url}: {last}") from last
            reason = f"adapter failed: {last}" if last else "no adapter for site"
            data = await extract_listing(page, llm, site, reason=reason, cfg=browser.cfg)
            await _screenshot(browser, site, page, screenshot_path)
            return data
    raise ExtractionFailed(f"{site}: could not extract {url}")  # pragma: no cover - loop always returns/raises


async def _capture_with_navigator(browser: BrowserManager, site: str, url: str, llm: Any | None,
                                  screenshot_path: Path | None) -> ListingData:
    """Recipe replay (no LLM) → agent loop (which re-learns the recipe on success)."""
    from bh.browser.agent_nav import Navigator, NavOutcome
    from bh.browser.recipes import RecipeStore, Replayer, ReplayFailed

    store = RecipeStore(browser.cfg)
    nav = Navigator(browser, site, llm, store=store)
    recipe = store.load(site)
    async with browser.page(site) as page:
        data: ListingData | None = None
        if recipe is not None and recipe.listing is not None:
            try:
                data = await Replayer(nav, recipe).listing(page, url, NavOutcome())
            except ReplayFailed as e:
                log.warning("recipe v%d replay failed on %s: %s; using the agent loop", recipe.version, url, e)
        if data is None:
            if llm is None:
                raise ExtractionFailed(f"{site}: the agentic navigator needs an LLM (run with --llm)")
            data = await nav.listing(page, url)
        await browser.goto(site, page, url)  # evidence screenshot of the listing as first shown
        await _screenshot(browser, site, page, screenshot_path)
        return data


async def iter_search_pages(browser: BrowserManager, site: str, query: str, max_pages: int | None = None,
                            llm: Any | None = None) -> AsyncIterator[list[Card]]:
    """Yield result cards one page (or "Load more" batch) at a time, up to the hard page cap.

    Uses the site adapter; falls back to the generic extractor if the site has no adapter or the
    adapter's selectors fail before any page was read. Callers may stop early (``aclosing`` + break).
    """
    from bh.browser.readonly import check_query

    check_query(browser.cfg, site, query)  # real-brand stop-list (read_only sites)
    cap = min(max_pages or browser.cfg.site(site).max_pages, browser.cfg.site(site).max_pages)
    adapter = browser.adapter(site)
    if adapter is not None:
        yielded = 0
        try:
            async with browser.page(site) as page, aclosing(adapter.iter_results(page, query, cap)) as it:
                async for batch in it:
                    yielded += 1
                    yield batch
            return
        except SelectorError as e:
            if yielded:
                log.warning("adapter %s failed after %d page(s) for %r: %s", site, yielded, query, e)
                return
            log.warning("adapter %s search failed: %s; using generic extractor", site, e)
            if llm is None:
                raise ExtractionFailed(f"{site}: search selectors failed and no LLM for fallback: {e}") from e
    if browser.cfg.site(site).navigator:  # discovery order: recipe replay → agent loop → generic extractor
        from bh.browser.agent_nav import NavigationFailed, Navigator, NavOutcome
        from bh.browser.recipes import RecipeStore, Replayer, ReplayFailed

        store = RecipeStore(browser.cfg)
        nav = Navigator(browser, site, llm, store=store)
        recipe = store.load(site)
        if recipe is not None and recipe.search is not None:
            yielded = 0
            try:
                async with browser.page(site) as page, \
                        aclosing(Replayer(nav, recipe).search(page, query, cap, NavOutcome())) as it:
                    async for batch in it:
                        yielded += 1
                        yield batch
                return
            except ReplayFailed as e:
                if yielded:
                    log.warning("recipe v%d replay stopped on %s after %d page(s): %s", recipe.version, site,
                                yielded, e)
                    return
                log.warning("recipe v%d replay failed on %s for %r: %s; using the agent loop", recipe.version,
                            site, query, e)
        if llm is None:
            raise ExtractionFailed(f"{site}: no adapter and no LLM for the generic extractor")
        yielded = 0
        try:
            async with browser.page(site) as page, aclosing(nav.search(page, query, cap)) as it:
                async for batch in it:
                    yielded += 1
                    yield batch
            return
        except NavigationFailed as e:
            if yielded or not browser.cfg.site(site).search_url:
                raise
            log.warning("navigator failed on %s for %r: %s; using generic extractor", site, query, e)
    if llm is None:
        raise ExtractionFailed(f"{site}: no adapter and no LLM for the generic extractor")
    template = browser.cfg.site(site).search_url
    if not template:
        raise ExtractionFailed(f"{site}: no adapter and no search_url template configured")
    url: str | None = template.format(base=browser.cfg.site(site).base_url.rstrip("/"), query=quote(query), page=1)
    async with browser.page(site) as page:
        for _ in range(cap):
            if url is None:
                return
            await browser.goto(site, page, url)
            cards, url = await extract_results(page, llm, site, cfg=browser.cfg)
            if not cards:
                return
            yield cards


async def search_site(browser: BrowserManager, site: str, query: str, max_pages: int | None = None,
                      llm: Any | None = None) -> list[Card]:
    """All cards for ``query`` up to the page cap (deduplicated by URL)."""
    out: dict[str, Card] = {}
    async with aclosing(iter_search_pages(browser, site, query, max_pages, llm)) as it:
        async for batch in it:
            for c in batch:
                out.setdefault(c.url, c)
    return list(out.values())
