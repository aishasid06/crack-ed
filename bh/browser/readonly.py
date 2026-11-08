"""Safeguards for real marketplaces (SPEC §9 Phase 11 items 5 and 6).

A real marketplace is registered ``mode: read_only`` (bh/config.py makes that mandatory for any host outside
ENFORCEMENT_HOSTS and caps its concurrency, pages and pacing). On such a site:

- **Stop-list** — a query matching a real brand name in ``discovery.real_brand_stoplist`` (e.g. a misspelling
  of our fictional mark that is someone's real brand) is never sent.
- **robots.txt** — every URL the browser is about to open, and every page it lands on after an in-page action,
  must be allowed for ``browser.robots_user_agent`` (or ``*``). An unreachable or 5xx robots.txt means
  "disallow everything"; 401/403 too; 404 means no rules.
- No login, drafting or submission (bh/enforce/guard.py), and no case about a real listing is ever decided,
  drafted, approved or submitted (preflight + ``check_listing_target``).

Refusals raise ``ReadOnlyBlocked`` and are never worked around.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from bh.config import AppConfig
from bh.errors import ReadOnlyBlocked

log = logging.getLogger(__name__)

Fetch = Callable[[str], Awaitable[tuple[int, str]]]  # robots.txt URL -> (status, body); status 0 = unreachable


def is_read_only(cfg: AppConfig, site: str) -> bool:
    return site in cfg.sites and cfg.site(site).mode == "read_only"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def stoplist_hit(cfg: AppConfig, query: str) -> str | None:
    """The stop-listed real brand name this query contains (as whole words), if any."""
    q = f" {' '.join(_words(query))} "
    for name in cfg.discovery.real_brand_stoplist:
        n = " ".join(_words(name))
        if n and f" {n} " in q:
            return name
    return None


def query_refusal(cfg: AppConfig, site: str, query: str) -> str | None:
    if not is_read_only(cfg, site):
        return None
    hit = stoplist_hit(cfg, query)
    return f"query {query!r} matches real brand {hit!r} on the stop-list; not sent to {site}" if hit else None


def check_query(cfg: AppConfig, site: str, query: str) -> None:
    if (why := query_refusal(cfg, site, query)) is not None:
        log.warning("read-only guard: %s", why)
        raise ReadOnlyBlocked(why)


def site_hosts(cfg: AppConfig, site: str) -> set[str]:
    sc = cfg.site(site)
    return {(urlparse(sc.base_url).hostname or "").lower(), *(h.lower() for h in sc.extra_hosts)}


async def _http_fetch(url: str, user_agent: str) -> tuple[int, str]:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True,
                                     headers={"User-Agent": f"{user_agent} (read-only research)"}) as h:
            r = await h.get(url)
            return r.status_code, r.text
    except httpx.HTTPError:
        return 0, ""


class RobotsPolicy:
    """robots.txt per origin, fetched once per BrowserManager. Applies to read_only sites only."""

    def __init__(self, cfg: AppConfig, fetch: Fetch | None = None) -> None:
        self.cfg = cfg
        self.agent = cfg.browser.robots_user_agent
        self._fetch = fetch or (lambda u: _http_fetch(u, self.agent))
        self._parsers: dict[str, RobotFileParser] = {}
        self._lock = asyncio.Lock()

    async def _parser(self, origin: str) -> RobotFileParser:
        async with self._lock:
            if origin not in self._parsers:
                status, body = await self._fetch(f"{origin}/robots.txt")
                rp = RobotFileParser()
                if status in (401, 403) or status == 0 or status >= 500:
                    rp.disallow_all = True  # unreachable or refused: read nothing
                elif 400 <= status < 500:
                    rp.allow_all = True  # no robots.txt
                else:
                    rp.parse(body.splitlines())
                self._parsers[origin] = rp
                log.info("robots.txt %s: status %s", origin, status or "unreachable")
            return self._parsers[origin]

    async def allowed(self, site: str, url: str) -> bool:
        if not is_read_only(self.cfg, site):
            return True
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        rp = await self._parser(f"{u.scheme}://{u.netloc}")
        return rp.can_fetch(self.agent, url)

    async def check(self, site: str, url: str) -> None:
        if not await self.allowed(site, url):
            msg = f"robots.txt disallows {url} for {self.agent!r} on read_only site {site!r}"
            log.warning("read-only guard: %s", msg)
            raise ReadOnlyBlocked(msg)
