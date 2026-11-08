"""Browser sessions (SPEC §7.2).

- One persistent Chromium context per site under ``<sessions_dir>/<site>/`` (cookies such as a
  dismissed consent banner or a portal login survive between runs).
- Per-site concurrency limits (asyncio semaphores) and polite pacing (``min_delay_ms``).
- Every navigation is checked for a challenge page; a match raises ``HumanRequired``. The agent
  never interacts with challenge pages — a human completes them (per-IP release on the mock portal).
- ``login(site)`` types credentials from ``.env`` via code. They never pass through an LLM and are
  never logged.
- When a run is active, each navigation is recorded as a trace span.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from bh.browser.adapters import ADAPTERS, SiteAdapter
from bh.config import AppConfig, ChallengeConfig, Secrets, get_config, get_secrets
from bh.errors import HumanRequired, LoginFailed
from bh.trace import current_run_id, task_span

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoginForm:
    """Where and how to log in. Selectors only — credentials are supplied from Secrets at call time."""

    login_path: str
    username: str
    password: str
    submit: str
    error: str
    probe_path: str = "/"  # a page that redirects to login when the session is missing


# Sites the agent may log into. Only our own mock portal is listed (SPEC §1).
LOGIN_FORMS: dict[str, LoginForm] = {
    "portal": LoginForm(login_path="/login", username="#username", password="#password",
                        submit="#login-submit", error=".pt-error"),
}


async def detect_challenge(page: Page, markers: ChallengeConfig) -> None:
    """Raise HumanRequired if the page looks like a human-verification challenge."""
    path = urlparse(page.url).path.lower()
    reason = None
    if any(p.lower() in path for p in markers.url_patterns):
        reason = f"challenge URL {path}"
    else:
        for sel in markers.selectors:
            if await page.locator(sel).count():
                reason = f"challenge element {sel}"
                break
    if reason is None and markers.title_patterns:
        title = (await page.title()).lower()
        if any(t.lower() in title for t in markers.title_patterns):
            reason = f"challenge title {title!r}"
    if reason:
        raise HumanRequired(f"Human verification required ({reason}); complete it in your browser", url=page.url)


class SitePacer:
    """Minimum delay between page-changing actions on one site (polite crawling, not evasion)."""

    def __init__(self, min_delay_ms: int) -> None:
        self.delay = min_delay_ms / 1000.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if (sleep := self._last + self.delay - now) > 0:
                await asyncio.sleep(sleep)
            self._last = time.monotonic()


class BrowserManager:
    def __init__(self, cfg: AppConfig | None = None, secrets: Secrets | None = None,
                 sessions_dir: Path | None = None, headless: bool | None = None) -> None:
        self.cfg = cfg or get_config()
        self.secrets = secrets or get_secrets()
        self.sessions_dir = sessions_dir or self.cfg.paths.resolve(self.cfg.paths.sessions_dir)
        self.headless = self.cfg.browser.headless if headless is None else headless
        self._pw: Playwright | None = None
        self._contexts: dict[str, BrowserContext] = {}
        self._ctx_lock = asyncio.Lock()
        self._sems = {s: asyncio.Semaphore(c.max_concurrency) for s, c in self.cfg.sites.items()}
        self._pacers = {s: SitePacer(c.min_delay_ms) for s, c in self.cfg.sites.items()}
        self._active: dict[str, int] = {s: 0 for s in self.cfg.sites}
        self.max_active: dict[str, int] = {s: 0 for s in self.cfg.sites}
        self._nav_seq = itertools.count(1)

    async def __aenter__(self) -> "BrowserManager":
        self._pw = await async_playwright().start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        for ctx in self._contexts.values():
            await ctx.close()
        self._contexts.clear()
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    # ------------------------------------------------------------ contexts / pages

    def site_url(self, site: str, path: str = "") -> str:
        return self.cfg.site(site).base_url.rstrip("/") + path

    async def context(self, site: str) -> BrowserContext:
        self.cfg.site(site)  # KeyError for unconfigured sites
        async with self._ctx_lock:
            if site not in self._contexts:
                if self._pw is None:
                    raise RuntimeError("BrowserManager must be used as 'async with'")
                user_dir = self.sessions_dir / site
                user_dir.mkdir(parents=True, exist_ok=True)
                ctx = await self._pw.chromium.launch_persistent_context(
                    str(user_dir), headless=self.headless, viewport={"width": 1280, "height": 900})
                ctx.set_default_navigation_timeout(self.cfg.browser.navigation_timeout_ms)
                ctx.set_default_timeout(self.cfg.browser.navigation_timeout_ms)
                self._contexts[site] = ctx
            return self._contexts[site]

    @asynccontextmanager
    async def page(self, site: str) -> AsyncIterator[Page]:
        """A fresh page on the site's context, within the site's concurrency limit."""
        async with self._sems[site]:
            ctx = await self.context(site)
            page = await ctx.new_page()
            self._active[site] += 1
            self.max_active[site] = max(self.max_active[site], self._active[site])
            try:
                yield page
            finally:
                self._active[site] -= 1
                await page.close()

    def pacer(self, site: str) -> SitePacer:
        return self._pacers[site]

    # ------------------------------------------------------------ navigation

    async def goto(self, site: str, page: Page, url: str) -> Any:
        """Paced, traced navigation followed by challenge detection."""

        async def _go() -> Any:
            await self._pacers[site].wait()
            resp = await page.goto(url, wait_until="domcontentloaded")
            await detect_challenge(page, self.cfg.browser.challenge)
            return resp

        if current_run_id() is None:
            return await _go()
        async with task_span(None, "browser:goto", site, f"goto:{next(self._nav_seq)}:{url}",
                             payload={"url": url}):
            return await _go()

    @asynccontextmanager
    async def action(self, site: str, name: str) -> AsyncIterator[None]:
        """Paced, traced in-page action (click, form submit, Load more, cookie dismiss, login submit)."""
        await self._pacers[site].wait()
        if current_run_id() is None:
            yield
            return
        async with task_span(None, "browser:action", site, f"action:{next(self._nav_seq)}:{name}",
                             payload={"action": name}):
            yield

    def adapter(self, site: str) -> SiteAdapter | None:
        name = self.cfg.site(site).adapter
        if name is None:
            return None
        cls = ADAPTERS.get(name)
        if cls is None:
            raise KeyError(f"site {site!r} names unknown adapter {name!r}")

        async def nav(page: Page, url: str) -> Any:
            return await self.goto(site, page, url)

        def step(name: str) -> Any:
            return self.action(site, name)

        return cls(self.cfg.site(site).base_url, goto=nav, step=step,
                   selector_timeout_ms=self.cfg.browser.selector_timeout_ms)

    # ------------------------------------------------------------ login

    async def login(self, site: str, page: Page) -> None:
        """Ensure ``page``'s context is logged into ``site``. Raises HumanRequired on a challenge."""
        from bh.enforce.guard import check_site

        check_site(self.cfg, site)  # login means typing credentials: allowlisted, non-read_only hosts only
        form = LOGIN_FORMS.get(site)
        if form is None:
            raise KeyError(f"no login form configured for site {site!r}")

        async def _login() -> None:
            await self.goto(site, page, self.site_url(site, form.probe_path))
            if not urlparse(page.url).path.rstrip("/").endswith(form.login_path.rstrip("/")):
                return  # session cookie still valid
            username, password = self.secrets.require_portal()
            await page.fill(form.username, username)
            await page.fill(form.password, password)  # typed by code; never logged or sent to an LLM
            async with self.action(site, "submit:login"), page.expect_navigation(wait_until="domcontentloaded"):
                await page.click(form.submit)
            await detect_challenge(page, self.cfg.browser.challenge)
            if urlparse(page.url).path.rstrip("/").endswith(form.login_path.rstrip("/")):
                msg = await page.locator(form.error).first.text_content() if await page.locator(form.error).count() \
                    else "still on login page"
                reason = re.sub(r"\s+", " ", msg or "").strip()
                raise LoginFailed(f"login to {site} failed: {reason}")
            log.info("logged into %s", site)

        if current_run_id() is None:
            await _login()
        else:
            async with task_span(None, "browser:login", site, f"login:{next(self._nav_seq)}:{site}"):
                await _login()
