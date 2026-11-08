"""Browser layer + adapters against live mock sites (SPEC §7.2, Phase 4)."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from pydantic import SecretStr
from sqlmodel import Session, select

from bh.browser.adapters.base import parse_money
from bh.browser.generic_extract import ExtractedListing, ExtractedResults, snapshot
from bh.browser.ops import capture_listing, search_site
from bh.browser.session import BrowserManager, detect_challenge
from bh.config import AppConfig, Secrets, load_config
from bh.db import ExtractionMethod, LLMCall, Task, TaskStatus
from bh.errors import ExtractionFailed, HumanRequired, LoginFailed, SelectorError
from bh.llm import LLMClient
from bh.trace import resume_task, run_span, task_span
from mocksites.app import MockSettings, create_app
from mocksites.catalog import Catalog, load_catalog
from mocksites.search import search as catalog_search
from tests.test_mocksites import ColourEmbedder

USER, PASS = "lawyer@vaelo.example", "browser-test-pass-9"


# ---------------------------------------------------------------- live server fixtures


@dataclass
class LiveServer:
    base: str
    app: Any
    catalog: Catalog
    requests: list[tuple[str, str]] = field(default_factory=list)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def seeded_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from seed import generate as G

    base = tmp_path_factory.mktemp("bseed")
    G.generate(base / "brand", base / "other", base / "catalog", base / "labels.json", seed=7)
    return base / "catalog"


def _start(tmp: Path, catalog_dir: Path, **kw: Any) -> Iterator[LiveServer]:
    settings = MockSettings(catalog_dir=catalog_dir, db_url=f"sqlite:///{tmp / 'mock.db'}", portal_user=USER,
                            portal_pass=PASS, uploads_dir=tmp / "uploads", challenge_probability=0.0,
                            challenge_rapid_requests=10_000, embedder=ColourEmbedder(), seed=3)
    for k, v in kw.items():
        setattr(settings, k, v)
    app = create_app(settings)
    port = _free_port()
    live = LiveServer(f"http://127.0.0.1:{port}", app, load_catalog(catalog_dir))

    @app.middleware("http")
    async def _record(request: Any, call_next: Any) -> Any:
        live.requests.append((request.method, request.url.path))
        return await call_next(request)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "mock server did not start"
    try:
        yield live
    finally:
        server.should_exit = True
        t.join(timeout=5)


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory, seeded_catalog: Path) -> Iterator[LiveServer]:
    yield from _start(tmp_path_factory.mktemp("srv"), seeded_catalog)


@pytest.fixture
def challenge_live(tmp_path: Path, seeded_catalog: Path) -> Iterator[LiveServer]:
    yield from _start(tmp_path, seeded_catalog, challenge_probability=1.0)


def _cfg_for(base: str, delay_ms: int = 0, **site_updates: Any) -> AppConfig:
    cfg = load_config()
    sites = {}
    for name, sc in cfg.sites.items():
        upd: dict[str, Any] = {"base_url": f"{base}/{name}", "min_delay_ms": delay_ms}
        upd.update(site_updates.get(name, {}))
        sites[name] = sc.model_copy(update=upd)
    browser = cfg.browser.model_copy(update={"selector_timeout_ms": 1500})
    return cfg.model_copy(update={"sites": sites, "browser": browser})


@pytest.fixture
def secrets() -> Secrets:  # overrides conftest: credentials matching the live portal
    return Secrets(mock_portal_user=SecretStr(USER), mock_portal_pass=SecretStr(PASS),
                   anthropic_api_key=SecretStr("sk-ant-test-0123456789abcdef"))


def _browser(live: LiveServer, tmp_path: Path, secrets: Secrets, **kw: Any) -> BrowserManager:
    return BrowserManager(_cfg_for(live.base, **kw), secrets, sessions_dir=tmp_path / "sessions", headless=True)


# ---------------------------------------------------------------- pure helpers


@pytest.mark.parametrize("text,expected", [("£1,234.56", (1234.56, "GBP")), ("€9", (9.0, "EUR")),
                                           ("12.50 USD", (12.5, "USD")), ("free", (None, None))])
def test_parse_money(text: str, expected: tuple[float | None, str | None]) -> None:
    assert parse_money(text) == expected


# ---------------------------------------------------------------- EchoBazaar


async def test_echobazaar_search_dismisses_banner_and_paginates(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    expected = catalog_search(live.catalog.for_site("echobazaar"), "vaelo")
    async with _browser(live, tmp_path, secrets) as b:
        ad = b.adapter("echobazaar")
        async with b.page("echobazaar") as page:
            await ad.search(page, "vaelo", 1)
            assert await page.locator("#cookie-consent").count() == 0  # dismissed
            p1 = await ad.parse_results(page)
            assert len(p1) == 12 and await ad.has_next(page)
        cards = await search_site(b, "echobazaar", "vaelo", max_pages=10)
    assert len(cards) == len(expected)
    by_id = {l.id: l for l in expected}
    for c in cards:
        l = by_id[c.external_id]
        assert c.title == l.title and c.price == l.price and c.currency == "GBP"
        assert c.seller_name == l.seller_name and c.seller_location == l.seller_location
        assert c.url.endswith(f"/echobazaar/item/{l.id}") and c.thumbnail_url.endswith(l.images[0])


async def test_echobazaar_page_cap(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets) as b:
        cards = await search_site(b, "echobazaar", "vaelo", max_pages=2)
    assert len(cards) == 24


async def test_echobazaar_no_results(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets) as b:
        assert await search_site(b, "echobazaar", "zzzqqqxxx") == []


async def test_echobazaar_listing_matches_catalog(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    sample = live.catalog.for_site("echobazaar")[:8]
    async with _browser(live, tmp_path, secrets) as b:
        for l in sample:
            d = await capture_listing(b, "echobazaar", f"{live.base}/echobazaar/item/{l.id}", llm=None)
            assert d.extraction_method == ExtractionMethod.adapter
            assert (d.external_id, d.title, d.price, d.currency) == (l.id, l.title, l.price, "GBP")
            assert d.description == l.description.strip()
            assert (d.seller_name, d.seller_location, d.seller_age_days, d.quantity, d.condition) == \
                (l.seller_name, l.seller_location, l.seller_age_days, l.quantity, l.condition)
            assert d.rating == pytest.approx(round(round(l.seller_rating / 5 * 100) / 20, 2))
            assert [u.rsplit("/", 1)[1] for u in d.image_urls] == l.images


# ---------------------------------------------------------------- TradeLoft


async def test_tradeloft_load_more_until_hidden(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    expected = catalog_search(live.catalog.for_site("tradeloft"), "vaelo")
    assert len(expected) > 16, "fixture should need several Load more clicks"
    async with _browser(live, tmp_path, secrets) as b:
        cards = await search_site(b, "tradeloft", "vaelo", max_pages=50)
        assert len(cards) == len(expected)
        assert {c.external_id for c in cards} == {l.id for l in expected}
        # button hidden at the end
        ad = b.adapter("tradeloft")
        async with b.page("tradeloft") as page:
            batches = [x async for x in ad.iter_results(page, "vaelo", 50)]
            assert not await ad.has_next(page)
    assert [len(x) for x in batches][:-1] == [8] * (len(batches) - 1)
    c0 = next(c for c in cards if c.external_id == expected[0].id)
    assert (c0.title, c0.price, c0.seller_name, c0.seller_location) == \
        (expected[0].title, expected[0].price, expected[0].seller_name, expected[0].seller_location)


async def test_tradeloft_page_cap(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets) as b:
        cards = await search_site(b, "tradeloft", "vaelo", max_pages=2)
    assert len(cards) == 16


async def test_tradeloft_filters_via_form(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets) as b:
        ad = b.adapter("tradeloft")
        async with b.page("tradeloft") as page:
            await ad.submit_search(page, "vaelo", max_price=15)
            cards = await ad.parse_results(page)
    assert cards and all(c.price <= 15 for c in cards)


async def test_tradeloft_listing_matches_catalog(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    sample = live.catalog.for_site("tradeloft")[:8]
    async with _browser(live, tmp_path, secrets) as b:
        for l in sample:
            d = await capture_listing(b, "tradeloft", f"{live.base}/tradeloft/listing/{l.id}", llm=None)
            assert (d.external_id, d.title, d.price) == (l.id, l.title, l.price)
            assert (d.seller_name, d.seller_location, d.seller_age_days, d.rating, d.quantity, d.condition) == \
                (l.seller_name, l.seller_location, l.seller_age_days, l.seller_rating, l.quantity, l.condition)
            assert d.description == l.description.strip()
            assert d.extra["Category"] == l.site_category and d.extra.get("Brand") == l.specs.get("Brand")
            assert [u.rsplit("/", 1)[1] for u in d.image_urls] == l.images


async def test_adapter_raises_selector_error_on_wrong_page(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets) as b:
        async with b.page("tradeloft") as page:
            await b.goto("tradeloft", page, f"{live.base}/tradeloft/")
            with pytest.raises(SelectorError):
                await b.adapter("tradeloft").parse_listing(page)


# ---------------------------------------------------------------- concurrency, pacing, persistence


async def test_per_site_concurrency_limit(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    urls = [f"{live.base}/echobazaar/item/{l.id}" for l in live.catalog.for_site("echobazaar")[:6]]
    async with _browser(live, tmp_path, secrets) as b:
        limit = b.cfg.site("echobazaar").max_concurrency
        await asyncio.gather(*(capture_listing(b, "echobazaar", u, None) for u in urls))
        assert b.max_active["echobazaar"] == limit


async def test_pacing_spaces_navigations(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets, delay_ms=300) as b:
        async with b.page("tradeloft") as page:
            t0 = time.monotonic()
            for _ in range(3):
                await b.goto("tradeloft", page, f"{live.base}/tradeloft/")
            assert time.monotonic() - t0 >= 0.6


async def test_consent_persists_in_site_session(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets) as b:
        async with b.page("echobazaar") as page:
            await b.goto("echobazaar", page, f"{live.base}/echobazaar/")
            await b.adapter("echobazaar").prepare(page)
    async with _browser(live, tmp_path, secrets) as b:  # new browser, same sessions dir
        async with b.page("echobazaar") as page:
            await b.goto("echobazaar", page, f"{live.base}/echobazaar/")
            assert await page.locator("#cookie-consent").count() == 0
    assert (tmp_path / "sessions" / "echobazaar").is_dir()


# ---------------------------------------------------------------- portal login + challenge


async def test_portal_login_with_env_credentials(live: LiveServer, tmp_path: Path, secrets: Secrets,
                                                 caplog: pytest.LogCaptureFixture, engine: Any) -> None:
    caplog.set_level(logging.DEBUG)
    async with _browser(live, tmp_path, secrets) as b:
        async with b.page("portal") as page:
            await b.login("portal", page)
            assert page.url.rstrip("/").endswith("/portal")
            assert "Start a new IP report" in await page.content()
            await b.login("portal", page)  # already logged in: no form submission
    assert PASS not in caplog.text
    with Session(engine) as s:
        assert list(s.exec(select(LLMCall))) == []  # credentials never go near the LLM
    assert sum(1 for m, p in live.requests if m == "POST" and p == "/portal/login") >= 1


async def test_portal_login_wrong_password(live: LiveServer, tmp_path: Path) -> None:
    bad = Secrets(mock_portal_user=SecretStr(USER), mock_portal_pass=SecretStr("wrong-pass-123"))
    async with BrowserManager(_cfg_for(live.base), bad, sessions_dir=tmp_path / "s", headless=True) as b:
        async with b.page("portal") as page:
            with pytest.raises(LoginFailed) as ei:
                await b.login("portal", page)
    assert "wrong-pass-123" not in str(ei.value)


async def test_challenge_pauses_task_and_human_release_resumes(challenge_live: LiveServer, tmp_path: Path,
                                                               secrets: Secrets, engine: Any) -> None:
    live = challenge_live
    async with run_span(None, {}) as run, _browser(live, tmp_path, secrets) as b:
        async with b.page("portal") as page:
            with pytest.raises(HumanRequired) as ei:
                async with task_span(run.id, "draft", "portal", "draft:case-1"):
                    await b.login("portal", page)
            assert "/portal/verify" in ei.value.url
            with Session(engine) as s:
                task = s.exec(select(Task).where(Task.dedup_key == "draft:case-1")).one()
            assert task.status == TaskStatus.waiting_human and "/portal/verify" in task.waiting_url

            # A human completes the check in their own browser (same machine → same IP).
            async with httpx.AsyncClient(base_url=live.base) as human:
                r = await human.post("/portal/verify", data={"confirm_human": "yes", "next": "/portal/"})
                assert r.status_code == 303

            resume_task(task.id)  # UI "Done" button
            async with task_span(run.id, "draft", "portal", "draft:case-1") as again:
                assert again.attempts == 2
                await b.login("portal", page)
            assert "Start a new IP report" in await page.content()

    with Session(engine) as s:
        assert s.get(Task, task.id).status == TaskStatus.succeeded
    # the agent itself never submitted the verification form
    agent_verify_posts = [r for r in live.requests if r == ("POST", "/portal/verify")]
    assert len(agent_verify_posts) == 1  # exactly the human's


async def test_detect_challenge_markers(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    async with _browser(live, tmp_path, secrets) as b:
        markers = b.cfg.browser.challenge
        async with b.page("portal") as page:
            await page.set_content("<html><head><title>Shop</title></head><body><h1>ok</h1></body></html>")
            await detect_challenge(page, markers)  # no raise
            await page.set_content("<title>Security check</title><p>hi</p>")
            with pytest.raises(HumanRequired):
                await detect_challenge(page, markers)
            await page.set_content("<div id='human-verification'>Please verify</div>")
            with pytest.raises(HumanRequired):
                await detect_challenge(page, markers)


# ---------------------------------------------------------------- generic extractor on an unknown layout

UNSEEN = "http://unseen-market.test"


def _unseen_listing_html(img_base: str) -> str:
    """A listing page in a layout no adapter knows (definition lists, no shared class names)."""
    return f"""<!doctype html><html><head><title>Hydra steel bottle | Bazaarly</title></head><body>
<nav><a href="/">Bazaarly</a> › <a href="/c/drinkware">Drinkware</a></nav>
<article>
  <header><h2>Hydra double-wall steel bottle, 600ml</h2><span>Item code BZ-7731</span></header>
  <section aria-label="Gallery">
    <img src="{img_base}/a.jpg" alt="Bottle front" width="300" height="300">
    <img src="{img_base}/b.jpg" alt="Bottle side" width="300" height="300">
  </section>
  <dl>
    <dt>Asking price</dt><dd>GBP 17.25</dd>
    <dt>State</dt><dd>Brand new</dd>
    <dt>In stock</dt><dd>40 units</dd>
  </dl>
  <div role="region" aria-label="About the trader">
    <p>Trader: <strong>northwind_goods</strong></p><p>Ships from Porto, Portugal</p>
    <p>On Bazaarly since 2 years · 96% happy buyers</p>
  </div>
  <p>Keeps drinks cold for a full day. Screw lid, powder coat.</p>
</article></body></html>"""


def _unseen_results_html() -> str:
    rows = "".join(f'<div><a href="/p/{i}">Steel bottle no. {i}</a> <span>£{10 + i}.00</span></div>' for i in range(5))
    return f"<html><head><title>Results | Bazaarly</title></head><body><h1>Results</h1>{rows}" \
           f'<a href="/search?q=bottle&pg=2">Next page</a></body></html>'


class FakeAnthropic:
    """Fake SDK client: answers by reading the snapshot it was sent (exercises the real LLMClient)."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=self)

    async def parse(self, **kw: Any) -> Any:
        self.requests.append(kw)
        text = kw["messages"][0]["content"]
        if kw["output_format"] is ExtractedListing:
            imgs = [u for u in text.split('"') if u.startswith("http") and u.endswith(".jpg")]
            parsed = ExtractedListing(
                is_product_listing="Hydra double-wall" in text, title="Hydra double-wall steel bottle, 600ml",
                description="Keeps drinks cold for a full day. Screw lid, powder coat.", price=17.25, currency="gbp",
                seller_name="northwind_goods", seller_location="Porto, Portugal", seller_age_days=730,
                seller_rating_out_of_5=4.8, quantity_available=40, condition="Brand new",
                image_urls=imgs + ["http://evil.example/made-up.jpg"])
        else:
            parsed = ExtractedResults(is_results_page=True, cards=[
                {"url": f"/p/{i}", "title": f"Steel bottle no. {i}", "price": 10 + i, "currency": "GBP"}
                for i in range(5)] + [{"url": "/p/999", "title": "hallucinated", "price": 1}],
                next_page_url="/search?q=bottle&pg=2")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=parsed.model_dump_json())],
                               usage=SimpleNamespace(input_tokens=900, output_tokens=120, cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0),
                               model="claude-haiku-5-5", stop_reason="end_turn", parsed_output=parsed)


async def _route_unseen(page: Any, live: LiveServer) -> list[str]:
    """Serve the unknown-layout site via request interception; returns a log of document loads."""
    img = f"{live.base}/media/{live.catalog.listings[0].images[0]}"
    hits: list[str] = []

    async def handler(route: Any) -> None:
        path = route.request.url[len(UNSEEN):]
        if route.request.resource_type == "document":
            hits.append(path)
        if path.startswith("/item/"):
            await route.fulfill(status=200, content_type="text/html",
                                body=_unseen_listing_html(img.rsplit("/", 1)[0]).replace("/a.jpg", "/" + img.rsplit("/", 1)[1]))
        elif path.startswith("/search"):
            await route.fulfill(status=200, content_type="text/html", body=_unseen_results_html())
        else:
            await route.fulfill(status=404, body="")

    await page.context.route(f"{UNSEEN}/**", handler)
    return hits


def _unseen_cfg(live: LiveServer) -> AppConfig:
    cfg = _cfg_for(live.base)
    unseen = cfg.sites["echobazaar"].model_copy(update={"base_url": UNSEEN, "adapter": None,
                                                     "search_url": "{base}/search?q={query}&pg={page}"})
    return cfg.model_copy(update={"sites": {**cfg.sites, "unseen": unseen}})


async def test_generic_extractor_on_unknown_layout(live: LiveServer, tmp_path: Path, secrets: Secrets,
                                                   engine: Any) -> None:
    cfg = _unseen_cfg(live)
    fake = FakeAnthropic()
    llm = LLMClient(cfg=cfg, secrets=secrets, client=fake, engine=engine)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        async with b.page("unseen") as page:
            await _route_unseen(page, live)
        async with task_span(run.id, "capture", "unseen", "capture:unseen-1"):
            # route is registered on the context, so capture_listing's fresh page sees it
            d = await capture_listing(b, "unseen", f"{UNSEEN}/item/7731", llm)
    assert d.extraction_method == ExtractionMethod.generic_llm and d.fallback_reason == "no adapter for site"
    assert (d.title, d.price, d.currency, d.seller_name, d.quantity) == \
        ("Hydra double-wall steel bottle, 600ml", 17.25, "GBP", "northwind_goods", 40)
    assert d.image_urls and all(u.startswith(live.base) for u in d.image_urls)  # made-up URL dropped
    req = fake.requests[0]
    sent = req["messages"][0]["content"]
    assert "Hydra double-wall steel bottle" in sent and "GBP 17.25" in sent and "northwind_goods" in sent
    assert "<dl>" not in sent and "<div" not in sent  # snapshot, not raw HTML
    assert req["model"] == "claude-haiku-5-5"
    with Session(engine) as s:
        (call,) = s.exec(select(LLMCall)).all()
        task = s.exec(select(Task).where(Task.dedup_key == "capture:unseen-1")).one()
    assert call.task_id == task.id and call.purpose == "generic_extract_listing:unseen" and call.input_tokens == 900


async def test_generic_results_extractor_validates_links(live: LiveServer, tmp_path: Path, secrets: Secrets,
                                                         engine: Any) -> None:
    cfg = _unseen_cfg(live)
    cfg = cfg.model_copy(update={"sites": {**cfg.sites, "unseen": cfg.sites["unseen"].model_copy(
        update={"max_pages": 1})}})
    llm = LLMClient(cfg=cfg, secrets=secrets, client=FakeAnthropic(), engine=engine)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        async with b.page("unseen") as page:
            await _route_unseen(page, live)
        cards = await search_site(b, "unseen", "bottle", llm=llm)
    assert [c.url for c in cards] == [f"{UNSEEN}/p/{i}" for i in range(5)]  # /p/999 not on the page → dropped


async def test_adapter_failure_retries_then_falls_back(live: LiveServer, tmp_path: Path, secrets: Secrets,
                                                       engine: Any) -> None:
    """EchoBazaar adapter pointed at a page it cannot parse: 1 retry with a fresh page, then generic."""
    cfg = _cfg_for(live.base, echobazaar={"base_url": UNSEEN})
    fake = FakeAnthropic()
    llm = LLMClient(cfg=cfg, secrets=secrets, client=fake, engine=engine)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        async with b.page("echobazaar") as page:
            hits = await _route_unseen(page, live)
        d = await capture_listing(b, "echobazaar", f"{UNSEEN}/item/7731", llm)
        assert d.extraction_method == ExtractionMethod.generic_llm
        assert d.fallback_reason.startswith("adapter failed:")
        assert hits == ["/item/7731", "/item/7731"]  # first attempt + 1 retry on a fresh page, then fallback
        assert len(fake.requests) == 1
        with pytest.raises(ExtractionFailed):
            await capture_listing(b, "echobazaar", f"{UNSEEN}/item/7731", llm=None)
        assert len(hits) == 4 and len(fake.requests) == 1  # no LLM → fails after retry, no extra call


async def test_generic_extractor_refuses_non_listing(live: LiveServer, tmp_path: Path, secrets: Secrets,
                                                     engine: Any) -> None:
    cfg = _unseen_cfg(live)
    llm = LLMClient(cfg=cfg, secrets=secrets, client=FakeAnthropic(), engine=engine)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        async with b.page("unseen") as page:
            await _route_unseen(page, live)
        with pytest.raises(ExtractionFailed):
            await capture_listing(b, "unseen", f"{UNSEEN}/search?q=x", llm)


async def test_snapshot_is_bounded(live: LiveServer, tmp_path: Path, secrets: Secrets) -> None:
    cfg = _cfg_for(live.base)
    cfg = cfg.model_copy(update={"browser": cfg.browser.model_copy(update={"snapshot_max_chars": 600})})
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        async with b.page("echobazaar") as page:
            await b.goto("echobazaar", page, f"{live.base}/echobazaar/search?q=vaelo")
            snap = await snapshot(page, cfg)
    assert len(snap.aria) <= 600 and snap.truncated and snap.image_candidates


@pytest.mark.skipif(not os.environ.get("BH_LIVE_LLM"), reason="set BH_LIVE_LLM=1 (and Anthropic credentials) "
                                                              "to run the real small model")
async def test_generic_extractor_live_model(live: LiveServer, tmp_path: Path, engine: Any) -> None:
    from bh.config import load_secrets

    cfg = _unseen_cfg(live)
    llm = LLMClient(cfg=cfg, secrets=load_secrets(), engine=engine)
    async with BrowserManager(cfg, load_secrets(), sessions_dir=tmp_path / "s") as b:
        async with b.page("unseen") as page:
            await _route_unseen(page, live)
        d = await capture_listing(b, "unseen", f"{UNSEEN}/item/7731", llm)
    assert d.price == pytest.approx(17.25) and d.seller_name == "northwind_goods" and d.quantity == 40
    assert d.seller_age_days and 600 <= d.seller_age_days <= 800


# ---------------------------------------------------------------- in-page action spans


async def test_in_page_actions_are_traced(live: LiveServer, tmp_path: Path, secrets: Secrets, engine: Any) -> None:
    async with run_span(None, {}) as run, _browser(live, tmp_path, secrets) as b:
        async with task_span(run.id, "discovery", "tradeloft", "disc:tl"):
            await search_site(b, "tradeloft", "vaelo", max_pages=3)
        async with task_span(run.id, "discovery", "echobazaar", "disc:mb"):
            await search_site(b, "echobazaar", "vaelo", max_pages=1)
        async with task_span(run.id, "draft", "portal", "draft:x"):
            async with b.page("portal") as page:
                await b.login("portal", page)
    with Session(engine) as s:
        tasks = list(s.exec(select(Task).where(Task.run_id == run.id)))
    by_key = {t.dedup_key: t for t in tasks}
    actions = [t for t in tasks if t.type == "browser:action"]
    names = sorted(t.payload["action"] for t in actions)
    assert names.count("click:load_more") == 2 and "submit:search_form" in names
    assert "click:cookie_dismiss" in names and "submit:login" in names
    assert all(t.status == TaskStatus.succeeded and t.started_at and t.finished_at for t in actions)
    parent_of = {t.payload["action"]: t.parent_id for t in actions}
    assert parent_of["click:load_more"] == by_key["disc:tl"].id
    assert parent_of["click:cookie_dismiss"] == by_key["disc:mb"].id
    login_span = next(t for t in tasks if t.type == "browser:login")
    assert parent_of["submit:login"] == login_span.id and login_span.parent_id == by_key["draft:x"].id
