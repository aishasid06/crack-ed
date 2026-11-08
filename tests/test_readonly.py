"""Real marketplace, read-only (SPEC §9 Phase 11 items 5 and 6): config rules, real-brand stop-list, robots.txt,
no cases against real sellers, and purging what was collected. Exercised on the mock server with a site
registered read_only (no real site is contacted)."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from sqlmodel import Session, select

import bh.config as C
from bh.browser.agent_nav import GuardrailBlocked, NavigationFailed, Navigator
from bh.browser.ops import iter_search_pages
from bh.browser.readonly import RobotsPolicy, query_refusal, stoplist_hit
from bh.browser.recipes import RecipeStore
from bh.browser.session import BrowserManager
from bh.config import PROJECT_ROOT, AppConfig, Secrets
from bh.db import Case, CaseStatus, Features, Listing, Run, Task, TaskStatus
from bh.discovery.purge import purge_site
from bh.discovery.worker import search_adaptive
from bh.enforce.approvals import approve_case
from bh.enforce.guard import check_listing_target
from bh.enforce.portal import load_case
from bh.errors import EnforcementBlocked, ReadOnlyBlocked
from bh.orchestrator.run import PreflightError, preflight
from tests.test_agent_nav import _cfg, _llm, bz_live  # noqa: F401 (fixture)
from tests.test_bazaarly import bz_catalog  # noqa: F401 (fixture)
from tests.test_browser import LiveServer, secrets  # noqa: F401 (fixture)

REAL = {"base_url": "https://www.realmarket.co.uk", "mode": "read_only", "max_concurrency": 1, "max_pages": 1,
        "min_delay_ms": 3000}


def _raw() -> dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())


def _with_site(raw: dict[str, Any], **site: Any) -> AppConfig:
    raw = copy.deepcopy(raw)
    raw["sites"]["realsite"] = {**REAL, **site}
    return AppConfig.model_validate(raw)


def _read_only(cfg: AppConfig, site: str, **upd: Any) -> AppConfig:
    sc = cfg.sites[site].model_copy(update={"mode": "read_only", "max_concurrency": 1, "max_pages": 2, **upd})
    return cfg.model_copy(update={"sites": {**cfg.sites, site: sc}})


@pytest.fixture
def fast_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests only: no 2 s pacing on the local mock server (the limit still applies to every real site)."""
    monkeypatch.setattr(C, "READ_ONLY_MIN_DELAY_MS", 0)


class Robots:
    def __init__(self, status: int = 200, body: str = "") -> None:
        self.status, self.body, self.calls = status, body, []

    async def __call__(self, url: str) -> tuple[int, str]:
        self.calls.append(url)
        return self.status, self.body


# ---------------------------------------------------------------- config rules (enforced in code)


def test_real_site_config_rules() -> None:
    raw = _raw()
    assert not C.is_real_host("127.0.0.1") and not C.is_real_host("unseen-market.test")  # local / reserved TLD
    assert C.is_real_host("www.realmarket.co.uk")
    assert _with_site(raw).sites["realsite"].mode == "read_only"
    with pytest.raises(ValidationError, match="must be registered mode: read_only"):
        _with_site(raw, mode="full")
    with pytest.raises(ValidationError, match="max_pages 5 > 2"):
        _with_site(raw, max_pages=5)
    with pytest.raises(ValidationError, match="max_concurrency 3 > 1"):
        _with_site(raw, max_concurrency=3)
    with pytest.raises(ValidationError, match="min_delay_ms 250 < 2000"):
        _with_site(raw, min_delay_ms=250)
    no_stop = copy.deepcopy(raw)
    no_stop["discovery"]["real_brand_stoplist"] = []
    with pytest.raises(ValidationError, match="real_brand_stoplist"):
        _with_site(no_stop)
    full = copy.deepcopy(raw)
    full["sites"]["mockbay"]["extra_hosts"] = ["cdn.example"]
    with pytest.raises(ValidationError, match="extra_hosts is only allowed on read_only"):
        AppConfig.model_validate(full)


def test_loosened_copy_cannot_open_a_browser(cfg: AppConfig, secrets: Secrets, tmp_path: Path) -> None:  # noqa: F811
    loose = _read_only(cfg, "mockbay", max_pages=10)  # model_copy skips validators ...
    with pytest.raises(ValueError, match="read_only site limits exceeded"):
        BrowserManager(loose, secrets, sessions_dir=tmp_path / "s")  # ... the browser manager re-validates


# ---------------------------------------------------------------- stop-list


def test_stoplist(cfg: AppConfig) -> None:
    assert cfg.discovery.real_brand_stoplist  # shipped with real brand names
    assert stoplist_hit(cfg, "valeo") == "valeo" and stoplist_hit(cfg, "VALEO water-bottle") == "valeo"
    assert stoplist_hit(cfg, "vaelo") is None and stoplist_hit(cfg, "velodrome flask") is None  # whole words only
    ro = _read_only(cfg, "mockbay", min_delay_ms=2000)
    assert query_refusal(ro, "mockbay", "valeo bottle") and query_refusal(ro, "mockbay", "copper bottle") is None
    assert query_refusal(cfg, "mockbay", "valeo") is None  # our own mock sites are unaffected


async def test_stoplisted_query_is_never_sent(cfg: AppConfig) -> None:
    ro = _read_only(cfg, "mockbay", min_delay_ms=2000)
    browser = SimpleNamespace(cfg=ro)  # any browser use would fail
    stats = await search_adaptive(browser, "mockbay", "valeo", prefilter=None, llm=None)  # type: ignore[arg-type]
    assert stats["stop"] == "stop_list" and stats["pages"] == 0 and "valeo" in stats["refused"]
    with pytest.raises(ReadOnlyBlocked, match="stop-list"):
        async for _ in iter_search_pages(browser, "mockbay", "Valeo bottle"):  # type: ignore[arg-type]
            pass


# ---------------------------------------------------------------- robots.txt


@pytest.mark.parametrize("status,body,expected", [
    (200, "User-agent: *\nDisallow: /search\n", {"/item/1": True, "/search?q=x": False}),
    (200, "User-agent: BountyHunter\nDisallow: /\n\nUser-agent: *\nAllow: /\n", {"/item/1": False}),
    (404, "", {"/anything": True}),        # no robots.txt: no rules
    (403, "", {"/item/1": False}),         # refused: read nothing
    (503, "", {"/item/1": False}),         # server error: read nothing
    (0, "", {"/item/1": False}),           # unreachable: read nothing
])
async def test_robots_policy(cfg: AppConfig, status: int, body: str, expected: dict[str, bool]) -> None:
    ro = _with_site(_raw())
    fetch = Robots(status, body)
    pol = RobotsPolicy(ro, fetch=fetch)
    for path, ok in expected.items():
        assert await pol.allowed("realsite", f"https://www.realmarket.co.uk{path}") is ok, path
    assert fetch.calls == ["https://www.realmarket.co.uk/robots.txt"]  # fetched once per origin
    assert await RobotsPolicy(cfg, fetch=Robots(0)).allowed("mockbay", "http://127.0.0.1/mockbay/")  # mock sites


async def test_goto_refused_before_any_request(bz_live: LiveServer, secrets: Secrets, tmp_path: Path,  # noqa: F811
                                               fast_limits: None) -> None:
    cfg = _read_only(_cfg(bz_live), "bazaarly")
    n = len(bz_live.requests)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        b.robots = RobotsPolicy(cfg, fetch=Robots(200, "User-agent: *\nDisallow: /bazaarly/\n"))
        async with b.page("bazaarly") as page:
            with pytest.raises(ReadOnlyBlocked, match="robots.txt disallows"):
                await b.goto("bazaarly", page, b.site_url("bazaarly", "/"))
    assert bz_live.requests[n:] == []


async def test_navigator_respects_robots_on_read_only_site(bz_live: LiveServer, secrets: Secrets,  # noqa: F811
                                                           engine: Any, tmp_path: Path, fast_limits: None) -> None:
    cfg = _read_only(_cfg(bz_live), "bazaarly")
    llm, fake = _llm(cfg, secrets, engine, [("type", "What are you looking for?", "copper bottle", True)] * 2)
    n = len(bz_live.requests)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        b.robots = RobotsPolicy(cfg, fetch=Robots(200, "User-agent: *\nDisallow: /bazaarly/search\n"))
        async with b.page("bazaarly") as page:
            with pytest.raises(NavigationFailed, match="no progress after a retry"):
                async for _ in Navigator(b, "bazaarly", llm).search(page, "copper bottle"):
                    pass
    assert ("POST", "/bazaarly/search") not in bz_live.requests[n:]  # the search form was never submitted
    assert "robots.txt disallows /bazaarly/search" in fake.prompts[1]


async def test_landing_on_a_disallowed_page_is_undone(bz_live: LiveServer, secrets: Secrets, engine: Any,  # noqa: F811
                                                      tmp_path: Path, fast_limits: None) -> None:
    cfg = _read_only(_cfg(bz_live), "bazaarly")
    llm, _ = _llm(cfg, secrets, engine, [("type", "What are you looking for?", "copper bottle", True)])
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        b.robots = RobotsPolicy(cfg, fetch=Robots(200, "User-agent: *\nDisallow: /bazaarly/results/\n"))
        async with b.page("bazaarly") as page:
            with pytest.raises(NavigationFailed, match="robots.txt disallows .*/bazaarly/results/"):
                async for _ in Navigator(b, "bazaarly", llm).search(page, "copper bottle"):
                    pass
            assert "/results/" not in page.url  # went back


# ---------------------------------------------------------------- never a case against a real seller


def _case(engine: Any, platform: str, url: str) -> int:
    with Session(engine) as s:
        l = Listing(platform=platform, url=url, title="x")
        s.add(l)
        s.commit()
        s.refresh(l)
        c = Case(listing_id=l.id, category="counterfeit_identical", confidence=0.9, rule_id="r", action="platform_report",
                 status=CaseStatus.awaiting_approval)
        s.add(c)
        s.commit()
        s.refresh(c)
        return c.id  # type: ignore[return-value]


def test_real_listings_cannot_be_drafted_approved_or_submitted(cfg: AppConfig, engine: Any) -> None:
    ro = _read_only(cfg, "mockbay", min_delay_ms=2000)
    with pytest.raises(EnforcementBlocked, match="read_only site 'mockbay'"):
        check_listing_target(ro, "mockbay", "http://127.0.0.1:8000/mockbay/listing/1")
    with pytest.raises(EnforcementBlocked, match="not in the enforcement allowlist"):
        check_listing_target(cfg, "mockbay", "https://real-shop.example/item/1")  # a host outside the allowlist, whatever the site
    check_listing_target(cfg, "mockbay", "http://127.0.0.1:8000/mockbay/listing/1")  # mock listing: fine

    cid = _case(engine, "realsite", "https://real-shop.example/item/9")
    with pytest.raises(EnforcementBlocked):
        load_case(engine, cid, cfg)  # drafting and submission both load cases here
    with pytest.raises(EnforcementBlocked):
        approve_case(cid, "lawyer1", engine)
    with Session(engine) as s:
        assert s.get(Case, cid).status == CaseStatus.awaiting_approval  # type: ignore[union-attr]


async def test_preflight_stops_real_sites_before_decide(bz_live: LiveServer, secrets: Secrets) -> None:  # noqa: F811
    cfg = _read_only(_cfg(bz_live), "mockbay", min_delay_ms=2000)
    for until in ("decide", "draft"):
        with pytest.raises(PreflightError, match="real marketplaces"):
            await preflight(cfg, until, secrets, ["mockbay"], None)  # type: ignore[arg-type]
    assert await preflight(cfg, "discovery", secrets, ["mockbay"], None) is None


# ---------------------------------------------------------------- purge


def test_purge_read_only_site(cfg: AppConfig, engine: Any, tmp_path: Path) -> None:
    paths = cfg.paths.model_copy(update={"data_dir": str(tmp_path)})
    ro = _read_only(cfg, "mockbay", min_delay_ms=2000).model_copy(update={"paths": paths})
    cid = _case(engine, "mockbay", "http://127.0.0.1:8000/mockbay/listing/77")
    other = _case(engine, "tradeloft", "http://127.0.0.1:8000/tradeloft/listing/TL-1")
    with Session(engine) as s:
        lid = s.get(Case, cid).listing_id  # type: ignore[union-attr]
        s.add(Features(listing_id=lid))
        run = Run()
        s.add(run)
        s.flush()
        s.add(Task(run_id=run.id, type="discovery", site="mockbay", dedup_key="d1",  # type: ignore[arg-type]
                   status=TaskStatus.succeeded, payload={"kept": [{"seller": "real_person"}]}))
        s.commit()
    (tmp_path / "listings" / "mockbay" / "x").mkdir(parents=True)
    (tmp_path / "listings" / "mockbay" / "x" / "shot.png").write_bytes(b"x")

    dry = purge_site(ro, "mockbay", apply=False, engine=engine)
    assert (dry["listings"], dry["cases"], dry["applied"]) == (1, 1, False)
    assert (tmp_path / "listings" / "mockbay").exists()
    purge_site(ro, "mockbay", apply=True, engine=engine)
    with Session(engine) as s:
        assert s.exec(select(Listing).where(Listing.platform == "mockbay")).first() is None
        assert s.get(Case, cid) is None and s.get(Case, other) is not None  # other sites untouched
        assert all(t.payload == {"purged": True} for t in s.exec(select(Task).where(Task.site == "mockbay")))
    assert not (tmp_path / "listings" / "mockbay").exists()
    with pytest.raises(ValueError, match="not read_only"):
        purge_site(cfg, "tradeloft", apply=True, engine=engine)  # mock data is never purged


# ---------------------------------------------------------------- JavaScript search box (no <form>), --query

JS = "http://jsshop.test"
_HOME = """<!doctype html><html><head><title>JS shop</title></head><body>
<header><a href="/">jsshop</a><div><input type="text" class="x1"><svg aria-hidden="true"></svg></div></header>
<main><h1>Games</h1><p>Search with the box above.</p></main>
<script>document.querySelector('header input').addEventListener('keydown', e => {
  if (e.key === 'Enter') location.href = '/products?q=' + encodeURIComponent(e.target.value); });</script>
</body></html>"""
_RESULTS = """<!doctype html><html><head><title>Results</title></head><body>
<header><a href="/">jsshop</a><div><input type="text" class="x1"></div></header>
<main>""" + "".join(f'<div class="card"><h4>Super Mario {i}</h4><a href="/products/{i}">Open Super Mario {i}</a></div>'
                    for i in range(1, 4)) + "</main></body></html>"


def _el(**kw: Any) -> dict[str, Any]:
    return {"index": 0, "tag": "input", "type": "text", "name": "", "href": "", "submits": False, "form": None,
            "consent_dialog": False, "header": True, "page_has_password": False, **kw}


def test_formless_search_box_rule() -> None:
    nav = Navigator.__new__(Navigator)
    nav.host, page = "jsshop.test", f"{JS}/"
    box = _el()
    nav._check("type", {"index": 0, "text": "mario"}, box, page, [box])  # header's only free text box: allowed
    for bad, others, why in [
        (_el(header=False), [], "not in the header, no search hint"),
        (_el(), [_el(index=1)], "two free text boxes: which one is search?"),
        (_el(page_has_password=True), [], "a login page"),
        (_el(type="email"), [], "email box"),
    ]:
        with pytest.raises(GuardrailBlocked):
            nav._check("type", {"index": 0, "text": "x"}, bad, page, [bad, *others])
    hinted = _el(header=False, name="Search products")
    nav._check("type", {"index": 0, "text": "x"}, hinted, page, [hinted])  # labelled search box outside header


async def test_navigator_searches_a_javascript_search_box(cfg: AppConfig, secrets: Secrets, engine: Any,  # noqa: F811
                                                          tmp_path: Path, fast_limits: None) -> None:
    site = cfg.sites["mockbay"].model_copy(update={"base_url": JS, "adapter": None, "navigator": True,
                                                   "mode": "read_only", "max_concurrency": 1, "max_pages": 1,
                                                   "min_delay_ms": 0})
    nav_cfg = cfg.navigator.model_copy(update={"settle_ms": 50, "recipes_dir": str(tmp_path / "r")})
    jcfg = cfg.model_copy(update={"sites": {**cfg.sites, "jsshop": site}, "navigator": nav_cfg})
    llm, fake = _llm(jcfg, secrets, engine, [("raw", "type", {"index": 1, "text": "Super Mario", "submit": True}),
                                            ("extract_results",), ("done", True)])
    seen: list[str] = []

    async def handler(route: Any) -> None:
        path = route.request.url[len(JS):]
        seen.append(path)
        body = _HOME if path in ("", "/") else _RESULTS if path.startswith("/products?") else None
        await (route.fulfill(status=200, content_type="text/html", body=body) if body else
               route.fulfill(status=404, body=""))

    async with BrowserManager(jcfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        b.robots = RobotsPolicy(jcfg, fetch=Robots(404))
        await (await b.context("jsshop")).route(f"{JS}/**", handler)
        async with b.page("jsshop") as page:
            cards = [c async for batch in Navigator(b, "jsshop", llm, store=RecipeStore(jcfg)).search(page, "Super Mario")
                     for c in batch]
    assert "(text box in the page header)" in fake.prompts[0]
    assert [c.url for c in cards] == [f"{JS}/products/{i}" for i in range(1, 4)]
    assert "/products?q=Super%20Mario" in seen
    recipe = RecipeStore(jcfg).load("jsshop")
    assert recipe is not None and recipe.search is not None and recipe.search.submit is None  # Enter, no form


async def test_query_option_only_for_read_only_sites(bz_live: LiveServer, secrets: Secrets) -> None:  # noqa: F811
    cfg = _read_only(_cfg(bz_live), "mockbay", min_delay_ms=2000)
    with pytest.raises(PreflightError, match="only for read_only"):
        await preflight(cfg, "discovery", secrets, ["tradeloft"], None, queries=["copper bottle"])  # full site
    with pytest.raises(PreflightError, match="stop-listed"):
        await preflight(cfg, "discovery", secrets, ["mockbay"], None, queries=["Valeo wipers"])
    with pytest.raises(PreflightError, match="non-empty"):
        await preflight(cfg, "discovery", secrets, ["mockbay"], None, queries=["  "])
    assert await preflight(cfg, "discovery", secrets, ["mockbay"], None, queries=["copper bottle"]) is None


def test_query_cli_refused_for_full_sites() -> None:
    from bh.orchestrator.run import main as run_main

    assert run_main(["--sites", "mockbay", "--query", "x", "--until", "discovery"]) == 2


async def test_manual_queries_replace_the_brand_queries(engine: Any) -> None:
    from bh.orchestrator.graph import node_generate_queries
    from bh.trace import run_span

    async with run_span(None, {}, engine=engine) as run:
        deps = SimpleNamespace(run_id=run.id, brand=object(), cfg=None, llm=None,
                               queries=["Super Mario Odyssey", "zelda"])
        out = await node_generate_queries({}, SimpleNamespace(context=deps))  # type: ignore[arg-type]
    assert out["queries"] == [{"text": "Super Mario Odyssey", "kind": "manual"}, {"text": "zelda", "kind": "manual"}]


def test_real_sites_only_when_named(cfg: AppConfig) -> None:
    from bh.discovery.worker import marketplace_sites

    assert cfg.site("realsite").mode == "read_only" and cfg.site("realsite").base_url == "https://sandbox.oxylabs.io"
    assert "realsite" not in marketplace_sites(cfg, with_navigator=True)  # never searched by default
    assert marketplace_sites(cfg, ["realsite"], with_navigator=True) == ["realsite"]


async def test_focusing_the_search_box_is_progress(cfg: AppConfig, secrets: Secrets, engine: Any,  # noqa: F811
                                                   tmp_path: Path, fast_limits: None) -> None:
    """Run 27: the agent clicked the header search box twice to focus it; each click changed nothing visible,
    was counted as a no-op, and the search stopped before it typed. A focus click is progress."""
    site = cfg.sites["mockbay"].model_copy(update={"base_url": JS, "adapter": None, "navigator": True,
                                                   "mode": "read_only", "max_concurrency": 1, "max_pages": 1,
                                                   "min_delay_ms": 0})
    nav_cfg = cfg.navigator.model_copy(update={"settle_ms": 50, "recipes_dir": str(tmp_path / "r")})
    jcfg = cfg.model_copy(update={"sites": {**cfg.sites, "jsshop": site}, "navigator": nav_cfg})
    click = ("raw", "click", {"index": 1})
    llm, fake = _llm(jcfg, secrets, engine, [click, click, ("raw", "type", {"index": 1, "text": "Mario", "submit": True}),
                                            ("extract_results",), ("done", True)])

    async def handler(route: Any) -> None:
        path = route.request.url[len(JS):]
        body = _HOME if path in ("", "/") else _RESULTS if path.startswith("/products?") else None
        await (route.fulfill(status=200, content_type="text/html", body=body) if body else
               route.fulfill(status=404, body=""))

    async with BrowserManager(jcfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        b.robots = RobotsPolicy(jcfg, fetch=Robots(404))
        await (await b.context("jsshop")).route(f"{JS}/**", handler)
        async with b.page("jsshop") as page:
            cards = [c async for batch in Navigator(b, "jsshop", llm).search(page, "Mario") for c in batch]
    assert len(cards) == 3 and not fake.script
    assert "focused the text box" in fake.prompts[1] and "Choose a different action" not in fake.prompts[1]
