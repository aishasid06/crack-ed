"""Learned recipes on Bazaarly: learned from a (fake-model) agent run, replayed with no LLM, re-learned when
they break, and held to the navigator's guardrails when tampered with."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, select

from bh.browser.nav_report import markdown as nav_markdown
from bh.browser.nav_report import summarise as nav_summary
from bh.browser.ops import capture_listing, search_site
from bh.browser.recipes import (FieldLoc, Replayer, choose_locations, Locator, Recipe, RecipeStore, SearchRecipe, field_pattern, parse_field,
                                path_pattern)
from bh.browser.session import BrowserManager
from bh.config import AppConfig, Secrets
from bh.db import ExtractionMethod, LLMCall, Task
from bh.errors import ExtractionFailed
from bh.trace import run_span
from mocksites.search import search
from tests.test_agent_nav import SEARCH, _cfg, _llm, bz_live  # noqa: F401 (fixture)
from tests.test_bazaarly import bz_catalog  # noqa: F401 (fixture)
from tests.test_browser import LiveServer, secrets  # noqa: F401 (fixture)

LISTING = [("extract_listing",), ("click", "Details"), ("extract_listing",), ("click", "Seller"), ("extract_listing",),
           ("done", True)]


def _tasks(engine: Any, run_id: int, type_: str | None = None) -> list[Task]:
    with Session(engine) as s:
        q = select(Task).where(Task.run_id == run_id)
        rows = list(s.exec(q.where(Task.type == type_) if type_ else q))
    return sorted(rows, key=lambda t: t.id or 0)


def _calls(engine: Any, run_id: int) -> list[LLMCall]:
    with Session(engine) as s:
        return list(s.exec(select(LLMCall).where(LLMCall.run_id == run_id)))


async def _search(cfg: AppConfig, secrets: Secrets, sessions: Path, query: str, llm: Any = None) -> tuple[int, list]:
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=sessions, headless=True) as b:
        cards = await search_site(b, "bazaarly", query, llm=llm)
    return run.id, cards


@pytest.fixture
def rcfg(bz_live: LiveServer, tmp_path: Path) -> AppConfig:  # noqa: F811
    return _cfg(bz_live, recipes_dir=tmp_path / "recipes")


async def test_search_recipe_learned_then_replayed_without_llm(bz_live: LiveServer, rcfg: AppConfig,  # noqa: F811
                                                               secrets: Secrets, engine: Any, tmp_path: Path) -> None:
    llm, _ = _llm(rcfg, secrets, engine, SEARCH)
    run1, first = await _search(rcfg, secrets, tmp_path / "s1", "vaelo", llm)
    recipe = RecipeStore(rcfg).load("bazaarly")
    assert recipe is not None and recipe.version == 1 and recipe.search is not None
    s = recipe.search
    assert (s.input.css, s.submit, s.cards.container, s.cards.title, s.next.css if s.next else None, s.next_learned) \
        == ('input[name="term"]', None, "article.bz-tile", "h2.bz-tile-name", "button.bz-next", "agent")
    task1 = _tasks(engine, run1, "agent:task")[0]
    assert task1.payload["mode"] == "agent" and task1.payload["recipe_version"] == 1
    assert _tasks(engine, run1, "recipe:learn")[0].payload["learned"] is True

    # replay: fresh browser profile (cookie modal again), no LLM at all
    run2, again = await _search(rcfg, secrets, tmp_path / "s2", "vaelo", llm=None)
    assert [(c.url, c.title) for c in again] == [(c.url, c.title) for c in first]
    cat = {f"{bz_live.base}/bazaarly/p/{l.id}": l for l in bz_live.catalog.for_site("bazaarly")}
    assert all(c.price == cat[c.url].price and c.currency == "GBP" for c in again)
    assert _calls(engine, run2) == []  # zero model calls
    task2 = _tasks(engine, run2, "recipe:task")[0]
    assert (task2.payload["mode"], task2.payload["recipe_version"], task2.payload["tokens"], task2.payload["success"]) \
        == ("replay", 1, 0, True)
    assert task2.payload["listings_found"] == len(first) and task2.payload["consent"] == ["Only necessary"]
    steps = [t.payload["action"] for t in _tasks(engine, run2, "recipe:step")]
    assert steps == ["open", "type", "read_results", "click", "read_results"]
    assert all(t.payload["kind"] == "code step" for t in _tasks(engine, run2, "recipe:step"))

    a, b = nav_summary(run1, engine=engine), nav_summary(run2, engine=engine)
    assert a["search"]["modes"] == ["agent"] and b["search"]["modes"] == ["replay"]
    assert a["tokens"] > 0 and (b["tokens"], b["llm_calls"], b["cost_usd"]) == (0, 0, 0.0)
    assert a["listings_found"] == b["listings_found"] == len(first) and b["recipe_versions"] == [1]
    assert "| Listings found (search) |" in nav_markdown(a, b)

    # a new query on the same recipe
    _, bottle = await _search(rcfg, secrets, tmp_path / "s2", "bottle", llm=None)
    expected = search(bz_live.catalog.for_site("bazaarly"), "bottle")
    assert expected and [c.url.split("/p/")[1] for c in bottle] == [l.id for l in expected[:30]]

    # a query with no results replays cleanly (no listing links on the page → not a broken pattern)
    run4, none = await _search(rcfg, secrets, tmp_path / "s2", "zzqxv", llm=None)
    assert none == [] and _tasks(engine, run4, "recipe:task")[0].payload["success"] is True


async def test_listing_recipe_learned_then_replayed_without_llm(bz_live: LiveServer, rcfg: AppConfig,  # noqa: F811
                                                                secrets: Secrets, engine: Any, tmp_path: Path) -> None:
    l0, l1 = bz_live.catalog.for_site("bazaarly")[:2]
    llm, _ = _llm(rcfg, secrets, engine, LISTING)
    async with BrowserManager(rcfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        await capture_listing(b, "bazaarly", f"{bz_live.base}/bazaarly/p/{l0.id}", llm)
        recipe = RecipeStore(rcfg).load("bazaarly")
        assert recipe is not None and recipe.listing is not None
        assert set(recipe.listing.fields) == {"title", "description", "price", "seller_name", "seller_location",
                                              "seller_age_days", "rating", "quantity", "condition"}
        assert recipe.listing.fields["rating"].tab.css == "#tab-seller"  # type: ignore[union-attr]
        assert recipe.listing.images is not None and recipe.listing.fields["title"].tab is None
        async with run_span(None, {}) as run:
            d = await capture_listing(b, "bazaarly", f"{bz_live.base}/bazaarly/p/{l1.id}", None,
                                      screenshot_path=tmp_path / "shot.png")
    assert (d.title, d.description, d.price, d.currency, d.seller_name, d.seller_location, d.seller_age_days,
            d.rating, d.quantity, d.condition) == \
        (l1.title, l1.description, l1.price, "GBP", l1.seller_name, l1.seller_location, l1.seller_age_days,
         l1.seller_rating, l1.quantity, l1.condition)
    assert d.image_urls == [f"{bz_live.base}/media/{i}" for i in l1.images]
    assert d.extraction_method == ExtractionMethod.recipe and (tmp_path / "shot.png").exists()
    assert _calls(engine, run.id) == []
    t = _tasks(engine, run.id, "recipe:task")[0]
    assert t.payload["mode"] == "replay" and t.payload["recipe_version"] == 1 and len(t.payload["fields_found"]) == 9
    assert [x.payload["action"] for x in _tasks(engine, run.id, "recipe:step")] == \
        ["open", "read_fields", "read_images", "click", "read_fields", "click", "read_fields"]


async def test_broken_recipe_falls_back_to_agent_and_is_relearned(bz_live: LiveServer, rcfg: AppConfig,  # noqa: F811
                                                                  secrets: Secrets, engine: Any, tmp_path: Path) -> None:
    llm, _ = _llm(rcfg, secrets, engine, SEARCH)
    first_run, first = await _search(rcfg, secrets, tmp_path / "s", "vaelo", llm)
    store = RecipeStore(rcfg)
    path = store.path("bazaarly")
    raw = json.loads(path.read_text())
    raw["search"]["cards"]["container"] = "article.bz-card"  # the site "changed its markup"
    path.write_text(json.dumps(raw))

    llm2, fake2 = _llm(rcfg, secrets, engine, SEARCH)
    run2, cards = await _search(rcfg, secrets, tmp_path / "s", "vaelo", llm2)
    assert [c.url for c in cards] == [c.url for c in first] and not fake2.script  # the agent did the work
    failed = _tasks(engine, run2, "recipe:task")[0]
    assert failed.payload["success"] is False and "matched nothing" in failed.payload["reason"]
    agent = _tasks(engine, run2, "agent:task")[0]
    assert agent.payload["mode"] == "agent" and agent.payload["recipe_version"] == 2
    fixed = store.load("bazaarly")
    assert fixed is not None and fixed.version == 2 and fixed.search.cards.container == "article.bz-tile"  # type: ignore[union-attr]
    assert [h["version"] for h in fixed.history] == [1, 2]
    # and the repaired recipe replays again
    run3, again = await _search(rcfg, secrets, tmp_path / "s", "vaelo", None)
    assert [c.url for c in again] == [c.url for c in first] and _calls(engine, run3) == []


async def test_broken_listing_recipe_falls_back_to_agent(bz_live: LiveServer, rcfg: AppConfig,  # noqa: F811
                                                         secrets: Secrets, engine: Any, tmp_path: Path) -> None:
    l0, l1 = bz_live.catalog.for_site("bazaarly")[:2]
    async with BrowserManager(rcfg, secrets, sessions_dir=tmp_path / "s", headless=True) as b:
        await capture_listing(b, "bazaarly", f"{bz_live.base}/bazaarly/p/{l0.id}", _llm(rcfg, secrets, engine, LISTING)[0])
        store = RecipeStore(rcfg)
        raw = json.loads(store.path("bazaarly").read_text())
        raw["listing"]["fields"]["title"]["css"] = "h1.gone"
        store.path("bazaarly").write_text(json.dumps(raw))
        with pytest.raises(ExtractionFailed, match="needs an LLM"):  # replay fails, and there is no model
            await capture_listing(b, "bazaarly", f"{bz_live.base}/bazaarly/p/{l1.id}", None)
        d = await capture_listing(b, "bazaarly", f"{bz_live.base}/bazaarly/p/{l1.id}",
                                  _llm(rcfg, secrets, engine, LISTING)[0])
    assert d.title == l1.title and d.extraction_method == ExtractionMethod.generic_llm
    r = store.load("bazaarly")
    assert r is not None and r.version == 2 and r.listing.fields["title"].css == "h1.bz-title"  # type: ignore[union-attr]


async def test_tampered_recipe_is_held_to_guardrails(bz_live: LiveServer, rcfg: AppConfig,  # noqa: F811
                                                     secrets: Secrets, engine: Any, tmp_path: Path) -> None:
    llm, _ = _llm(rcfg, secrets, engine, SEARCH)
    await _search(rcfg, secrets, tmp_path / "s", "vaelo", llm)
    store = RecipeStore(rcfg)
    raw = json.loads(store.path("bazaarly").read_text())
    raw["search"]["input"] = {"css": 'input[name="email"]', "name": "Email for deals", "tag": "input", "role": ""}
    store.path("bazaarly").write_text(json.dumps(raw))
    n = len(bz_live.requests)
    async with run_span(None, {}) as run, BrowserManager(rcfg, secrets, sessions_dir=tmp_path / "s",
                                                         headless=True) as b:
        with pytest.raises(ExtractionFailed, match="no adapter and no LLM"):
            await search_site(b, "bazaarly", "me@example.com", llm=None)
    assert ("POST", "/bazaarly/newsletter") not in bz_live.requests[n:]
    step = [t for t in _tasks(engine, run.id, "recipe:step") if t.payload["action"] == "type"][0]
    assert "guardrail refused" in step.payload["outcome"]


# ---------------------------------------------------------------- units


def test_path_pattern() -> None:
    assert path_pattern(["/bazaarly/results/ab12/1", "/bazaarly/results/ab12/2"]) == \
        r"^/bazaarly/results/[^/]+/[^/]+/?$"
    assert path_pattern(["/bazaarly/p/BZ-1", "/bazaarly/p/BZ-2"]) == r"^/bazaarly/p/[^/]+/?$"


@pytest.mark.parametrize("field,learn_text,value,other_text,expected", [
    ("rating", "Rated 3.1 out of 5 by 54 buyers", "3.1", "Rated 4.7 out of 5 by 9 buyers", 4.7),
    ("price", "Asking £12.99", "12.99", "Asking £1,234.50", 1234.5),
    ("seller_location", "From Leeds, UK", "Leeds, UK", "From Guangzhou, China", "Guangzhou, China"),
    ("seller_age_days", "On Bazaarly for 32 days", "32", "On Bazaarly for 1,200 days", 1200),
    ("title", "Steel bottle", "Steel bottle", "Glass jar", "Glass jar"),
])
def test_field_patterns_generalise(field: str, learn_text: str, value: str, other_text: str, expected: Any) -> None:
    loc = FieldLoc(css="x", pattern=field_pattern(field, learn_text, value))
    assert parse_field(field, learn_text, loc) == (float(value) if field in ("rating", "price") else
                                                   int(value) if field == "seller_age_days" else value)
    assert parse_field(field, other_text, loc) == expected


async def test_store_versions_and_bad_files(rcfg: AppConfig) -> None:
    store = RecipeStore(rcfg)
    part = SearchRecipe(input=Locator(css="#q"), results_url="^/r$",
                        cards={"container": "li", "href": "^/p/[^/]+$"})  # type: ignore[arg-type]
    assert await store.save_part("bazaarly", "search", part, "a") == 1
    assert await store.save_part("bazaarly", "search", part, "same again") == 1  # unchanged → same version
    assert await store.save_part("bazaarly", "search", part.model_copy(update={"next_learned": "heuristic"}), "b") == 2
    store.path("bazaarly").write_text("{not json")
    assert store.load("bazaarly") is None  # unreadable recipe is ignored (agent loop runs instead)
    store.path("bazaarly").write_text(Recipe(site="mockbay").model_dump_json())
    assert store.load("bazaarly") is None  # a recipe for another site is ignored
    with pytest.raises(ValueError):
        store.path("../evil")


def test_value_inside_the_title_is_not_learned_from_the_title() -> None:
    """Live run 23: condition "Pre-owned" was learned from the h1 "Pre-owned Vaelo Arc Copper Bottle 900ml"
    (found first, on arrival) instead of the Details tab's "Condition: Pre-owned"."""
    details = Locator(css="#tab-details", name="Details", role="tab")
    title = {"css": "h1.t", "text": "Pre-owned Vaelo Arc Copper Bottle 900ml", "value": "Pre-owned Vaelo Arc Copper Bottle 900ml"}
    arrival = {"title": title, "condition": {"css": "h1.t", "text": title["text"], "value": "Pre-owned"}}
    on_details = {"title": title | {"css": "h1.t"},
                  "condition": {"css": "li.c", "text": "Condition: Pre-owned", "value": "Pre-owned"}}
    got = choose_locations([(None, arrival), (details, on_details)])
    assert got["title"].css == "h1.t" and got["title"].tab is None
    assert (got["condition"].css, got["condition"].tab) == ("li.c", details)
    assert parse_field("condition", "Condition: New", got["condition"]) == "New"
    # a field whose only hit is an element another field fills whole is not learned at all
    only_title = choose_locations([(None, {"title": title, "condition": arrival["condition"]})])
    assert "condition" not in only_title


async def test_recipe_without_next_control_still_pages(bz_live: LiveServer, rcfg: AppConfig,  # noqa: F811
                                                       secrets: Secrets, engine: Any, tmp_path: Path) -> None:
    """Live run 23 learned from a one-page search, so its recipe has no next-page control. Replay of a longer
    search must still read page 2 (visible "Next" control found by code)."""
    llm, _ = _llm(rcfg, secrets, engine, SEARCH)
    _, first = await _search(rcfg, secrets, tmp_path / "s", "vaelo", llm)
    store = RecipeStore(rcfg)
    raw = json.loads(store.path("bazaarly").read_text())
    raw["search"]["next"], raw["search"]["next_learned"] = None, "none"
    store.path("bazaarly").write_text(json.dumps(raw))
    run, again = await _search(rcfg, secrets, tmp_path / "s", "vaelo", None)
    assert len(first) > 10 and [c.url for c in again] == [c.url for c in first]
    assert [t.payload["action"] for t in _tasks(engine, run, "recipe:step")] == \
        ["open", "type", "read_results", "click", "read_results"]


# ---------------------------------------------------------------- fixes from the live sandbox run (runs 25/26)


async def test_caller_stopping_early_is_not_a_failure(bz_live: LiveServer, rcfg: AppConfig,  # noqa: F811
                                                      secrets: Secrets, engine: Any, tmp_path: Path) -> None:
    """Run 25: the site allows 1 result page, discovery stopped reading after it, and the agent's search was
    recorded as "failed: GeneratorExit". Stopping early is normal; the recipe must still be learned."""
    from contextlib import aclosing

    from bh.browser.ops import iter_search_pages

    async def first_page(llm: Any) -> int:
        async with run_span(None, {}) as run, BrowserManager(rcfg, secrets, sessions_dir=tmp_path / "s",
                                                             headless=True) as b:
            async with aclosing(iter_search_pages(b, "bazaarly", "vaelo", llm=llm)) as it:
                async for _ in it:
                    break  # e.g. max_pages reached
        return run.id

    run1 = await first_page(_llm(rcfg, secrets, engine, SEARCH[:2])[0])  # type + extract_results, then stopped
    t = _tasks(engine, run1, "agent:task")[0]
    assert t.status.value == "succeeded" and "stopped by the caller after 1" in t.payload["reason"]
    assert RecipeStore(rcfg).load("bazaarly").search is not None  # type: ignore[union-attr]
    run2 = await first_page(None)  # replay, also stopped after page 1
    r = _tasks(engine, run2, "recipe:task")[0]
    assert r.status.value == "succeeded" and r.payload["success"] is True and "stopped by the caller" in r.payload["reason"]


@pytest.mark.parametrize("text,expected", [("£1,234.50", 1234.5), ("89,99 €", 89.99), ("1.234,56 €", 1234.56),
                                           ("Asking £12.99", 12.99), ("1,234", 1234.0), ("€ 7", 7.0), ("free", None)])
def test_parse_price_both_conventions(text: str, expected: float | None) -> None:
    from bh.browser.recipes import parse_price

    assert parse_price(text) == expected


def test_comma_decimal_price_pattern() -> None:
    """Run 25: "89,99 €" was not learned (only "89.99" was looked for), so replay returned no price."""
    from bh.browser.recipes import _candidates

    assert "89,99" in _candidates("price", 89.99) and "1.234,56" in _candidates("price", 1234.56)
    loc = FieldLoc(css="x", pattern=field_pattern("price", "89,99 €", "89,99"))
    assert parse_field("price", "89,99 €", loc) == 89.99 and parse_field("price", "1.299,00 €", loc) == 1299.0


_ONE = """<!doctype html><html><body><header><a href="/">shop</a><input type="text"></header>
<nav><a href="/products/category/nintendo">Nintendo</a></nav><main><div class="grid">{cards}</div></main></body></html>"""
_CARD = ('<div class="card"><img src="/img/{i}.jpg" alt=""><a class="card-link" href="/products/{i}">'
         '<h4 class="title">Game {i}</h4></a><div class="price">{p} €</div></div>')


async def test_card_pattern_learned_from_a_single_result() -> None:
    """Run 25: one search result, and the card pattern needed two, so no search recipe was learned."""
    from playwright.async_api import async_playwright

    from bh.browser.adapters.base import Card
    from bh.browser.recipes import _READ_CARDS_JS, probe_cards

    pages = {"/one": _ONE.format(cards=_CARD.format(i=5, p="89,99")),
             "/many": _ONE.format(cards="".join(_CARD.format(i=i, p=f"{i}9,99") for i in range(1, 4)))}

    async def handler(route: Any) -> None:
        path = route.request.url[len("http://shop.test"):]
        await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=pages.get(path, ""))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        await page.route("http://shop.test/**", handler)
        await page.goto("http://shop.test/one")
        probe = await probe_cards(page, [Card(site="s", url="http://shop.test/products/5", title="Game 5")])
        assert probe is not None and probe["container"] == "div.card" and probe["covered"] == probe["total"] == 1
        assert (probe["link"], probe["title"], probe["price"]) == ("a.card-link", "h4.title", "div.price")
        await page.goto("http://shop.test/many")
        raw = await page.evaluate(_READ_CARDS_JS, {"container": probe["container"], "link": probe["link"],
                                                   "title": probe["title"], "title_label_prefix": None,
                                                   "price": probe["price"]})
        await browser.close()
    assert [(r["url"].rsplit("/", 1)[1], r["title"], r["price"]) for r in raw] == \
        [("1", "Game 1", "19,99 €"), ("2", "Game 2", "29,99 €"), ("3", "Game 3", "39,99 €")]
    assert Replayer._card("s", raw[0]).price == 19.99 and Replayer._card("s", raw[0]).currency == "EUR"  # type: ignore[union-attr]


def test_recipes_from_before_the_fixes_are_ignored(rcfg: AppConfig) -> None:
    """Run 27 replayed a recipe learned before comma-decimal prices were supported (no price field). Recipes of an
    older schema are ignored, so the agent re-learns them."""
    store = RecipeStore(rcfg)
    store.path("bazaarly").parent.mkdir(parents=True, exist_ok=True)
    store.path("bazaarly").write_text(Recipe(site="bazaarly", version=1).model_copy(
        update={"schema_version": 1}).model_dump_json())
    assert store.load("bazaarly") is None
