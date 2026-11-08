"""Shopper-facing search: plain-language verdicts, the live activity feed, query planning, and the
shopper app's home, searching and results views."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.db import Features, Listing, Run, RunStatus, Task, TaskStatus, session_scope, utcnow
from ui.shop import data as shop_data
from ui.shop import feed
from ui.shop.verdicts import CATEGORY_GROUP, reason

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


# ---------------------------------------------------------------- verdicts


def test_every_category_has_a_group_and_a_plain_reason() -> None:
    l = Listing(platform="mockbay", url="u", title="t", price=7.5)
    f = Features(listing_id=1, replica_kw=["1:1"], price_ratio=0.23, origin_evidence="seller location: Osaka, Japan",
                 modification_kw=["refurbished"])
    for cat, group in CATEGORY_GROUP.items():
        text = reason(cat, l, f, "Acme")
        assert text.endswith(".") and "s.10" not in text and "rule" not in text.lower()
        assert group in {"genuine", "careful", "avoid", "lookalike", "hidden"}
    assert "“1:1”" in reason("counterfeit_identical", l, f, "Acme") and "23%" in reason("counterfeit_identical", l, f, "Acme")
    assert "Osaka" in reason("grey_import", l, f, "Acme")
    assert CATEGORY_GROUP["unrelated"] == "hidden" and CATEGORY_GROUP["authorised"] == "genuine"


# ---------------------------------------------------------------- feed


def _task(i: int, type_: str, site: str | None, status: TaskStatus, payload: dict | None = None,
          ago: int = 0) -> Task:
    start = utcnow() - timedelta(seconds=100 - i)
    return Task(id=i, run_id=1, type=type_, site=site, dedup_key=f"{type_}:{i}", status=status,
                started_at=start, finished_at=None if status == TaskStatus.running else start + timedelta(seconds=1),
                payload=payload or {})


def test_feed_shows_each_site_the_agent_is_on_and_the_current_step() -> None:
    tasks = [
        _task(1, "load_brand", None, TaskStatus.succeeded),
        _task(2, "identify_photo", None, TaskStatus.succeeded, {"product": "Arc Bottle"}),
        _task(3, "plan_search", None, TaskStatus.succeeded, {"queries": ["acme bottle", "acme"]}),
        _task(4, "discovery", "mockbay", TaskStatus.running, {"query": "acme bottle"}),
        _task(5, "browser:goto", "mockbay", TaskStatus.running, {"url": "http://x/mockbay/search?q=acme&page=2"}),
        _task(6, "discovery", "tradeloft", TaskStatus.succeeded, {"query": "acme", "cards": 8, "passing": 5}),
        _task(7, "browser:goto", "tradeloft", TaskStatus.running, {"url": "http://x/tradeloft/visual-search"}),
        _task(8, "discovery", "otherbay", TaskStatus.waiting_human, {"query": "acme"}),
    ]
    f = feed.build(tasks, "Acme", finished=False)
    assert [s for _, s in f.steps] == ["done", "active", "pending", "pending"]
    now = {n.site: n for n in f.now}
    assert now["mockbay"].doing == "Searching for “acme bottle”"
    assert feed.short_url(now["mockbay"].url) == "/mockbay/search?q=acme&page=2"
    assert now["tradeloft"].doing == "Opening a page"
    joined = " ".join(f.events)
    assert "Arc Bottle" in joined and "8 results, 5 worth a closer look" in joined
    assert "prove we're human" in joined


def test_feed_counts_progress_and_finishes() -> None:
    tasks = [_task(1, "capture_all", None, TaskStatus.succeeded, {"n": 3}),
             _task(2, "capture", "mockbay", TaskStatus.succeeded), _task(3, "capture", "mockbay", TaskStatus.running),
             _task(4, "classify_all", None, TaskStatus.running, {"n": 2}),
             _task(5, "classify", "mockbay", TaskStatus.succeeded)]
    f = feed.build(tasks, "Acme", finished=False)
    assert (f.opened, f.to_open, f.checked, f.to_check) == (1, 3, 1, 2)
    done = feed.build(tasks, "Acme", finished=True)
    assert all(s == "done" for _, s in done.steps) and done.now == []


# ---------------------------------------------------------------- query planning


def test_queries_put_the_shoppers_words_first_then_the_photo_product_then_the_mark() -> None:
    pytest.importorskip("imagehash")
    import numpy as np

    from bh.shopper import PhotoMatch, identify_product, shopper_queries

    product = SimpleNamespace(id=7, name="Acme Arc Bottle 900ml", short_name="Arc Bottle 900ml")
    ctx = SimpleNamespace(marks=["ACME"], name="Acme", product=lambda pid: product if pid == 7 else None,
                          nearest_clip=lambda v: (float(v[0]), 7))
    assert identify_product(ctx, np.array([0.9]), 0.8) == PhotoMatch(7, "Acme Arc Bottle 900ml", 0.9)
    assert identify_product(ctx, np.array([0.5]), 0.8) is None
    qs = shopper_queries("Copper  bottle", PhotoMatch(7, product.name, 0.9), ctx, 3)
    assert [q.text for q in qs] == ["copper bottle", "acme arc bottle 900ml", "acme"]
    assert [q.text for q in shopper_queries(None, None, ctx, 3)] == ["acme"]


# ---------------------------------------------------------------- app


@pytest.fixture
def shop(engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    from tests.ui_fixture import populate
    from ui import workers

    ids = populate(engine)
    with session_scope(engine) as s:
        run = s.get(Run, ids["run"])
        assert run is not None
        run.config_snapshot = {"stage": "shopper", "search_id": "abcdef012345"}
        run.status = RunStatus.finished
    shop_data.set_engine(engine)
    monkeypatch.setattr(shop_data, "uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(workers, "state_dir", lambda: tmp_path)
    (tmp_path / "abcdef012345.json").write_text('{"text": "acme bottle", "photo": null}', encoding="utf-8")
    yield ids
    shop_data.set_engine(None)


def test_home_asks_for_a_name_or_photo(shop: dict) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.text_input[0].label == "What are you looking for?"
    html = " ".join(m.value for m in at.markdown)
    assert "genuine before you buy" in html
    for word in ("demo", "fictional", "synthetic", "mock data"):  # no demo notices anywhere shoppers look
        assert word not in html.lower()
    assert "Vaelo" not in html and "bottle" not in html.lower()  # the shopper UI is brand-neutral
    assert "Vaelo" not in at.text_input[0].placeholder


def test_results_group_listings_in_plain_language(shop: dict) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["search"] = "abcdef012345"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    headings = [e.label for e in at.expander]
    assert ":green[**Looks genuine**] (1)" in headings and any(h.startswith(":red[**Avoid**]") for h in headings)
    assert all(e.proto.expanded for e in at.expander if "Avoid" in e.label or "Looks genuine" in e.label)
    assert not any(ch in html + " ".join(headings) for ch in "✓✗≈")  # colour carries the meaning, not glyphs
    assert "abcdef" not in html and "“acme bottle”" not in html  # the search is shown without quotes
    assert "signs of a fake" in html and "official price" in html
    for jargon in ("s.10", "tm_identical", "Confidence", "Expected value", "tokens"):
        assert jargon not in html


def test_unknown_search_id_falls_back_to_home(shop: dict) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["search"] = "../../etc"
    at.run()
    assert not at.exception and at.text_input[0].label == "What are you looking for?"


# ---------------------------------------------------------------- reporting fakes to the brand's legal team


def test_report_puts_a_fake_in_the_lawyers_queue_and_never_submits(shop: dict, tmp_path: Path) -> None:
    from bh.db import CaseStatus, init_db, make_engine
    from bh.enforce.approvals import approve_case
    from ui.data import review_queue
    from ui.shop import report

    main = make_engine(f"sqlite:///{tmp_path / 'main.db'}")
    init_db(main)
    bundle = next(b for b in shop_data.results(shop["run"]) if b.case.id == shop["tm_identical"])
    assert report.status_for(bundle.listing.url, main) is None

    first = report.report(bundle, main)
    assert first.status == "awaiting_approval" and "legal team" in first.message
    queue = review_queue(main)
    assert [b.listing.url for b in queue] == [bundle.listing.url]
    case = queue[0].case
    assert case.shopper_reports == 1 and case.decided_by is None and case.submission_ref is None
    assert "Reported by a shopper" in case.reasoning and case.notice_text

    again = report.report(bundle, main)  # a second report adds to the count, no duplicate case
    assert again.case_id == first.case_id and len(review_queue(main)) == 1
    with Session(main) as s:
        from bh.db import Case
        assert s.get(Case, first.case_id).shopper_reports == 2  # type: ignore[union-attr]

    approve_case(first.case_id, "Priya Shah", engine=main)  # only a named lawyer moves it on
    assert report.status_for(bundle.listing.url, main).status == CaseStatus.approved  # type: ignore[union-attr]


def test_results_offer_a_marketplace_report_only_for_listings_to_avoid(shop: dict) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["search"] = "abcdef012345"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [b.label for b in at.button]
    avoid = [b for b in shop_data.results(shop["run"]) if shop_data.CATEGORY_GROUP[str(b.case.category)] == "avoid"]
    expected = sorted(f"Report to {'MockBay' if b.listing.platform == 'mockbay' else 'TradeLoft'}" for b in avoid)
    assert sorted(l for l in labels if l.startswith("Report to")) == expected and expected
    assert "Your reports" in " ".join(m.value for m in at.sidebar.markdown)


def _avoid_bundle(shop: dict):
    return next(b for b in shop_data.results(shop["run"]) if b.case.id == shop["tm_identical"])


def test_report_form_checks_what_the_marketplace_needs(shop: dict) -> None:
    from ui.shop import reports

    ok = reports.ReportForm("counterfeit", "Listed at a quarter of the official price.", "Sam", "sam@example.com",
                            "shopper", True, True)
    assert ok.errors() == {}
    bad = reports.ReportForm("nope", "short", " ", "sam@", "alien", True, False)
    assert set(bad.errors()) == {"reason", "details", "reporter_name", "reporter_email", "relationship", "declaration"}
    text = reports.draft_details(_avoid_bundle(shop), "Acme")
    assert "official price" in text and "Seller: cheapbottles_88." in text
    assert reports.REASON_FOR_CATEGORY["counterfeit_identical"] in reports.REASONS


def test_sending_a_report_starts_the_agent_and_the_sidebar_links_to_the_submitted_form(
        shop: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from bh.db import ShopperReport
    from ui import workers
    from ui.shop import reports

    started: list[str] = []
    monkeypatch.setattr(workers, "start", lambda name, **kw: started.append(name))
    b = _avoid_bundle(shop)
    form = reports.ReportForm("counterfeit", "Listed at a quarter of the official price.", "Sam Shopper",
                              "sam@example.com", "shopper", True, True)
    rid = reports.create(b, form, "abcdef012345")
    assert started == [f"report-{rid}"]
    with session_scope(shop_data.get_engine()) as s:
        r = s.get(ShopperReport, rid)
        assert r.status == "queued" and r.evidence_path and r.listing_url == b.listing.url  # type: ignore[union-attr]
        r.status, r.report_ref = "submitted", "LR-1A2B3C4D"  # type: ignore[union-attr]
        r.report_url = "http://127.0.0.1:8000/mockbay/reports/LR-1A2B3C4D"  # type: ignore[union-attr]

    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["search"] = "abcdef012345"
    at.run()
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert "LR-1A2B3C4D" in side and 'href="http://127.0.0.1:8000/mockbay/reports/LR-1A2B3C4D"' in side
    assert 'target="_blank"' in side and "Open the submitted form" in side
    main = " ".join(m.value for m in at.main.markdown)
    assert "View the submitted form" in main  # the card shows it too, instead of the Report button


def test_nothing_is_filed_until_the_shopper_submits_the_report(shop: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Viewing results or opening the report form never files anything; only the form's submit button does."""
    from bh.db import ShopperReport
    from ui import workers

    started: list[str] = []
    monkeypatch.setattr(workers, "start", lambda name, **kw: started.append(name))
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["search"] = "abcdef012345"
    at.run()
    report_buttons = [b for b in at.button if b.label.startswith("Report to")]
    assert report_buttons
    report_buttons[0].click().run()  # opens the report form only
    assert not at.exception, [e.value for e in at.exception]
    with Session(shop_data.get_engine()) as s:
        assert s.exec(select(ShopperReport)).all() == []
    assert not [n for n in started if n.startswith("report-")]


# ---------------------------------------------------------------- navigator logic in the shopper search


def test_feed_narrates_the_navigator_and_recipe_replay() -> None:
    tasks = [
        _task(1, "plan_search", None, TaskStatus.succeeded, {"queries": ["acme"]}),
        _task(2, "agent:task", "bazaarly", TaskStatus.running, {"task": "search", "subject": "acme"}),
        _task(3, "agent:consent", "bazaarly", TaskStatus.succeeded, {"button": "Reject non-essential"}),
        _task(4, "agent:model", "bazaarly", TaskStatus.succeeded,
              {"action": "click", "target": "Next", "reason": "More results are on the next page."}),
        _task(5, "agent:action", "bazaarly", TaskStatus.running, {"action": "click", "target": "Next"}),
        _task(6, "recipe:task", "otherbay", TaskStatus.succeeded, {"task": "search"}),
        _task(7, "recipe:step", "otherbay", TaskStatus.running, {"action": "type", "target": "Search"}),
        _task(8, "recipe:learn", "bazaarly", TaskStatus.succeeded, {"learned": True}),
    ]
    f = feed.build(tasks, "Acme", finished=False)
    assert [s for _, s in f.steps][:2] == ["done", "active"]  # a navigator search task is the search step
    now = {n.site: n.doing for n in f.now}
    assert now["bazaarly"] == "Going to the next page"
    assert now["otherbay"] == "Typing in the search box"  # a replayed step reads like any other step
    text = " ".join(f.events)
    assert "the agent decided: Going to the next page, because more results are on the next page." in text
    assert "Reject non-essential" in text
    done = feed.build(tasks + [_task(9, "recipe:step", "otherbay", TaskStatus.succeeded,
                                     {"action": "click", "target": "#tab-details"})], "Acme", finished=True)
    assert not any("Otherbay" in e for e in done.events)  # replayed steps show live, not one log line each
    shopper_text = " ".join(done.events + [n.doing for n in f.now]).lower()
    for internal in ("recipe", "learn", "without ai", "route"):  # how it works stays out of the shopper's view
        assert internal not in shopper_text


def test_shopper_search_uses_claude_only_when_allowed(monkeypatch: pytest.MonkeyPatch, cfg) -> None:
    pytest.importorskip("imagehash")
    from pydantic import SecretStr

    import bh.config
    from bh import shopper
    from bh.config import Secrets

    never = cfg.model_copy(update={"shopper": cfg.shopper.model_copy(update={"use_llm": "never"})})
    assert shopper.make_llm(never) is None
    monkeypatch.setattr(bh.config, "get_secrets", lambda: Secrets())
    assert shopper.make_llm(cfg) is None  # auto, no key
    monkeypatch.setattr(bh.config, "get_secrets", lambda: Secrets(anthropic_api_key=SecretStr("sk-ant-test")))
    assert shopper.make_llm(cfg) is not None  # auto, key set


def test_shopper_search_covers_navigator_sites_by_recipe_or_with_claude(monkeypatch: pytest.MonkeyPatch, cfg) -> None:
    pytest.importorskip("imagehash")
    from types import SimpleNamespace as NS

    from bh import shopper
    from bh.browser import recipes

    navigable = [s for s, c in cfg.sites.items() if c.navigator and c.mode != "read_only"]
    assert navigable, "config should define an adapter-less site for the navigator"
    real = [s for s, c in cfg.sites.items() if c.mode == "read_only"]
    monkeypatch.setattr(recipes.RecipeStore, "load", lambda self, site: None)
    without = shopper.search_sites(cfg, llm=None)
    assert not set(navigable) & set(without) and {"mockbay", "tradeloft"} <= set(without)
    assert set(navigable) <= set(shopper.search_sites(cfg, llm=object()))
    monkeypatch.setattr(recipes.RecipeStore, "load", lambda self, site: NS(search=object(), version=2))
    assert set(navigable) <= set(shopper.search_sites(cfg, llm=None))  # learned recipe: replay with no tokens
    # real (read_only) marketplaces are never searched from the shopper app, with or without Claude or a recipe
    assert not set(real) & set(shopper.search_sites(cfg, llm=object()))
    assert not set(real) & set(shopper.search_sites(cfg, llm=None))
    off = cfg.model_copy(update={"shopper": cfg.shopper.model_copy(update={"navigator_sites": False})})
    assert not set(navigable) & set(shopper.search_sites(off, llm=object()))


def test_every_marketplace_gets_a_fair_share_of_the_listings_opened() -> None:
    pytest.importorskip("imagehash")
    from bh.shopper import BUDGET_REASON, pick_fairly

    def card(site: str, prio: float, kept: bool = True):
        return SimpleNamespace(card=SimpleNamespace(site=site), priority=prio, keep=kept,
                               reasons=[] if kept else [BUDGET_REASON])

    scored = ([card("big", 0.99 - i / 100) for i in range(20)] +
              [card("small", 0.5 - i / 100, kept=False) for i in range(3)] +
              [SimpleNamespace(card=SimpleNamespace(site="small"), priority=0.9, keep=False, reasons=[])])  # no match
    picked = pick_fairly(scored, 6)
    assert [p.card.site for p in picked] == ["big", "small", "big", "small", "big", "small"]
    assert picked[0].priority == 0.99 and picked[1].priority == 0.5  # best first within each site
    assert sum(s.keep for s in scored) == 6 and not scored[-1].keep  # a non-match is never opened


@pytest.mark.parametrize("action,target,expected", [
    ("type", 'input[name="term"]', "Typing in the search box"),
    ("read_results", "article.bz-tile", "Reading the search results"),
    ("read_fields", "title, price", "Reading the listing details"),
    ("read_images", "div.bz-photos img", "Looking at the photos"),
    ("click", "#tab-details", "Opening the “Details” tab"),
    ("click", '[7] tab "Seller"', "Opening the “Seller” tab"),
    ("click", '[12] button "Next"', "Going to the next page"),
    ("click", '[3] link "Copper bottle 900ml"', "Clicking “Copper bottle 900ml”"),
    ("click", "div.bz-x > a", "Clicking through"),
])
def test_browsing_steps_never_show_selectors_or_element_numbers(action: str, target: str, expected: str) -> None:
    assert feed._verb(action, target) == expected
