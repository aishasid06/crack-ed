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
