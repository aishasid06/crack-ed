from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from bh.db import (
    Action, Brand, Case, CaseStatus, Category, Listing, MarkType, Run, Task, TradeMark, session_scope,
)

EXPECTED_TABLES = {"brand", "trademark", "product", "authorisedseller", "listing", "features", "case",
                   "task", "llmcall", "run"}


def test_tables_created(engine: Engine) -> None:
    assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())


def test_sqlite_wal_and_fk(engine: Engine) -> None:
    with engine.connect() as c:
        assert c.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert c.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_json_and_enum_roundtrip(engine: Engine) -> None:
    with session_scope(engine) as s:
        b = Brand(name="Vaelo", slug="vaelo")
        s.add(b)
        s.flush()
        s.add(TradeMark(brand_id=b.id, mark_text="VAELO", mark_type=MarkType.word,
                        registration_number="UK00004123456", classes=[21]))
        lst = Listing(platform="echobazaar", url="http://127.0.0.1:8000/echobazaar/item/1",
                      image_urls=["a.jpg", "b.jpg"], price=6.99)
        s.add(lst)
        s.flush()
        s.add(Case(listing_id=lst.id, category=Category.counterfeit_identical, confidence=0.91,
                   rule_id="tm_identical", action=Action.platform_report,
                   legal_basis=[{"statute": "Trade Marks Act 1994", "section": "s.10(1)"}],
                   legal_elements={"not_exhausted": {"status": "met", "evidence": "25% of RRP"}}))

    with Session(engine) as s:
        tm = s.exec(select(TradeMark)).one()
        assert tm.classes == [21] and tm.mark_type == MarkType.word
        case = s.exec(select(Case)).one()
        assert case.status == CaseStatus.classified
        assert case.legal_elements["not_exhausted"]["status"] == "met"
        assert s.exec(select(Listing)).one().image_urls == ["a.jpg", "b.jpg"]


def test_listing_url_unique(engine: Engine) -> None:
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            s.add(Listing(platform="echobazaar", url="u"))
            s.add(Listing(platform="echobazaar", url="u"))


def test_task_dedup_key_unique_per_run(engine: Engine) -> None:
    with session_scope(engine) as s:
        r1, r2 = Run(), Run()
        s.add_all([r1, r2])
        s.flush()
        s.add(Task(run_id=r1.id, type="discovery", dedup_key="echobazaar:vaelo"))
        s.add(Task(run_id=r2.id, type="discovery", dedup_key="echobazaar:vaelo"))  # other run: fine
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            run = s.exec(select(Run)).first()
            s.add(Task(run_id=run.id, type="discovery", dedup_key="echobazaar:vaelo"))


def test_session_scope_rolls_back(engine: Engine) -> None:
    with pytest.raises(RuntimeError):
        with session_scope(engine) as s:
            s.add(Brand(name="X", slug="x"))
            raise RuntimeError("boom")
    with Session(engine) as s:
        assert s.exec(select(Brand)).first() is None
