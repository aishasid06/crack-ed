"""Replaying a saved shopper search: matching, a faithful step-by-step copy, and truthful records underneath."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.db import Case, LLMCall, Run, RunStatus, Task, session_scope
from tests.ui_fixture import populate
from ui import workers
from ui.shop import replay


@pytest.fixture
def saved(engine: Engine, tmp_path: Path) -> dict[str, object]:
    ids = populate(engine)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"not really a jpeg, but hashable")
    with session_scope(engine) as s:
        run = s.get(Run, ids["run"])
        assert run is not None
        run.config_snapshot = {"stage": "shopper", "search_id": "orig00000001", "query": "Acme Copper  Bottle",
                               "has_photo": False}
        run.status = RunStatus.finished
        s.add(Run(status=RunStatus.finished, config_snapshot={
            "stage": "shopper", "search_id": "orig00000002", "query": "acme bottle", "has_photo": True,
            "photo_sha256": replay._sha(photo)}))
    return {**ids, "photo": photo}


def test_a_repeated_search_finds_its_saved_run(saved: dict, engine: Engine, tmp_path: Path) -> None:
    assert replay.find_source("acme copper bottle", None, engine) == saved["run"]  # case and spacing ignored
    assert replay.find_source("acme copper bottle!", None, engine) is None  # different words: run it for real
    photo_run = replay.find_source("Acme bottle", saved["photo"], engine)
    assert photo_run is not None and photo_run != saved["run"]
    other = tmp_path / "other.jpg"
    other.write_bytes(b"a different photo")
    assert replay.find_source("acme bottle", other, engine) is None  # same words, different photo
    assert replay.find_source("acme bottle", None, engine) is None  # the saved one had a photo


def test_replay_recreates_the_run_step_by_step_without_llm_calls(saved: dict, engine: Engine) -> None:
    src = saved["run"]
    rid = replay.replay("replay000001", src, engine, seconds=1.0, tick=0.05)
    with Session(engine) as s:
        new, old = s.get(Run, rid), s.get(Run, src)
        assert new is not None and old is not None
        assert new.status == RunStatus.finished and new.config_snapshot["replay_of"] == src
        assert new.config_snapshot["search_id"] == "replay000001"
        tasks = lambda r: Counter((t.type, t.dedup_key, str(t.status)) for t in s.exec(select(Task).where(Task.run_id == r)))  # noqa: E731
        assert tasks(rid) == tasks(src)  # every step, with its outcome; none left running
        cases = lambda r: sorted((c.listing_id, str(c.category), c.rule_id) for c in s.exec(select(Case).where(Case.run_id == r)))  # noqa: E731
        assert cases(rid) == cases(src)
        assert s.exec(select(LLMCall).where(LLMCall.run_id == src)).all()  # the saved run made a call...
        assert s.exec(select(LLMCall).where(LLMCall.run_id == rid)).all() == []  # ...the replay records none
        parents = {t.dedup_key: t.parent_id for t in s.exec(select(Task).where(Task.run_id == rid))}
        ids = {t.id for t in s.exec(select(Task).where(Task.run_id == rid))}
        assert all(p is None or p in ids for p in parents.values())  # parent links point inside the new run
    assert rid not in [r["run"] for r in replay.saved_searches(engine)]  # a replay is never itself a source


def test_the_app_launches_the_replayer_for_a_saved_search() -> None:
    cmd = workers.command_for("search-abc123def456", text="acme", photo=None, replay_of=7)
    assert cmd[1:] == ["-m", "ui.shop.replay", "--search-id", "abc123def456", "--source", "7"]
    assert workers.command_for("search-abc123def456", text="acme", photo=None, replay_of=None)[2] == "bh.shopper"
