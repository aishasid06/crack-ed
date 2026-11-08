"""Streamlit UI (Phase 9): every page renders against a populated DB, and the human decisions the UI
writes (approve, reject, Done) go through the approval / trace functions with the reviewer's name."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from bh.db import Case, CaseStatus, Task, TaskStatus  # noqa: E402
from tests.ui_fixture import populate  # noqa: E402
from ui import data, workers  # noqa: E402

PAGES = ["run", "review", "human", "cases", "trace", "evaluation"]


def _page_script(name: str, reviewer: str) -> None:
    import importlib

    import streamlit as st

    from ui import theme

    st.session_state.setdefault("reviewer", reviewer)
    theme.inject()
    importlib.import_module(f"ui.views.{name}").render()


@pytest.fixture
def ids(engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    monkeypatch.setattr(workers, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(workers, "mock_sites_up", lambda timeout=0.6: False)
    return populate(engine)


def _app(name: str, reviewer: str = "") -> AppTest:
    at = AppTest.from_function(_page_script, args=(name, reviewer), default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(ids: dict[str, int], page: str) -> None:
    _app(page, "A. Lawyer")


def test_queue_is_ordered_by_ev_and_latest_case_per_listing(ids: dict[str, int]) -> None:
    q = data.review_queue()
    assert [b.case.status for b in q] == [CaseStatus.drafted, CaseStatus.awaiting_approval]
    assert [b.case.expected_value for b in q] == sorted((b.case.expected_value for b in q), reverse=True)


def test_review_shows_citation_elements_notice_and_declarations(ids: dict[str, int]) -> None:
    at = _app("review", "A. Lawyer")
    html = " ".join(m.value for m in at.markdown)
    assert "Trade Marks Act 1994, s.10(1)" in html
    assert "Not exhausted" in html and "23% of RRP" in html
    assert "Trading Standards" in html  # overlay
    assert "Notice of infringing listing" in html
    assert "good-faith belief" in html and "Signed: A. Lawyer" in html
    assert "not legal advice" in html


def test_approve_requires_a_name(ids: dict[str, int]) -> None:
    at = _app("review", "")
    approve = next(b for b in at.button if b.label == "Approve and submit")
    assert approve.disabled


def test_approve_and_submit_records_the_reviewer(engine: Engine, ids: dict[str, int]) -> None:
    at = _app("review", "Priya Shah")
    next(b for b in at.button if b.label == "Approve and submit").click().run()
    with Session(engine) as s:
        c = s.get(Case, ids["tm_identical"])
        assert c is not None and c.status == CaseStatus.approved and c.decided_by == "Priya Shah"
        assert c.decided_at is not None and c.declared_by is None  # declarations are the submit loop's job


def test_reject_never_approves(engine: Engine, ids: dict[str, int]) -> None:
    at = _app("review", "Priya Shah")
    next(b for b in at.button if b.label == "Reject").click().run()
    with Session(engine) as s:
        c = s.get(Case, ids["tm_identical"])
        assert c is not None and c.status == CaseStatus.rejected and c.submission_ref is None


def test_done_releases_the_challenge_task_and_its_case(engine: Engine, ids: dict[str, int]) -> None:
    at = _app("human", "Priya Shah")
    html = " ".join(m.value for m in at.markdown)
    assert "UK00004123457" in html  # review-page diff is shown
    next(b for b in at.button if b.label == "Done").click().run()
    with Session(engine) as s:
        t = s.get(Task, ids["challenge_task"])
        assert t is not None and t.status == TaskStatus.pending
        c = s.get(Case, t.payload["case_id"])
        assert c is not None and c.status == CaseStatus.approved  # it had been approved before the challenge


def test_task_tree_puts_children_under_parents(ids: dict[str, int]) -> None:
    tree = data.task_tree(data.tasks_for_runs([ids["run"]]))
    depth = {t.dedup_key: d for d, t in tree}
    keys = [t.dedup_key for _, t in tree]
    assert depth["fan_out"] == 0 and depth["discovery:echobazaar:q0"] == 1
    assert keys.index("fan_out") < keys.index("discovery:echobazaar:q0")


def test_pipeline_command_prefers_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers, "has_module", lambda name: name == "bh.orchestrator.run")
    assert workers.pipeline_command("vaelo", False)[1:] == ["-m", "bh.orchestrator.run", "--brand", "vaelo"]
    monkeypatch.setattr(workers, "has_module", lambda name: False)
    assert workers.pipeline_command("vaelo", True)[1:] == ["-m", "ui.pipeline", "--brand", "vaelo", "--llm"]


def test_worker_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(workers, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(workers, "command_for", lambda name, **kw: [sys.executable, "-c", "print('[bh] exit 0')"])
    w = workers.start("pipeline")
    assert w.pid
    import time

    for _ in range(50):
        if not workers.status("pipeline").alive:
            break
        time.sleep(0.1)
    st = workers.status("pipeline")
    assert not st.alive and st.label == "finished"
    assert "[bh] exit 0" in workers.tail("pipeline")


def test_evaluation_renders_the_eval_report(ids: dict[str, int], tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """The page reads what eval/evaluate.py writes: built here with eval.metrics itself, so a format change on
    either side fails this test."""
    import json

    from eval import metrics as M
    from ui.views import evaluation

    rows = [M.Row("a", "echobazaar", "easy", "counterfeit_identical", "platform_report", True, True,
                  "counterfeit_identical", "platform_report", 0.95),
            M.Row("b", "tradeloft", "hard", "lookalike", "escalate", False, True, "lookalike", "escalate", 0.40),
            M.Row("c", "echobazaar", "hard", "unrelated", "ignore", False, False, M.NOT_CAPTURED, M.NOT_CAPTURED, 0.0),
            M.Row("d", "tradeloft", "easy", "copied_photos", "platform_report", True, True, "copied_photos",
                  "platform_report", 0.81)]
    cats = ["counterfeit_identical", "copied_photos", "lookalike", "unrelated"]
    report = {"run": {"run_id": 7}, "split": "test", "generated_at": "2026-11-08T10:00:00",
              "headline": M.headline(rows, 50), "by_site": M.breakdown(rows, "site", 50),
              "by_difficulty": M.breakdown(rows, "difficulty", 50), "classification": M.classification(rows, cats),
              "pr_curve": M.pr_curve(rows, 0.7)}
    folder = tmp_path / "eval" / "run_7_test"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    (folder / "report.md").write_text("# Evaluation of run 7", encoding="utf-8")
    monkeypatch.setattr(evaluation, "eval_dir", lambda: tmp_path / "eval")
    at = _app("evaluation")
    values = dict(zip([m.label for m in at.metric], [m.value for m in at.metric]))
    assert values["Discovery recall"] == "100%" and values["Precision on “should report”"] == "100%"
    assert values["False-alarm rate"] == "0%"
    assert any("Run 7" in c.value for c in at.caption)


def test_run_page_passes_the_orchestrator_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers, "has_module", lambda name: name == "bh.orchestrator.run")
    cmd = workers.command_for("pipeline", brand="vaelo", use_llm=True, navigator=True, until="classify",
                              sites=["echobazaar", "bazaarly"], brand_dir="data/brands/acme")
    assert cmd[1:] == ["-m", "bh.orchestrator.run", "--brand", "vaelo", "--brand-dir", "data/brands/acme",
                       "--until", "classify", "--sites", "echobazaar", "bazaarly", "--navigator", "--llm"]


def test_orchestrator_stage_chips_follow_the_runs_tasks() -> None:
    from types import SimpleNamespace as NS

    from ui.views.run import orchestrator_stages

    tasks = [NS(type="discovery", status="succeeded"), NS(type="classify_all", status="running"),
             NS(type="draft_all", status="waiting_human")]
    assert [s["status"] for s in orchestrator_stages(tasks, finished=False)] == \
        ["succeeded", "running", "pending", "waiting_human", "pending"]
