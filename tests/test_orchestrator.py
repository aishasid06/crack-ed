"""LangGraph orchestrator end to end (SPEC §7.1, Phase 8): mock sites live, LLM mocked, CLIP stubbed."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlmodel import Session, select

from bh.browser.session import BrowserManager
from bh.classify import LLMVerdict
from bh.config import AppConfig
from bh.db import Case, CaseStatus, LLMCall, Run, RunStatus, Task, TaskStatus
from bh.enforce.approvals import approve_case
from bh.enforce.submit import process_approved
from bh.llm import LLMClient
from bh.orchestrator.graph import build_graph
from bh.orchestrator.run import PreflightError, preflight, run_pipeline, summarise
from bh.trace import run_span
from mocksites.store import ReportDraft, Submission, store_session
from tests.test_browser import LiveServer, _start, live, seeded_catalog, secrets  # noqa: F401 (fixtures)
from tests.test_discovery import TinyImageEncoder, _cfg

LAWYER = "A. Lawyer (Vaelo legal)"


class FixedScorer:
    """Stand-in for the fitted scorer: every listing gets the same P(report)."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict(self, feats: list[Any], counterfeit_price_ratio: float) -> list[float]:
        return [self.p] * len(feats)


def _fake_llm(cfg: AppConfig, secrets: Any, engine: Any) -> LLMClient:
    """Borderline vision check answering with a rule outside the candidates. classify must refuse it and
    keep the rules' outcome. Every call is still recorded with model and tokens."""

    class Msgs:
        async def parse(self, **kw: Any) -> Any:
            v = LLMVerdict.model_validate({"rule_id": "not_a_rule", "reasoning": "x", "elements": []})
            return SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
                                   model="claude-opus-5-5", stop_reason="end_turn", parsed_output=v)

    return LLMClient(cfg=cfg, secrets=secrets, client=SimpleNamespace(beta=SimpleNamespace(messages=Msgs())),
                     engine=engine)


def _pipeline_cfg(live: LiveServer, seeded: Path, tmp: Path, *, borderline: tuple[float, float] = (0.4, 0.7),
                  budget_minutes: int = 18) -> AppConfig:
    cfg = _cfg(live.base, seeded, tmp, max_queries=6)
    th = cfg.thresholds.model_copy(update={"borderline_low": borderline[0], "borderline_high": borderline[1]})
    budgets = cfg.budgets.model_copy(update={"review_budget_minutes": budget_minutes})  # ~3 drafts at 6 min each
    return cfg.model_copy(update={"thresholds": th, "budgets": budgets})


def _marketplace_requests(live: LiveServer) -> int:
    """Searches, listing pages and images (not preflight's GET of each site's home page)."""
    return sum(p.startswith(("/echobazaar/", "/tradeloft/")) for _, p in live.requests)


# ---------------------------------------------------------------- graph shape + preflight


def test_graph_nodes_follow_until() -> None:
    full = set(build_graph("draft").get_graph().nodes)
    assert {"load_brand", "generate_queries", "discover", "dedup_and_prefilter", "capture", "classify", "decide",
            "draft", "report"} <= full
    short = set(build_graph("discovery").get_graph().nodes)
    assert "capture" in short and "report" in short and not {"classify", "decide", "draft"} & short
    with pytest.raises(ValueError):
        build_graph("submit")  # type: ignore[arg-type]  # submission is never part of the pipeline


async def test_preflight_fails_before_browsing(live: LiveServer, seeded_catalog: Path, tmp_path: Path,
                                               secrets: Any) -> None:
    cfg = _cfg(live.base, seeded_catalog, tmp_path)
    with pytest.raises(PreflightError, match="protects"):
        await preflight(cfg, "discovery", secrets, None, "some-other-brand")
    missing = cfg.model_copy(update={"models": cfg.models.model_copy(update={"scorer_path": str(tmp_path / "x")})})
    with pytest.raises(PreflightError, match="--until discovery"):
        await preflight(missing, "classify", secrets, None, None)
    assert await preflight(missing, "discovery", secrets, None, cfg.brand) is None  # no scorer needed
    down = _cfg("http://127.0.0.1:9", seeded_catalog, tmp_path)
    with pytest.raises(PreflightError, match="make mock"):
        await preflight(down, "discovery", secrets, None, None)
    with pytest.raises(PreflightError, match="no marketplace"):
        await preflight(cfg, "discovery", secrets, ["nowhere"], None)


# ---------------------------------------------------------------- end to end


async def test_full_pipeline_one_run(live: LiveServer, seeded_catalog: Path, tmp_path: Path, secrets: Any,
                                     engine: Any) -> None:
    # band covers the fixed score → every case gets one (mocked) strong-model call
    cfg = _pipeline_cfg(live, seeded_catalog, tmp_path, borderline=(0.9, 0.96))
    llm = _fake_llm(cfg, secrets, engine)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        state = await run_pipeline(cfg, llm=llm, encoder=TinyImageEncoder(), browser=b, engine=engine,
                                   scorer=FixedScorer(0.95), brand=cfg.brand)
        max_active = dict(b.max_active)
    rid = state["run_id"]

    with Session(engine) as s:
        runs = list(s.exec(select(Run)))
        tasks = list(s.exec(select(Task).where(Task.run_id == rid)))
        cases = list(s.exec(select(Case).where(Case.run_id == rid)))
        calls = list(s.exec(select(LLMCall).where(LLMCall.run_id == rid)))
    # one command → one Run holding every stage
    assert len(runs) == 1 and runs[0].status == RunStatus.finished and runs[0].brand_id is not None
    types = Counter(t.type for t in tasks)
    for stage in ("load_brand", "generate_queries", "prefilter", "decide", "draft_all", "report", "classify_all"):
        assert types[stage] == 1, stage
    assert types["discovery"] == len(state["sites"]) * len(state["queries"])
    assert types["capture"] == len(state["kept_urls"]) == len(state["listing_ids"]) > 0
    assert types["classify"] == len(cases) == len(state["listing_ids"])
    assert all(t.status == TaskStatus.succeeded for t in tasks), [(t.type, t.error) for t in tasks if t.error]
    assert all(d["status"] == "succeeded" for d in state["discovery"])

    # per-site concurrency limits held across the Send fan-out, and work really ran in parallel
    assert max_active["echobazaar"] <= cfg.site("echobazaar").max_concurrency
    assert max_active["tradeloft"] <= cfg.site("tradeloft").max_concurrency
    totals = state["totals"]
    assert totals["max_parallelism"] > 1 and totals["wall_time_s"] > 0
    assert totals["tasks"]["total"] == len(tasks)

    # tokens and model recorded per LLM call and aggregated per model
    assert len(calls) == len(cases) and all(c.task_id for c in calls)
    opus = totals["llm"]["by_model"]["claude-opus-5-5"]
    assert opus["calls"] == len(cases) and opus["input_tokens"] == 1200 * len(cases)
    assert all(c.llm_used and "unknown rule" in c.reasoning for c in cases)  # invented rule refused

    # decision → drafting: only in-budget platform_report cases drafted, steps 1-4 only, never submitted
    by_status = Counter(str(c.status) for c in cases)
    drafted = [c for c in cases if c.status == CaseStatus.drafted]
    assert drafted and len(drafted) == len(state["drafting"]["drafted"]) <= 3
    assert all(c.in_review_budget and c.portal_draft_id and c.declared_by is None for c in drafted)
    assert by_status["submitted"] == by_status["approved"] == 0
    with store_session(live.app.state.store) as s:
        for c in drafted:
            d = s.get(ReportDraft, c.portal_draft_id)
            assert "5" not in d.steps and d.submitted_ref is None
    assert summarise(state)["listings"] == len(state["listing_ids"])

    # the pipeline's output feeds the separate submit-on-approval loop
    approve_case(drafted[0].id, LAWYER, engine)
    async with run_span(None, {}, engine=engine) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        res = await process_approved(run.id, cfg, b, engine)
    assert res["submitted"] == [drafted[0].id]
    with store_session(live.app.state.store) as s:
        sub = s.exec(select(Submission).where(Submission.draft_id == drafted[0].portal_draft_id)).one()
    assert sub.data["5"]["signature"] == LAWYER


@pytest.fixture
def challenge_portal(tmp_path: Path, seeded_catalog: Path) -> Iterator[LiveServer]:
    yield from _start(tmp_path / "chal", seeded_catalog, challenge_probability=1.0)


async def test_challenge_pauses_and_resume_skips_finished_work(challenge_portal: LiveServer, seeded_catalog: Path,
                                                              tmp_path: Path, secrets: Any, engine: Any) -> None:
    live = challenge_portal
    cfg = _pipeline_cfg(live, seeded_catalog, tmp_path)
    enc, scorer = TinyImageEncoder(), FixedScorer(0.95)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        first = await run_pipeline(cfg, encoder=enc, browser=b, engine=engine, scorer=scorer)
    rid = first["run_id"]
    paused = first["drafting"]["waiting_human"]
    assert first["drafting"]["drafted"] == [] and "/portal/verify" in paused["url"]
    assert "--resume" in summarise(first)["next"]
    with Session(engine) as s:
        assert s.get(Case, paused["case_id"]).status == CaseStatus.waiting_human
        waiting = list(s.exec(select(Task).where(Task.run_id == rid, Task.status == TaskStatus.waiting_human)))
    draft_task = next(t for t in waiting if t.type == "draft")  # plus the browser step spans beneath it
    assert "/portal/verify" in draft_task.waiting_url and {t.type for t in waiting} <= {
        "draft", "browser:login", "browser:goto", "browser:action"}

    async with httpx.AsyncClient(base_url=live.base) as human:  # a human passes the check in their own browser
        await human.post("/portal/verify", data={"confirm_human": "yes", "next": "/portal/"})

    before = _marketplace_requests(live)
    async with BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        second = await run_pipeline(cfg, encoder=enc, browser=b, engine=engine, scorer=scorer, resume_run_id=rid)
    assert second["run_id"] == rid
    # nothing searched or captured again: kept cards and listing ids came back from the stored tasks
    assert _marketplace_requests(live) == before
    assert {d["status"] for d in second["discovery"]} == {"skipped"}
    assert {c["status"] for c in second["captures"]} == {"succeeded"}  # listing id restored from the task
    assert second["listing_ids"] == first["listing_ids"] and second["case_ids"] == first["case_ids"]
    assert second["classification"]["classified"] == 0
    assert second["drafting"]["waiting_human"] is None and second["drafting"]["drafted"]

    with Session(engine) as s:
        assert len(list(s.exec(select(Run)))) == 1
        run = s.get(Run, rid)
        tasks = list(s.exec(select(Task).where(Task.run_id == rid)))
        assert s.get(Case, paused["case_id"]).status == CaseStatus.drafted
    assert run.status == RunStatus.finished
    assert Counter(t.type for t in tasks)["discovery"] == len(first["sites"]) * len(first["queries"])  # no dupes
    assert not [t for t in tasks if t.status in (TaskStatus.waiting_human, TaskStatus.running)]
    assert not [t for t in tasks if t.status == TaskStatus.failed]  # a hand-off is not a failure
    stale = [t for t in tasks if t.status == TaskStatus.superseded]
    assert stale and all("superseded" in t.error and t.finished_at for t in stale)
    assert next(t for t in tasks if t.id == draft_task.id).status == TaskStatus.succeeded

    with pytest.raises(PreflightError, match="not found"):
        await run_pipeline(cfg, encoder=enc, engine=engine, scorer=scorer, resume_run_id=rid + 99,
                           secrets=secrets)
