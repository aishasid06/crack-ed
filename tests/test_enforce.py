"""Decision, drafting, approval and submission against the live mock portal (SPEC §7.1, §7.4, Phase 7)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from sqlmodel import Session, select

from bh.brand import load_brand_assets
from bh.browser.session import BrowserManager
from bh.config import AppConfig
from bh.db import Action, Case, CaseStatus, Features, Listing, Task, TaskStatus, session_scope
from bh.decide import decide, decide_cases, select_within_budget
from bh.enforce.approvals import DecisionError, approve_case, confirm_final, reject_case, resume_case
from bh.enforce.portal import draft_cases
from bh.enforce.submit import compare_review, process_approved
from bh.notice import assert_factual, build_plan, notice_text
from bh.rules import get_rules
from bh.trace import resume_task, run_span
from mocksites.store import ReportDraft, Submission, store_session
from tests.test_browser import LiveServer, _start, live, seeded_catalog, secrets  # noqa: F401 (fixtures)
from tests.test_discovery import _cfg

RULES = get_rules()
LAWYER = "A. Lawyer (Vaelo legal)"


def _elements(rule_id: str, unknown: str | None = None) -> dict[str, Any]:
    els = {n: {"status": "met", "evidence": f"{n} evidence"} for n in RULES.rule(rule_id).requires}
    if unknown:
        els[unknown] = {"status": "unknown", "evidence": "could not determine"}
    return els


def _basis(rule_id: str) -> list[dict[str, Any]]:
    r = RULES.rule(rule_id)
    return [{"rule_id": r.id, **r.legal_basis.model_dump(), "citation": r.legal_basis.citation()}]


@pytest.fixture
def cases(live: LiveServer, seeded_catalog: Path, tmp_path: Path, engine: Any) -> dict[str, int]:
    """Listings from the live catalog + classified cases: two counterfeits (TM), one copied photo (CDPA),
    one low-confidence similar sign, one with an unknown element, one grey import."""
    specs = [("tm_a", "tm_identical", 0.95, None), ("tm_b", "tm_identical", 0.92, None),
             ("cp", "copied_photos", 0.9, None), ("sim_low", "tm_similar", 0.5, None),
             ("unk", "tm_identical", 0.97, "not_exhausted"), ("grey", "grey_import", 0.3, None)]
    out = {}
    listings = live.catalog.for_site("mockbay")
    with session_scope(engine) as s:
        for (name, rule, conf, unknown), l in zip(specs, listings, strict=False):
            shot = tmp_path / f"{name}.png"
            Image.new("RGB", (400, 300), (200, 210, 220)).save(shot)
            row = Listing(platform="mockbay", url=f"{live.base}/mockbay/item/{l.id}", title=l.title, price=l.price,
                          currency="GBP", seller_name=l.seller_name, seller_location=l.seller_location,
                          quantity=l.quantity, screenshot_path=str(shot))
            s.add(row)
            s.flush()
            s.add(Features(listing_id=row.id, clip_sim=0.9))
            r = RULES.rule(rule)
            c = Case(listing_id=row.id, run_id=None, category=r.category, confidence=conf, rule_id=rule,
                     action=Action(r.action), legal_basis=_basis(rule), legal_elements=_elements(rule, unknown),
                     reasoning=f"{rule}.")
            s.add(c)
            s.flush()
            out[name] = c.id
    return out


def _portal_cfg(live: LiveServer, seeded: Path, tmp: Path, **enforce: Any) -> AppConfig:
    cfg = _cfg(live.base, seeded, tmp)
    return cfg.model_copy(update={"enforce": cfg.enforce.model_copy(update={"draft_only_in_budget": False,
                                                                             **enforce})})


def _status(engine: Any, cid: int) -> Case:
    with Session(engine) as s:
        return s.get(Case, cid)


# ---------------------------------------------------------------- decision


def test_decide_ev_downgrades_and_queue(live: LiveServer, seeded_catalog: Path, tmp_path: Path, engine: Any,
                                        cases: dict[str, int]) -> None:
    cfg = _portal_cfg(live, seeded_catalog, tmp_path)
    summary = decide_cases(list(cases.values()), cfg, engine)
    c = {k: _status(engine, v) for k, v in cases.items()}
    assert c["tm_a"].status == CaseStatus.awaiting_approval and c["tm_a"].action == Action.platform_report
    assert c["sim_low"].action == Action.escalate and "confidence 0.50 < 0.70" in c["sim_low"].reasoning
    assert c["unk"].action == Action.escalate and "unknown element(s): not_exhausted" in c["unk"].reasoning
    assert c["grey"].action == Action.escalate and c["grey"].status == CaseStatus.classified
    assert summary["downgraded"] == 2 and summary["awaiting_approval"] == 3
    # EV = price × max(qty,1) × clip × p_success × confidence − cost
    with Session(engine) as s:
        l = s.get(Listing, c["tm_a"].listing_id)
    p = cfg.decision.p_success_for("mockbay", "tm_identical")
    expected = round(l.price * max(l.quantity, 1) * 0.9 * p * 0.95 - cfg.decision.cost_for("platform_report"), 2)
    assert c["tm_a"].expected_value == expected
    queue = sorted((x for x in c.values() if x.status == CaseStatus.awaiting_approval), key=lambda x: x.review_rank)
    assert [x.expected_value for x in queue] == sorted((x.expected_value for x in queue), reverse=True)
    # notice: factual, names URL, right and provision
    n = c["tm_a"].notice_text
    assert l.url in n and "UK00004123456" in n and "Trade Marks Act 1994 s.10(1)" in n
    assert "Copyright, Designs and Patents Act 1988 ss.16-17, 20" in c["cp"].notice_text


def test_review_budget_is_greedy_by_ev_per_minute(cfg: AppConfig) -> None:
    mk = lambda i, ev: Case(id=i, listing_id=i, category="counterfeit_identical", confidence=1, rule_id="tm_identical",  # noqa: E731
                            action=Action.platform_report, expected_value=ev)
    small = cfg.model_copy(update={"budgets": cfg.budgets.model_copy(update={"review_budget_minutes": 12})})
    assert select_within_budget([mk(1, 50), mk(2, 500), mk(3, 100), mk(4, -5)], small) == {2, 3}


@pytest.mark.parametrize("text", ["We will take legal action", "you may be liable for damages",
                                  "Cease and desist immediately", "court proceedings will follow"])
def test_notice_rejects_threats(text: str) -> None:
    with pytest.raises(ValueError):
        assert_factual(text)


# ---------------------------------------------------------------- drafting, approval, submission


async def test_draft_approve_submit_and_reject(live: LiveServer, seeded_catalog: Path, tmp_path: Path, engine: Any,
                                               secrets: Any, cases: dict[str, int]) -> None:
    cfg = _portal_cfg(live, seeded_catalog, tmp_path)
    decide_cases(list(cases.values()), cfg, engine)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        drafted = await draft_cases(run.id, cfg, b, engine)
    assert {cid for cid, _ in drafted["drafted"]} == {cases["tm_a"], cases["tm_b"], cases["cp"]}
    for k in ("tm_a", "tm_b", "cp"):
        c = _status(engine, cases[k])
        assert c.status == CaseStatus.drafted and c.portal_draft_id and c.declared_by is None
        with store_session(live.app.state.store) as s:
            d = s.get(ReportDraft, c.portal_draft_id)
        assert set(d.steps) == {"1", "2", "3"} and len(d.evidence) == 1  # steps 1-4 done, declarations untouched
        assert d.steps["3"]["items"][0]["url"].endswith(Path(_listing_url(engine, c)).name)
    with store_session(live.app.state.store) as s:
        cp = s.get(ReportDraft, _status(engine, cases["cp"]).portal_draft_id)
    assert cp.steps["2"]["right_type"] == "copyright" and cp.steps["3"]["items"][0]["statute"].startswith("Copyright")

    with pytest.raises(DecisionError):
        approve_case(cases["tm_a"], "  ", engine)  # a named human is required
    approve_case(cases["tm_a"], LAWYER, engine)
    approve_case(cases["cp"], LAWYER, engine)
    reject_case(cases["tm_b"], LAWYER, engine)

    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        res = await process_approved(run.id, cfg, b, engine)
    assert sorted(res["submitted"]) == sorted([cases["tm_a"], cases["cp"]]) and not res["waiting_human"]

    a, rej = _status(engine, cases["tm_a"]), _status(engine, cases["tm_b"])
    assert a.status == CaseStatus.submitted and a.submission_ref.startswith("IPR-") and a.declared_by == LAWYER
    assert a.declared_at and a.review_verified_at and a.review_diff == {}
    assert rej.status == CaseStatus.rejected and rej.declared_by is None and rej.submission_ref is None
    mine = {_status(engine, cases[k]).portal_draft_id for k in ("tm_a", "tm_b", "cp")}
    with store_session(live.app.state.store) as s:  # the live server is shared across tests: filter to ours
        subs = [x for x in s.exec(select(Submission)) if x.draft_id in mine]
        rej_draft = s.get(ReportDraft, rej.portal_draft_id)
    urls = {it["url"] for sub in subs for it in sub.data["3"]["items"]}
    assert len(subs) == 2 and _listing_url(engine, a) in urls and _listing_url(engine, rej) not in urls
    assert all(sub.data["5"]["signature"] == LAWYER for sub in subs)
    assert "5" not in rej_draft.steps and rej_draft.submitted_ref is None  # rejected: never signed, never sent

    # visible on /portal/submissions
    async with httpx.AsyncClient(base_url=live.base) as h:
        await h.post("/portal/login", data={"username": secrets.mock_portal_user.get_secret_value(),
                                            "password": secrets.mock_portal_pass.get_secret_value()})
        page = (await h.get("/portal/submissions")).text
    assert a.submission_ref in page and _listing_url(engine, a) in page and _listing_url(engine, rej) not in page

    # a second cycle does nothing more (idempotent)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        assert (await process_approved(run.id, cfg, b, engine))["submitted"] == []


def _listing_url(engine: Any, c: Case) -> str:
    with Session(engine) as s:
        return s.get(Listing, c.listing_id).url


async def test_review_mismatch_blocks_submission(live: LiveServer, seeded_catalog: Path, tmp_path: Path,
                                                 engine: Any, secrets: Any, cases: dict[str, int]) -> None:
    cfg = _portal_cfg(live, seeded_catalog, tmp_path)
    decide_cases([cases["tm_a"]], cfg, engine)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        await draft_cases(run.id, cfg, b, engine, [cases["tm_a"]])
    c = _status(engine, cases["tm_a"])
    with store_session(live.app.state.store) as s:  # someone edits the draft: wrong URL and section
        d = s.get(ReportDraft, c.portal_draft_id)
        steps = dict(d.steps)
        steps["3"] = {"items": [{**steps["3"]["items"][0], "url": f"{live.base}/mockbay/item/999999",
                                 "section": "s.10(2)"}]}
        d.steps = steps
    approve_case(c.id, LAWYER, engine)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        res = await process_approved(run.id, cfg, b, engine)
    c = _status(engine, c.id)
    assert c.status == CaseStatus.waiting_human and res["submitted"] == []
    assert set(c.review_diff) == {"listing_urls", "provisions"}
    assert c.review_diff["provisions"] == {"expected": ["Trade Marks Act 1994 s.10(1)"],
                                           "actual": ["Trade Marks Act 1994 s.10(2)"]}
    with store_session(live.app.state.store) as s:
        assert [x for x in s.exec(select(Submission)) if x.draft_id == c.portal_draft_id] == []
    with Session(engine) as s:
        t = s.exec(select(Task).where(Task.type == "submit")).one()
    assert t.status == TaskStatus.waiting_human and t.payload["review_diff"]

    # a human looks at the diff and resumes: the draft is discarded and redone from the approved case
    assert resume_case(c.id, engine) == CaseStatus.approved
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        res = await process_approved(run.id, cfg, b, engine)
    assert res["submitted"] == [c.id] and _status(engine, c.id).status == CaseStatus.submitted


async def test_require_final_confirm(live: LiveServer, seeded_catalog: Path, tmp_path: Path, engine: Any,
                                     secrets: Any, cases: dict[str, int]) -> None:
    cfg = _portal_cfg(live, seeded_catalog, tmp_path, require_final_confirm=True)
    decide_cases([cases["tm_a"]], cfg, engine)
    approve_case(cases["tm_a"], LAWYER, engine)  # approve before drafting: the loop drafts first
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        res = await process_approved(run.id, cfg, b, engine)
    c = _status(engine, cases["tm_a"])
    assert res["awaiting_final_confirm"] == [c.id] and c.status == CaseStatus.approved
    assert c.review_verified_at and c.submission_ref is None and c.portal_draft_id
    with pytest.raises(DecisionError):
        confirm_final(c.id, "Someone Else", engine)
    confirm_final(c.id, LAWYER, engine)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        res = await process_approved(run.id, cfg, b, engine)
    assert res["submitted"] == [c.id] and _status(engine, c.id).status == CaseStatus.submitted


@pytest.fixture
def challenge_portal(tmp_path: Path, seeded_catalog: Path) -> Iterator[LiveServer]:
    yield from _start(tmp_path / "chal", seeded_catalog, challenge_probability=1.0)


async def test_challenge_during_drafting_waits_for_human(challenge_portal: LiveServer, seeded_catalog: Path,
                                                         tmp_path: Path, engine: Any, secrets: Any,
                                                         cases: dict[str, int]) -> None:
    live = challenge_portal
    cfg = _portal_cfg(live, seeded_catalog, tmp_path)
    decide_cases(list(cases.values()), cfg, engine)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        out = await draft_cases(run.id, cfg, b, engine)
    assert out["drafted"] == [] and "/portal/verify" in out["waiting_human"]["url"]
    paused = out["waiting_human"]["case_id"]
    assert _status(engine, paused).status == CaseStatus.waiting_human
    others = [cid for cid in (cases["tm_a"], cases["tm_b"], cases["cp"]) if cid != paused]
    assert all(_status(engine, cid).status == CaseStatus.awaiting_approval for cid in others)  # loop stopped

    async with httpx.AsyncClient(base_url=live.base) as human:  # human verifies in their own browser
        await human.post("/portal/verify", data={"confirm_human": "yes", "next": "/portal/"})
    with Session(engine) as s:
        task = s.exec(select(Task).where(Task.type == "draft", Task.status == TaskStatus.waiting_human)).one()
    resume_task(task.id, engine)
    assert resume_case(paused, engine) == CaseStatus.awaiting_approval
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        out = await draft_cases(run.id, cfg, b, engine)
    assert len(out["drafted"]) == 3 and out["waiting_human"] is None


def test_approval_state_rules(engine: Any, cases: dict[str, int]) -> None:
    with pytest.raises(DecisionError):
        approve_case(cases["tm_a"], LAWYER, engine)  # still 'classified' (not decided)
    with pytest.raises(DecisionError):
        resume_case(cases["tm_a"], engine)
    reject_case(cases["grey"], LAWYER, engine)
    with pytest.raises(DecisionError):
        approve_case(cases["grey"], LAWYER, engine)


def test_compare_review_reports_every_field(live: LiveServer, seeded_catalog: Path, tmp_path: Path, engine: Any,
                                            cases: dict[str, int]) -> None:
    cfg = _portal_cfg(live, seeded_catalog, tmp_path)
    assets = load_brand_assets(Path(cfg.paths.brand_dir))
    with Session(engine) as s:
        c = s.get(Case, cases["tm_a"])
        plan = build_plan(c, s.get(Listing, c.listing_id), assets)
    good = {"right": {"Type": "Trade mark", "Registration": plan.registration_number, "Classes": "21",
                      "Territory": "UK"}, "rows": [[plan.items[0].url, "Trade Marks Act 1994 s.10(1)",
                                                    " ".join(plan.items[0].details.split())]],
            "evidence": 1, "owner": {"Owner": plan.owner_name}, "declarations": f"Declarations ... Signed: {LAWYER}",
            "submit_button": True}
    assert compare_review(plan, good, LAWYER) == {}
    bad = {**good, "right": {**good["right"], "Registration": "UK00000000000"}, "evidence": 0,
           "declarations": "Signed: Mallory"}
    assert set(compare_review(plan, bad, LAWYER)) == {"registration_number", "evidence_count", "signatory"}
    assert "Trade Marks Act 1994" in notice_text(plan, assets)


def test_notice_states_facts_not_analysis(live: LiveServer, seeded_catalog: Path, tmp_path: Path, engine: Any,
                                          cases: dict[str, int]) -> None:
    cfg = _portal_cfg(live, seeded_catalog, tmp_path)
    decide_cases(list(cases.values()), cfg, engine)
    for k in ("tm_a", "cp"):
        n = _status(engine, cases[k]).notice_text
        for internal in ("cluster", "own_photo", "logo not checked", "RRP", "CLIP", "pHash", "evidence"):
            assert internal not in n.split("Evidence:")[0], (k, internal)
        assert "VAELO, VAELO" not in n
    assert "photographs reproduced" in _status(engine, cases["cp"]).notice_text


async def test_redeciding_keeps_one_queue(live: LiveServer, seeded_catalog: Path, tmp_path: Path, engine: Any,
                                          secrets: Any, cases: dict[str, int]) -> None:
    cfg = _portal_cfg(live, seeded_catalog, tmp_path)
    decide_cases(list(cases.values()), cfg, engine)
    async with run_span(None, {}) as run, BrowserManager(cfg, secrets, sessions_dir=tmp_path / "s") as b:
        await draft_cases(run.id, cfg, b, engine, [cases["tm_a"]])
    decide_cases(list(cases.values()), cfg, engine)  # e.g. after new cases arrive
    ranks = [c.review_rank for c in (_status(engine, v) for v in cases.values())
             if c.status in (CaseStatus.awaiting_approval, CaseStatus.drafted)]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_approvals_cli(engine: Any, cases: dict[str, int], capsys: pytest.CaptureFixture[str],
                       live: LiveServer, seeded_catalog: Path, tmp_path: Path) -> None:
    from bh.enforce.approvals import main

    decide_cases(list(cases.values()), _portal_cfg(live, seeded_catalog, tmp_path), engine)
    assert main(["approve", str(cases["tm_a"]), "--by", LAWYER], engine) == 0
    out = capsys.readouterr().out
    assert f"case {cases['tm_a']}: approved by {LAWYER}" in out
    c = _status(engine, cases["tm_a"])
    assert c.status == CaseStatus.approved and c.decided_by == LAWYER and c.decided_at is not None

    assert main(["reject", str(cases["tm_b"]), "--by", LAWYER], engine) == 0
    assert _status(engine, cases["tm_b"]).status == CaseStatus.rejected

    assert main(["approve", str(cases["tm_b"]), "--by", LAWYER], engine) == 2  # rejected cannot be approved
    assert "error:" in capsys.readouterr().err
    assert main(["approve", str(cases["cp"]), "--by", "   "], engine) == 2  # a named human is required
    assert main(["resume", str(cases["cp"])], engine) == 2  # not waiting for a human
    with pytest.raises(SystemExit):
        main(["approve", str(cases["cp"])], engine)  # --by is mandatory
