"""Decision (SPEC §7.4): action, expected value, review queue.

    harm           = price × max(quantity, 1) × clip_sim
    p_success      = config prior for (platform, rule_id)
    cost           = lawyer-minutes for the primary (+ secondary) actions × rate
    expected_value = harm × p_success × confidence − cost

The action comes from the matched rule; platform_report is downgraded to escalate when confidence is
below the rule's min_confidence or any required element is unknown. platform_report cases move to
``awaiting_approval``, ranked by EV; a greedy EV/cost pass marks those that fit review_budget_minutes.

    python -m bh.decide [--run latest|<id>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.brand import config_for_run, load_brand_assets
from bh.config import AppConfig, get_config
from bh.db import Action, Case, CaseStatus, Features, Listing, get_engine, session_scope
from bh.notice import NotAReportableCase, build_plan, notice_text
from bh.rules import RuleSet, get_rules
from bh.trace import run_span, task_span

DECIDABLE = (CaseStatus.classified, CaseStatus.awaiting_approval)


@dataclass
class Decision:
    action: Action
    downgrade_reason: str | None
    harm: float
    p_success: float
    cost: float
    expected_value: float


def decide(case: Case, listing: Listing, features: Features | None, rules: RuleSet, cfg: AppConfig) -> Decision:
    rule = rules.rule(case.rule_id)
    action = Action(rule.action)
    reason = None
    if action == Action.platform_report:
        unknown = [n for n in rule.requires if case.legal_elements.get(n, {}).get("status") == "unknown"]
        if rule.min_confidence is not None and case.confidence < rule.min_confidence:
            action, reason = Action.escalate, f"confidence {case.confidence:.2f} < {rule.min_confidence:.2f}"
        elif unknown:
            action, reason = Action.escalate, f"unknown element(s): {', '.join(unknown)}"
    clip = (features.clip_sim if features and features.clip_sim is not None else 0.0)
    harm = (listing.price or 0.0) * max(listing.quantity or 1, 1) * clip
    p = cfg.decision.p_success_for(listing.platform, rule.id)
    cost = cfg.decision.cost_for(str(action)) + sum(cfg.decision.cost_for(a) for a in case.secondary_actions)
    ev = harm * p * case.confidence - cost
    return Decision(action, reason, round(harm, 2), p, round(cost, 2), round(ev, 2))


def select_within_budget(cases: list[Case], cfg: AppConfig) -> set[int]:
    """Greedy by EV per lawyer-minute, positive EV only, until review_budget_minutes is used."""
    budget = cfg.budgets.review_budget_minutes
    minutes = lambda c: cfg.decision.lawyer_minutes.get(str(c.action), 0) or 1  # noqa: E731
    chosen, used = set(), 0.0
    for c in sorted((c for c in cases if c.expected_value > 0), key=lambda c: -c.expected_value / minutes(c)):
        if used + minutes(c) <= budget:
            chosen.add(c.id)  # type: ignore[arg-type]
            used += minutes(c)
    return chosen


def decide_cases(case_ids: list[int], cfg: AppConfig, engine: Engine | None = None,
                 rules: RuleSet | None = None) -> dict[str, Any]:
    eng = engine or get_engine()
    rules = rules or get_rules()
    assets = load_brand_assets(cfg.paths.resolve(cfg.paths.brand_dir))
    summary: dict[str, Any] = {"decided": 0, "awaiting_approval": 0, "downgraded": 0, "by_action": {}}
    with session_scope(eng) as s:
        everything = [c for c in (s.get(Case, i) for i in case_ids) if c is not None]
        cases = [c for c in everything if c.status in DECIDABLE]
        for c in cases:
            listing = s.get(Listing, c.listing_id)
            assert listing is not None
            feats = s.exec(select(Features).where(Features.listing_id == c.listing_id)).first()
            d = decide(c, listing, feats, rules, cfg)
            c.action, c.expected_value = d.action, d.expected_value
            if d.downgrade_reason:
                summary["downgraded"] += 1
                if "Downgraded to escalate" not in c.reasoning:
                    c.reasoning += f" Downgraded to escalate: {d.downgrade_reason}."
            c.status = CaseStatus.awaiting_approval if d.action == Action.platform_report else CaseStatus.classified
            if c.status == CaseStatus.awaiting_approval:
                try:
                    c.notice_text = notice_text(build_plan(c, listing, assets, feats), assets)
                except NotAReportableCase as e:
                    c.status, c.action = CaseStatus.classified, Action.escalate
                    c.reasoning += f" Not reportable via portal ({e}); escalated."
            summary["decided"] += 1
            summary["by_action"][str(c.action)] = summary["by_action"].get(str(c.action), 0) + 1
        # the review queue holds every case waiting for a lawyer, including ones already drafted
        queue = sorted((c for c in everything if c.status in (CaseStatus.awaiting_approval, CaseStatus.drafted)),
                       key=lambda c: -c.expected_value)
        in_budget = select_within_budget(queue, cfg)
        for rank, c in enumerate(queue, 1):
            c.review_rank, c.in_review_budget = rank, c.id in in_budget
            if c.status == CaseStatus.drafted:  # keep the notice preview in step with the current plan
                listing = s.get(Listing, c.listing_id)
                feats = s.exec(select(Features).where(Features.listing_id == c.listing_id)).first()
                c.notice_text = notice_text(build_plan(c, listing, assets, feats), assets)  # type: ignore[arg-type]
        summary["awaiting_approval"] = len(queue)
        summary["in_review_budget"] = len(in_budget)
    return summary


def cases_of_run(run_id: int | None, engine: Engine | None = None) -> list[int]:
    with Session(engine or get_engine()) as s:
        cases = list(s.exec(select(Case)))
    runs = sorted({c.run_id for c in cases if c.run_id is not None})
    if not runs:
        return []
    rid = run_id if run_id is not None else runs[-1]
    return [c.id for c in cases if c.run_id == rid]  # type: ignore[misc]


async def run_decide(run_id: int | None, cfg: AppConfig | None = None, engine: Engine | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    ids = cases_of_run(run_id, engine)
    if ids:  # decide with the brand the cases' run was started with
        with Session(engine or get_engine()) as s:
            first = s.get(Case, ids[0])
        cfg = config_for_run(cfg, first.run_id if first else None, engine)
    async with run_span(None, {"stage": "decide", "cases_from_run": run_id}, engine=engine) as run:
        async with task_span(run.id, "decide", None, "decide", payload={"cases": len(ids)}) as t:
            summary = await asyncio.to_thread(decide_cases, ids, cfg, engine)
            t.payload = {**t.payload, **summary}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest", help="run whose cases to decide (classification run id)")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run_decide(None if args.run == "latest" else int(args.run))), indent=2))


if __name__ == "__main__":
    main()
