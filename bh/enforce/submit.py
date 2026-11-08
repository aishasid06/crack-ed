"""Submit-on-approval loop (SPEC §7.1).

For every case a human approved ("Approve and submit"):
  1. resume (or create) the portal draft, sign the declarations as Case.decided_by → declared_by/at;
  2. read the review page and verify it against the approved case: listing URL(s), right type,
     registration number (or copyright work), statute/section per URL, evidence count, signatory;
  3. full match → click Submit and store the reference. Any mismatch → do NOT submit: the case goes to
     waiting_human with the field-by-field diff in Case.review_diff.
With enforce.require_final_confirm, step 3 additionally waits for confirm_final() from the approver.
Rejected cases are never touched.

    python -m bh.enforce.submit [--once | --poll SECONDS]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.brand import BrandAssets
from bh.browser.session import BrowserManager
from bh.config import AppConfig, get_config
from bh.db import Case, CaseStatus, get_engine, utcnow
from bh.enforce.guard import check_site
from bh.enforce.portal import SITE, BrandsByRun, PortalClient, load_case, set_case
from bh.errors import HumanRequired, ReviewMismatch
from bh.notice import ReportPlan, build_plan
from bh.trace import run_span, task_span

log = logging.getLogger(__name__)


def compare_review(plan: ReportPlan, review: dict[str, Any], signatory: str) -> dict[str, dict[str, Any]]:
    """Field-by-field differences between what was approved and what the portal will submit
    (right, registration/work, URL, statute/section and infringement text per URL, evidence, owner, signatory)."""
    diff: dict[str, dict[str, Any]] = {}

    def check(name: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            diff[name] = {"expected": expected, "actual": actual}

    right = review.get("right", {})
    check("right_type", plan.right_type_label, right.get("Type"))
    if plan.right_type in ("trade_mark", "both"):
        check("registration_number", plan.registration_number, right.get("Registration"))
        check("classes", ", ".join(map(str, plan.classes)), right.get("Classes"))
        check("territory", plan.territory, right.get("Territory"))
    if plan.right_type in ("copyright", "both"):
        check("work_description", plan.work_description, right.get("Work"))
        check("first_publication_date", plan.first_publication_date, right.get("First published"))
    rows = review.get("rows", [])
    check("listing_urls", [it.url for it in plan.items], [r[0] for r in rows if r])
    check("provisions", [f"{it.statute} {it.section}" for it in plan.items], [r[1] for r in rows if len(r) > 1])
    check("infringement_details", [" ".join(it.details.split()) for it in plan.items],
          [r[2] for r in rows if len(r) > 2])
    check("evidence_count", len(plan.evidence), review.get("evidence"))
    owner = review.get("owner", {})
    check("owner", plan.owner_name, owner.get("Owner"))
    signed = review.get("declarations", "")
    check("signatory", signatory, signed.split("Signed:", 1)[1].strip() if "Signed:" in signed else None)
    if not review.get("submit_button"):
        diff["submit_button"] = {"expected": True, "actual": False}
    return diff


async def submit_one(run_id: int, case_id: int, client: PortalClient, assets: BrandAssets, cfg: AppConfig,
                     engine: Engine) -> str:
    """Returns the outcome: submitted | awaiting_final_confirm | waiting_human."""
    case, listing, feats = load_case(engine, case_id)
    if case.status != CaseStatus.approved or not case.decided_by:
        raise ValueError(f"case {case_id} is not approved by a named lawyer")
    plan = build_plan(case, listing, assets, feats)
    async with task_span(run_id, "submit", SITE, f"submit:case:{case_id}",
                         payload={"case_id": case_id, "stage": "submit"}) as t:
        if t.already_done:
            return "already_processed"
        async with client.b.page(SITE) as page:
            try:
                draft_id = case.portal_draft_id
                if not draft_id:
                    draft_id = await client.create_draft(page, plan)
                    await asyncio.to_thread(set_case, engine, case_id, portal_draft_id=draft_id)
                ref = await client.sign_and_review(page, draft_id, case.decided_by)
                if ref:  # submitted earlier (e.g. crash after Submit): just record it
                    await asyncio.to_thread(set_case, engine, case_id, status=CaseStatus.submitted,
                                            submission_ref=ref, submitted_at=case.submitted_at or utcnow())
                    t.payload = {**t.payload, "submission_ref": ref, "note": "already submitted"}
                    return "submitted"
                await asyncio.to_thread(set_case, engine, case_id, declared_by=case.decided_by, declared_at=utcnow())
                review = await client.read_review(page)
                diff = compare_review(plan, review, case.decided_by)
                if diff:
                    await asyncio.to_thread(set_case, engine, case_id, review_diff=diff,
                                            status=CaseStatus.waiting_human)
                    t.payload = {**t.payload, "review_diff": diff}
                    raise ReviewMismatch(f"review page does not match the approved case: {sorted(diff)}", url=page.url)
                await asyncio.to_thread(set_case, engine, case_id, review_diff={}, review_verified_at=utcnow())
                if cfg.enforce.require_final_confirm and not case.final_confirmed:
                    t.payload = {**t.payload, "verified": True, "note": "awaiting final confirmation"}
                    return "awaiting_final_confirm"
                ref = await client.submit(page)
            except HumanRequired:
                await asyncio.to_thread(set_case, engine, case_id, status=CaseStatus.waiting_human)
                raise
        await asyncio.to_thread(set_case, engine, case_id, status=CaseStatus.submitted, submission_ref=ref,
                                submitted_at=utcnow())
        t.payload = {**t.payload, "submission_ref": ref, "draft_id": draft_id}
        return "submitted"


def approved_cases(engine: Engine) -> list[int]:
    with Session(engine) as s:
        cases = list(s.exec(select(Case).where(Case.status == CaseStatus.approved)))
    return [c.id for c in sorted(cases, key=lambda c: (c.decided_at or utcnow()))]  # type: ignore[misc]


async def process_approved(run_id: int, cfg: AppConfig, browser: BrowserManager,
                           engine: Engine | None = None) -> dict[str, Any]:
    check_site(cfg, SITE)  # before any case is loaded or any page opened
    eng = engine or get_engine()
    client = PortalClient(browser, cfg)
    brands = BrandsByRun(cfg, eng)  # each case is signed with its own brand's data
    out: dict[str, Any] = {"submitted": [], "awaiting_final_confirm": [], "waiting_human": [], "failed": []}
    for cid in approved_cases(eng):
        try:
            result = await submit_one(run_id, cid, client, brands.for_case(cid), cfg, eng)
            out.setdefault(result, []).append(cid)
        except ReviewMismatch as e:
            out["waiting_human"].append({"case_id": cid, "url": e.url, "reason": str(e)})
        except HumanRequired as e:
            out["waiting_human"].append({"case_id": cid, "url": e.url, "reason": str(e)})
            break  # portal challenge (per-IP): later cases would be challenged too until a human verifies
        except Exception as e:
            log.warning("submitting case %s failed: %s", cid, e)
            out["failed"].append(cid)
    return out


async def run_submit_once(cfg: AppConfig | None = None, engine: Engine | None = None) -> dict[str, Any] | None:
    cfg = cfg or get_config()
    check_site(cfg, SITE)  # refuse before launching a browser
    if not approved_cases(engine or get_engine()):
        return None
    async with run_span(None, {"stage": "submit"}, engine=engine) as run, BrowserManager(cfg) as b:
        return await process_approved(run.id, cfg, b, engine)


async def poll(interval: float, cfg: AppConfig | None = None) -> None:
    while True:
        res = await run_submit_once(cfg)
        if res:
            log.info("submit cycle: %s", res)
        await asyncio.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poll", type=float, default=None, help="poll for approved cases every N seconds")
    args = ap.parse_args()
    if args.poll:
        asyncio.run(poll(args.poll))
    else:
        print(json.dumps(asyncio.run(run_submit_once()), indent=2, default=str))


if __name__ == "__main__":
    main()
