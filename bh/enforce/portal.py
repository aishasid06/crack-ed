"""Rights-holder report portal automation (our own mock portal only — SPEC §1).

Drafting fills steps 1-4 (owner, right, listing + statute/section, evidence) and stops: declarations are
never touched before approval. After approval (bh.enforce.submit) the declarations are signed in the
approver's name, the review page is read back for verification, and only then may Submit be clicked.

Credentials are typed by code (BrowserManager.login); challenge pages raise HumanRequired.

    python -m bh.enforce.portal [--run latest|<id>]     # draft reports for awaiting_approval cases
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from playwright.async_api import Page
from PIL import Image
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.brand import BrandAssets, assets_for_run
from bh.browser.session import BrowserManager, detect_challenge
from bh.config import AppConfig, get_config
from bh.db import Case, CaseStatus, Features, Listing, get_engine, session_scope
from bh.enforce.guard import check_site, check_url
from bh.errors import BountyHunterError, HumanRequired
from bh.notice import ReportPlan, build_plan
from bh.trace import run_span, task_span

log = logging.getLogger(__name__)
SITE = "portal"
_DRAFT = re.compile(r"/report/(D[0-9a-f]+)/")

_REVIEW_JS = """
() => {
  const dl = sel => { const o = {}; document.querySelectorAll(sel + ' dt').forEach(dt => {
      o[dt.textContent.trim()] = (dt.nextElementSibling?.textContent || '').trim(); }); return o; };
  return {
    heading: document.querySelector('#review-heading')?.textContent?.trim() || null,
    owner: dl('#summary-owner dl'),
    right: dl('#summary-right dl'),
    rows: Array.from(document.querySelectorAll('#summary-listings tbody tr'))
            .map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent.replace(/\\s+/g, ' ').trim())),
    evidence: document.querySelectorAll('#summary-evidence li').length,
    declarations: (document.querySelector('#summary-declarations')?.textContent || '').replace(/\\s+/g, ' ').trim(),
    submit_button: !!document.querySelector('#final-submit'),
  };
}
"""


class PortalError(BountyHunterError):
    """The portal rejected a step (validation errors) or was in an unexpected state."""


class PortalClient:
    def __init__(self, browser: BrowserManager, cfg: AppConfig) -> None:
        check_site(cfg, SITE)
        self.b, self.cfg = browser, cfg

    def url(self, path: str) -> str:
        return self.b.site_url(SITE, path)

    async def _next(self, page: Page, step: str) -> None:
        """Submit the current step form, then check for challenges and validation errors."""
        check_url(self.cfg, page.url)  # never submit a form on a page that is off the allowlist
        async with self.b.action(SITE, f"submit:{step}"), page.expect_navigation(wait_until="domcontentloaded"):
            await page.click("#step-next")
        await detect_challenge(page, self.cfg.browser.challenge)
        if await page.locator(".pt-error").count():
            errors = [(await e.text_content() or "").strip() for e in await page.locator(".pt-field-error").all()]
            raise PortalError(f"portal rejected {step}: {'; '.join(errors) or 'validation error'}")

    def _evidence_files(self, plan: ReportPlan) -> list[Path]:
        out = []
        d = self.cfg.paths.resolve(self.cfg.paths.data_dir) / "evidence" / f"case_{plan.case_id}"
        d.mkdir(parents=True, exist_ok=True)
        limit = self.cfg.enforce.max_evidence_mb * 1024 * 1024
        for i, src in enumerate(plan.evidence, 1):
            if src.stat().st_size <= limit:
                dst = d / f"evidence_{i}{src.suffix.lower()}"
                shutil.copyfile(src, dst)
            else:  # re-encode oversized screenshots so the portal accepts them
                dst = d / f"evidence_{i}.jpg"
                with Image.open(src) as im:
                    im.convert("RGB").save(dst, "JPEG", quality=80, optimize=True)
            out.append(dst)
        return out

    async def create_draft(self, page: Page, plan: ReportPlan) -> str:
        """Fill steps 1-4 and stop at step 5 (declarations untouched). Returns the portal draft id."""
        check_site(self.cfg, SITE)
        await self.b.login(SITE, page)
        if not page.url.rstrip("/").endswith("/portal"):
            await self.b.goto(SITE, page, self.url("/"))
        async with self.b.action(SITE, "submit:new_report"), page.expect_navigation(wait_until="domcontentloaded"):
            await page.click("#new-report")
        await detect_challenge(page, self.cfg.browser.challenge)
        m = _DRAFT.search(page.url)
        if not m:
            raise PortalError(f"could not start a report (at {page.url})")
        draft_id = m.group(1)

        await page.fill("#owner_name", plan.owner_name)
        await page.fill("#owner_email", plan.owner_email)
        await page.fill("#owner_address", plan.owner_address)
        await page.check("input[name=relationship][value=owner]")
        await self._next(page, "step1_owner")

        await page.check(f"input[name=right_type][value={plan.right_type}]")
        if plan.right_type in ("trade_mark", "both"):
            await page.fill("#registration_number", plan.registration_number or "")
            await page.fill("#classes", ", ".join(map(str, plan.classes)))
            await page.select_option("#territory", plan.territory or "UK")
        if plan.right_type in ("copyright", "both"):
            await page.fill("#work_description", plan.work_description or "")
            await page.fill("#first_publication_date", plan.first_publication_date or "")
        await self._next(page, "step2_right")

        rows = page.locator(".pt-url-row")
        for i, it in enumerate(plan.items):
            if i >= await rows.count():
                async with self.b.action(SITE, "click:add_url"):
                    await page.click("#add-url")
            row = rows.nth(i)
            await row.locator(".pt-url").fill(it.url)
            await row.locator(".pt-statute").select_option(it.statute)
            await row.locator(".pt-section").fill(it.section)
            await row.locator(".pt-details").fill(it.details)
        await self._next(page, "step3_listings")

        files = await asyncio.to_thread(self._evidence_files, plan)
        await page.set_input_files("#evidence", [str(f) for f in files])
        await self._next(page, "step4_evidence")
        if not page.url.endswith(f"/report/{draft_id}/step/5"):
            raise PortalError(f"draft {draft_id} did not reach the declarations step (at {page.url})")
        return draft_id

    async def sign_and_review(self, page: Page, draft_id: str, signatory: str) -> str | None:
        """Sign the declarations as ``signatory`` and open the review page.
        Returns a submission reference instead if the draft was already submitted."""
        check_site(self.cfg, SITE)
        await self.b.login(SITE, page)
        await self.b.goto(SITE, page, self.url(f"/report/{draft_id}/step/5"))
        if "/submissions/" in page.url:
            return await self._ref_from_page(page)
        if not page.url.endswith(f"/report/{draft_id}/step/5"):
            raise PortalError(f"draft {draft_id} is incomplete (portal sent us to {page.url})")
        for box in await page.locator("#step-form input[type=checkbox]").all():
            await box.check()
        await page.fill("#signature", signatory)
        await self._next(page, "step5_declarations")
        if not page.url.endswith(f"/report/{draft_id}/review"):
            raise PortalError(f"expected the review page, at {page.url}")
        return None

    async def read_review(self, page: Page) -> dict[str, Any]:
        return await page.evaluate(_REVIEW_JS)

    async def submit(self, page: Page) -> str:
        check_site(self.cfg, SITE)
        check_url(self.cfg, page.url)
        async with self.b.action(SITE, "click:final_submit"), page.expect_navigation(wait_until="domcontentloaded"):
            await page.click("#final-submit")
        await detect_challenge(page, self.cfg.browser.challenge)
        return await self._ref_from_page(page)

    async def _ref_from_page(self, page: Page) -> str:
        m = re.search(r"/submissions/(IPR-[0-9A-F]+)", page.url)
        if not m:
            raise PortalError(f"no submission reference at {page.url}")
        return m.group(1)


# ---------------------------------------------------------------- drafting stage


def load_case(engine: Engine, case_id: int) -> tuple[Case, Listing, Features | None]:
    with Session(engine) as s:
        case = s.get(Case, case_id)
        listing = s.get(Listing, case.listing_id) if case else None
        feats = s.exec(select(Features).where(Features.listing_id == case.listing_id)).first() if case else None
    if case is None or listing is None:
        raise KeyError(f"case {case_id} not found")
    return case, listing, feats


def set_case(engine: Engine, case_id: int, **fields: Any) -> None:
    with session_scope(engine) as s:
        case = s.get(Case, case_id)
        assert case is not None
        for k, v in fields.items():
            setattr(case, k, v)


class BrandsByRun:
    """Brand assets for each case, taken from the run that created the case (its config snapshot), so a
    loop serving several brands always drafts and signs with the right owner and registration."""

    def __init__(self, cfg: AppConfig, engine: Engine) -> None:
        self.cfg, self.engine = cfg, engine
        self._cache: dict[int | None, BrandAssets] = {}

    def for_case(self, case_id: int) -> BrandAssets:
        with Session(self.engine) as s:
            case = s.get(Case, case_id)
        rid = case.run_id if case else None
        if rid not in self._cache:
            self._cache[rid] = assets_for_run(self.cfg, rid, self.engine)
        return self._cache[rid]


def draftable(engine: Engine, cfg: AppConfig, case_ids: list[int] | None = None) -> list[int]:
    with Session(engine) as s:
        q = select(Case).where(Case.status == CaseStatus.awaiting_approval, Case.portal_draft_id.is_(None))  # type: ignore[union-attr]
        cases = [c for c in s.exec(q) if case_ids is None or c.id in case_ids]
    if cfg.enforce.draft_only_in_budget:
        cases = [c for c in cases if c.in_review_budget]
    return [c.id for c in sorted(cases, key=lambda c: (c.review_rank or 10**6))]  # type: ignore[misc]


async def draft_one(run_id: int, case_id: int, client: PortalClient, assets: BrandAssets, engine: Engine) -> str:
    case, listing, feats = load_case(engine, case_id)
    plan = build_plan(case, listing, assets, feats)
    async with task_span(run_id, "draft", SITE, f"draft:case:{case_id}", payload={"case_id": case_id,
                                                                                 "stage": "draft"}) as t:
        if t.already_done:
            return case.portal_draft_id or ""
        try:
            async with client.b.page(SITE) as page:
                draft_id = await client.create_draft(page, plan)
        except HumanRequired:
            await asyncio.to_thread(set_case, engine, case_id, status=CaseStatus.waiting_human)
            raise
        t.payload = {**t.payload, "draft_id": draft_id}
    await asyncio.to_thread(set_case, engine, case_id, portal_draft_id=draft_id, status=CaseStatus.drafted)
    return draft_id


async def draft_cases(run_id: int, cfg: AppConfig, browser: BrowserManager, engine: Engine | None = None,
                      case_ids: list[int] | None = None) -> dict[str, Any]:
    """Draft portal reports (steps 1-4) for awaiting_approval cases. Stops at the first challenge page:
    the portal challenge is per-IP, so further attempts would be challenged too until a human verifies."""
    check_site(cfg, SITE)  # before any case is loaded or any page opened
    eng = engine or get_engine()
    client = PortalClient(browser, cfg)
    out: dict[str, Any] = {"drafted": [], "failed": [], "waiting_human": None}
    brands = BrandsByRun(cfg, eng)
    for cid in draftable(eng, cfg, case_ids):
        try:
            out["drafted"].append((cid, await draft_one(run_id, cid, client, brands.for_case(cid), eng)))
        except HumanRequired as e:
            out["waiting_human"] = {"case_id": cid, "url": e.url}
            break
        except Exception as e:  # recorded on the task; keep going with the next case
            log.warning("drafting case %s failed: %s", cid, e)
            out["failed"].append(cid)
    return out


async def run_drafting(cfg: AppConfig | None = None, engine: Engine | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    check_site(cfg, SITE)  # refuse before launching a browser
    async with run_span(None, {"stage": "draft"}, engine=engine) as run, BrowserManager(cfg) as b:
        return await draft_cases(run.id, cfg, b, engine)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print(json.dumps(asyncio.run(run_drafting()), indent=2, default=str))


if __name__ == "__main__":
    main()
