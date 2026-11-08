"""File a shopper's report through a marketplace's own "Report this listing" form.

The agent opens the listing, follows its "Report this listing" link, fills in the form with what the shopper
confirmed (reason, description, their name and email, how they came across it), attaches the evidence
screenshot, ticks the shopper's declaration and submits. The shopper authorised this report in the UI; the
agent never adds a rights-owner declaration (that is the lawyers' portal, bh/enforce/portal.py).

Progress is written to the ShopperReport row so the shopper UI can show it live. Challenge pages pause the
report (waiting_human); the agent never solves them.

    python -m bh.enforce.listing_report --report-id N
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Any

from bh.config import AppConfig, get_config
from bh.db import ShopperReport, get_engine, session_scope, utcnow
from bh.errors import BountyHunterError, HumanRequired

log = logging.getLogger(__name__)
_REF = re.compile(r"/reports/(LR-[0-9A-F]+)")


class ReportRejected(BountyHunterError):
    """The marketplace's form refused the report (validation errors)."""


def site_for(cfg: AppConfig, url: str) -> str:
    for name, site in cfg.sites.items():
        if site.adapter and url.startswith(site.base_url.rstrip("/") + "/"):
            return name
    raise ValueError(f"{url} is not a listing on a configured marketplace")


def _update(report_id: int, **fields: Any) -> None:
    with session_scope(get_engine()) as s:
        r = s.get(ShopperReport, report_id)
        assert r is not None
        for k, v in fields.items():
            setattr(r, k, v)


async def file_report(browser: Any, site: str, r: ShopperReport, progress: Any = None) -> tuple[str, str]:
    """Fill and submit the marketplace's report form for ``r``. Returns (reference, report page URL)."""
    from bh.browser.session import detect_challenge
    from bh.enforce.guard import check_site, check_url

    check_site(browser.cfg, site)  # submitting a form is enforcement: allowlisted, non-read_only sites only
    say = progress or (lambda text: None)
    name = browser.cfg.site(site).label(site)
    adapter = browser.adapter(site)
    async with browser.page(site) as page:
        say(f"Opening the listing on {name}")
        await browser.goto(site, page, r.listing_url)
        if adapter is not None:
            await adapter.prepare(page)  # e.g. dismiss a cookie banner
        say(f"Opening {name}'s “Report this listing” form")
        async with browser.action(site, "click:report_listing"), page.expect_navigation(wait_until="domcontentloaded"):
            await page.click("#report-listing")
        await detect_challenge(page, browser.cfg.browser.challenge)
        say("Filling in the form")
        await page.select_option("#lr-reason", r.reason)
        await page.fill("#lr-details", r.details)
        await page.check(f"#lr-relationship-{r.relationship}")
        await page.fill("#lr-name", r.reporter_name)
        await page.fill("#lr-email", r.reporter_email)
        if r.evidence_path and Path(r.evidence_path).exists():
            say("Attaching the screenshot as evidence")
            await page.set_input_files("#lr-evidence", r.evidence_path)
        await page.check("#lr-declaration")
        say("Submitting the report")
        check_url(browser.cfg, page.url)  # never submit a form on a page that is off the allowlist
        async with browser.action(site, "submit:listing_report"), \
                page.expect_navigation(wait_until="domcontentloaded"):
            await page.click("#lr-submit")
        await detect_challenge(page, browser.cfg.browser.challenge)
        if await page.locator(".lr-err").count():
            errors = [(await e.text_content() or "").strip() for e in await page.locator(".lr-err").all()]
            raise ReportRejected(f"{name} rejected the report: {'; '.join(errors)}")
        m = _REF.search(page.url)
        if not m:
            raise ReportRejected(f"no report reference after submitting (at {page.url})")
        return m.group(1), page.url


async def run_report(report_id: int, cfg: AppConfig | None = None, browser: Any | None = None) -> str:
    from bh.browser.session import BrowserManager
    from bh.trace import run_span, task_span

    cfg = cfg or get_config()
    with session_scope(get_engine()) as s:
        r = s.get(ShopperReport, report_id)
        if r is None:
            raise KeyError(f"shopper report {report_id} not found")
        s.expunge(r)
    if r.status == "submitted":
        return "submitted"
    site = site_for(cfg, r.listing_url)
    _update(report_id, status="filing", progress="Starting a browser", error=None, waiting_url=None)

    def say(text: str) -> None:
        _update(report_id, progress=text)

    async with run_span(None, {"stage": "listing_report", "report_id": report_id}) as run:
        try:
            async with task_span(run.id, "listing_report", site, f"listing_report:{report_id}",
                                 payload={"report_id": report_id, "url": r.listing_url}) as t:
                if browser is not None:
                    ref, url = await file_report(browser, site, r, say)
                else:
                    async with BrowserManager(cfg) as b:
                        ref, url = await file_report(b, site, r, say)
                t.payload = {**t.payload, "ref": ref}
        except HumanRequired as e:
            _update(report_id, status="waiting_human", waiting_url=e.url,
                    progress=f"{cfg.site(site).label(site)} asked for a security check")
            return "waiting_human"
        except Exception as e:
            _update(report_id, status="failed", error=str(e)[:500], progress=None)
            raise
    _update(report_id, status="submitted", report_ref=ref, report_url=url, submitted_at=utcnow(),
            progress=f"Report {ref} received by {cfg.site(site).label(site)}")
    return "submitted"


def main() -> int:
    from bh.shopper import use_shopper_db

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-id", type=int, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    use_shopper_db(get_config())
    try:
        outcome = asyncio.run(run_report(args.report_id))
        print(f"[bh] report {args.report_id}: {outcome}", flush=True)
        code = 0
    except Exception:
        log.exception("report %s failed", args.report_id)
        code = 1
    print(f"[bh] exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
