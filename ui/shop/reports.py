"""Shopper reports to marketplaces: prepare the report a shopper confirms, start the agent that files it
through the marketplace's own report form, and read back where each report has got to."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from bh.config import get_config
from bh.db import ShopperReport, session_scope
from mocksites.listing_report_forms import DECLARATION, REASONS, RELATIONSHIPS  # noqa: F401 - re-exported
from ui import theme, workers
from ui.data import CaseBundle
from ui.shop import data
from ui.shop.verdicts import reason

REASON_FOR_CATEGORY = {
    "counterfeit_identical": "counterfeit",
    "counterfeit_similar": "imitation_brand",
    "copied_photos": "copied_photos",
}
ACTIVE = ("queued", "filing")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def site_label(site: str) -> str:
    cfg = get_config()
    return cfg.sites[site].label(site) if site in cfg.sites else site.capitalize()


def draft_details(b: CaseBundle, brand: str) -> str:
    """Plain facts for the report, which the shopper can edit before sending."""
    l, cfg = b.listing, get_config()
    lines = [reason(str(b.case.category), l, b.features, brand, cfg.thresholds.counterfeit_price_ratio)]
    if l.price is not None and b.product is not None:
        lines.append(f"It is listed at {theme.money(l.price)}; the official price of the {b.product.name} is "
                     f"{theme.money(b.product.rrp)}.")
    if l.seller_name:
        lines.append(f"Seller: {l.seller_name}.")
    return " ".join(lines)


@dataclass
class ReportForm:
    reason: str
    details: str
    reporter_name: str
    reporter_email: str
    relationship: str
    attach_screenshot: bool
    declaration: bool

    def errors(self) -> dict[str, str]:
        e = {}
        if self.reason not in REASONS:
            e["reason"] = "Choose what's wrong with the listing."
        if len(self.details.strip()) < 20:
            e["details"] = "Describe the problem in at least 20 characters."
        if not self.reporter_name.strip():
            e["reporter_name"] = "Enter your name. The marketplace needs it to act on the report."
        if not EMAIL.match(self.reporter_email.strip()):
            e["reporter_email"] = "Enter a valid email address so the marketplace can reply."
        if self.relationship not in RELATIONSHIPS:
            e["relationship"] = "Choose how you came across the listing."
        if not self.declaration:
            e["declaration"] = "Confirm the information is accurate to send the report."
        return e


def create(b: CaseBundle, form: ReportForm, search_id: str | None, eng: Engine | None = None) -> int:
    """Record the shopper's report and start the agent that files it."""
    shot = b.screenshot if form.attach_screenshot else None
    with session_scope(eng or data.get_engine()) as s:
        r = ShopperReport(search_id=search_id, listing_url=b.listing.url, listing_title=b.listing.title,
                          site=b.listing.platform, reason=form.reason, details=form.details.strip(),
                          reporter_name=form.reporter_name.strip(), reporter_email=form.reporter_email.strip(),
                          relationship=form.relationship, evidence_path=str(shot) if shot else None,
                          progress="Waiting for the agent to start")
        s.add(r)
        s.flush()
        rid = r.id
    workers.start(f"report-{rid}")
    return rid  # type: ignore[return-value]


def retry(report_id: int, eng: Engine | None = None) -> None:
    with session_scope(eng or data.get_engine()) as s:
        r = s.get(ShopperReport, report_id)
        if r is None or r.status == "submitted":
            return
        r.status, r.error, r.waiting_url, r.progress = "queued", None, None, "Trying again"
    workers.start(f"report-{report_id}")


def _settle(r: ShopperReport) -> ShopperReport:
    """A report whose agent process has gone away without finishing counts as failed."""
    if r.status in ACTIVE:
        w = workers.status(f"report-{r.id}")
        if not w.alive and w.started_at is not None:
            r.status, r.error = "failed", r.error or "The agent stopped before it finished."
    return r


def all_reports(eng: Engine | None = None) -> list[ShopperReport]:
    with Session(eng or data.get_engine()) as s:
        rows = list(s.exec(select(ShopperReport).order_by(col(ShopperReport.id).desc())))
    return [_settle(r) for r in rows]


def for_url(url: str, eng: Engine | None = None) -> ShopperReport | None:
    with Session(eng or data.get_engine()) as s:
        r = s.exec(select(ShopperReport).where(ShopperReport.listing_url == url)
                   .order_by(col(ShopperReport.id).desc())).first()
    return _settle(r) if r else None
