"""A shopper reports a listing to the brand's legal team.

The report copies the listing, its features and the classification from the shopper database into the
enforcement database as a case for the lawyers' review queue (``ui/staff.py``). Nothing is filed with a
marketplace until a named lawyer approves it there: the shopper is not the rights owner and cannot sign the
portal declarations (SPEC §1). After approval the submit loop files the factual notice as usual.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.brand import load_brand_assets, upsert_brand
from bh.config import get_config
from bh.db import Action, Case, CaseStatus, Features, Listing, Product, get_engine, session_scope
from bh.decide import decide_cases
from bh.notice import NotAReportableCase, build_plan, notice_text
from ui.data import CaseBundle

OPEN = (CaseStatus.classified, CaseStatus.awaiting_approval, CaseStatus.drafted, CaseStatus.approved,
        CaseStatus.waiting_human, CaseStatus.submitted, CaseStatus.rejected)
_LISTING_FIELDS = ("platform", "external_id", "title", "description", "price", "currency", "seller_name",
                   "seller_location", "seller_age_days", "rating", "quantity", "image_urls", "local_image_paths",
                   "screenshot_path", "discovered_by_query", "extraction_method")
_SKIP_FEATURES = {"id", "listing_id", "matched_product_id", "computed_at"}


@dataclass(frozen=True)
class ReportStatus:
    case_id: int
    status: str

    @property
    def message(self) -> str:
        return {
            "awaiting_approval": "Reported. The brand's legal team will review it.",
            "drafted": "Reported. The brand's legal team will review it.",
            "classified": "Reported. The brand's legal team will look into it.",
            "approved": "The brand's lawyer approved a takedown request.",
            "waiting_human": "The brand's legal team is working on this.",
            "submitted": "The brand has asked the marketplace to take this listing down.",
            "rejected": "The brand's legal team reviewed this and decided not to act.",
        }.get(self.status, "Reported.")


def status_for(url: str, main: Engine | None = None) -> ReportStatus | None:
    """Where the brand's legal team has got to with a listing a shopper reported."""
    with Session(main or get_engine()) as s:
        listing = s.exec(select(Listing).where(Listing.url == url)).first()
        if listing is None:
            return None
        cases = [c for c in s.exec(select(Case).where(Case.listing_id == listing.id)) if c.shopper_reports]
    if not cases:
        return None
    c = max(cases, key=lambda c: c.id or 0)
    return ReportStatus(c.id, str(c.status))  # type: ignore[arg-type]


def report(b: CaseBundle, main: Engine | None = None) -> ReportStatus:
    """File a shopper's report. Reporting the same listing again only adds to its report count."""
    eng = main or get_engine()
    cfg = get_config()
    assets = load_brand_assets(cfg.paths.resolve(cfg.paths.brand_dir))
    upsert_brand(assets, eng)
    src_case, src_listing, src_feats = b.case, b.listing, b.features
    with session_scope(eng) as s:
        listing = s.exec(select(Listing).where(Listing.url == src_listing.url)).first()
        if listing is None:
            listing = Listing(url=src_listing.url, platform=src_listing.platform)
            s.add(listing)
        for f in _LISTING_FIELDS:
            setattr(listing, f, getattr(src_listing, f))
        s.flush()
        lid = listing.id
        existing = [c for c in s.exec(select(Case).where(Case.listing_id == lid)) if c.status in OPEN]
        reported = [c for c in existing if c.shopper_reports]
        if reported:  # already with the legal team: count the extra report
            c = max(reported, key=lambda c: c.id or 0)
            c.shopper_reports = (c.shopper_reports or 0) + 1
            return ReportStatus(c.id, str(c.status))  # type: ignore[arg-type]
        feats = s.exec(select(Features).where(Features.listing_id == lid)).first()
        if feats is None:
            feats = Features(listing_id=lid)  # type: ignore[arg-type]
            s.add(feats)
        if src_feats is not None:
            for k, v in src_feats.model_dump().items():
                if k not in _SKIP_FEATURES:
                    setattr(feats, k, v)
            if b.product is not None:
                prod = s.exec(select(Product).where(Product.name == b.product.name)).first()
                feats.matched_product_id = prod.id if prod else None
        case = Case(listing_id=lid, run_id=None, category=src_case.category, confidence=src_case.confidence,  # type: ignore[arg-type]
                    rule_id=src_case.rule_id, legal_basis=list(src_case.legal_basis),
                    legal_elements=dict(src_case.legal_elements), action=src_case.action,
                    secondary_actions=list(src_case.secondary_actions), status=CaseStatus.classified,
                    reasoning=(src_case.reasoning + " Reported by a shopper.").strip(), shopper_reports=1)
        s.add(case)
        s.flush()
        cid = case.id
    decide_cases([cid], cfg, eng)  # type: ignore[list-item]
    with session_scope(eng) as s:
        c = s.get(Case, cid)
        assert c is not None
        if c.status != CaseStatus.awaiting_approval:
            # The confidence threshold would normally send this to escalation; a shopper asked for it to be
            # looked at, so put it in front of a lawyer whenever there is a right the portal can act on.
            listing = s.get(Listing, c.listing_id)
            feats = s.exec(select(Features).where(Features.listing_id == c.listing_id)).first()
            try:
                c.notice_text = notice_text(build_plan(c, listing, assets, feats), assets)  # type: ignore[arg-type]
                c.status, c.action, c.in_review_budget = CaseStatus.awaiting_approval, Action.platform_report, True
                c.reasoning += " Sent for review because a shopper reported it; check the evidence carefully."
            except NotAReportableCase as e:
                c.reasoning += f" Not reportable through the platform portal ({e})."
        return ReportStatus(cid, str(c.status))  # type: ignore[arg-type]
