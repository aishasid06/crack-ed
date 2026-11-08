"""Report plan and factual notice text for a case.

``ReportPlan`` is the single source of truth for what goes into the rights-holder portal: the drafting
step fills the form from it and the submit step verifies the portal's review page against it.

Notices are factual: URL, right relied on, registration / work, statute and section, evidence. They
never threaten proceedings (Trade Marks Act 1994 ss.21-21F) — enforced by ``assert_factual``.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from bh.brand import BrandAssets
from bh.db import Case, Features, Listing

TRADE_MARK_STATUTE = "Trade Marks Act 1994"
COPYRIGHT_STATUTE = "Copyright, Designs and Patents Act 1988"
RIGHT_TYPE_LABELS = {"trade_mark": "Trade mark", "copyright": "Copyright", "both": "Trade mark and copyright"}

# Wording that could amount to a threat of proceedings; must never appear in a notice.
THREAT_PATTERNS = [r"\blegal action\b", r"\bsue\b", r"\blawsuit\b", r"\bproceedings\b", r"\bcourt\b",
                   r"\bdamages\b", r"\binjunction\b", r"\bsolicitors? will\b", r"\bwe will take\b",
                   r"\bliable\b", r"\bprosecut\w*", r"\bcease and desist\b"]


class ReportItem(BaseModel):
    url: str
    statute: str
    section: str
    details: str


class ReportPlan(BaseModel):
    case_id: int
    owner_name: str
    owner_email: str
    owner_address: str
    right_type: str  # trade_mark | copyright
    registration_number: str | None = None
    classes: list[int] = Field(default_factory=list)
    territory: str | None = None
    work_description: str | None = None
    first_publication_date: str | None = None
    items: list[ReportItem]
    evidence: list[Path]

    @property
    def right_type_label(self) -> str:
        return RIGHT_TYPE_LABELS[self.right_type]


class NotAReportableCase(ValueError):
    pass


def _details(case: Case, listing: Listing, assets: BrandAssets, features: Features | None) -> str:
    """Plain facts for the platform. The classifier's analysis (price ratios, seller clusters, routes)
    stays in the case file for the lawyer; it is not asserted in the notice."""
    marks = ", ".join(sorted({t.mark_text for t in assets.trademarks}))
    classes = ", ".join(str(c) for c in sorted({c for t in assets.trademarks for c in t.classes}))
    price = f" at £{listing.price:,.2f}" if listing.price is not None else ""
    facts = [f"The listing '{listing.title}' is offered by seller '{listing.seller_name}'{price}."]
    if case.rule_id == "copied_photos":
        facts.append("The listing displays photographs reproduced from the rights owner's product photography.")
    elif features is not None and features.misspelt_brand and not features.brand_in_text:
        facts.append(f"The listing uses a sign similar to the registered mark {marks} for goods in class {classes}.")
    else:
        facts.append(f"The listing uses the sign {marks}, identical to the registered mark, for goods in class "
                     f"{classes}.")
    if features is not None and features.replica_kw:
        facts.append(f"The listing describes the goods as: {', '.join(repr(k) for k in features.replica_kw)}.")
    return " ".join(facts)[:1400]


def build_plan(case: Case, listing: Listing, assets: BrandAssets, features: Features | None = None) -> ReportPlan:
    """Map an approved/awaiting case onto the portal's report fields (rule legal basis → right type)."""
    if not case.legal_basis:
        raise NotAReportableCase(f"case {case.id}: rule {case.rule_id} has no legal basis to report")
    if assets.meta.owner is None:
        raise NotAReportableCase("brand assets have no rights-owner details (brand.json 'owner')")
    basis = case.legal_basis[0]
    statute, section = basis["statute"], basis["section"]
    owner = assets.meta.owner
    plan = dict(case_id=case.id, owner_name=owner.name, owner_email=owner.email, owner_address=owner.address,
                items=[ReportItem(url=listing.url, statute=statute, section=section,
                                  details=_details(case, listing, assets, features))],
                evidence=[Path(listing.screenshot_path)] if listing.screenshot_path else [])
    if statute == TRADE_MARK_STATUTE:
        if not assets.trademarks:
            raise NotAReportableCase(f"case {case.id}: no trade mark registration for this brand")
        word = next((t for t in assets.trademarks if t.mark_type == "word"), assets.trademarks[0])
        plan.update(right_type="trade_mark", registration_number=word.registration_number,
                    classes=list(word.classes), territory=word.territory)
    elif statute == COPYRIGHT_STATUTE:
        if assets.meta.copyright is None:
            raise NotAReportableCase("brand assets have no copyright work details (brand.json 'copyright')")
        plan.update(right_type="copyright", work_description=assets.meta.copyright.work_description,
                    first_publication_date=assets.meta.copyright.first_published)
    else:
        raise NotAReportableCase(f"case {case.id}: {statute} is not reportable through the platform portal")
    if not plan["evidence"]:
        raise NotAReportableCase(f"case {case.id}: no screenshot captured as evidence")
    return ReportPlan(**plan)


def notice_text(plan: ReportPlan, assets: BrandAssets) -> str:
    """Plain, factual notice (E-Commerce Regulations 2002 reg.19: exact URL, right relied on, owner)."""
    it = plan.items[0]
    if plan.right_type == "trade_mark":
        word = next(t for t in assets.trademarks if t.registration_number == plan.registration_number)
        right = (f"UK registered trade mark {plan.registration_number} ({word.mark_text}, {word.mark_type} mark), "
                 f"classes {', '.join(map(str, plan.classes))}, territory {plan.territory}")
    else:
        right = f"copyright in {plan.work_description}, first published {plan.first_publication_date}"
    text = (
        f"Notice of infringing listing\n\n"
        f"Rights owner: {plan.owner_name}, {plan.owner_address} ({plan.owner_email}).\n"
        f"Right relied on: {right}.\n"
        f"Listing: {it.url}\n"
        f"Provision: {it.statute} {it.section}.\n"
        f"Facts: {it.details}\n"
        f"Evidence: screenshot of the listing captured on review ({len(plan.evidence)} file(s)).\n\n"
        f"We ask that the listing be reviewed and removed. This notice identifies the listing and the right "
        f"relied on so that the platform has the information it needs to act.")
    # Our wording must be factual. The quoted facts come from automated evidence and the third-party
    # listing title, which we report as-is rather than reject.
    assert_factual(text.replace(it.details, ""))
    return text


def assert_factual(text: str) -> None:
    for pat in THREAT_PATTERNS:
        if re.search(pat, text, re.I):
            raise ValueError(f"notice text contains threatening wording matching {pat!r}")
