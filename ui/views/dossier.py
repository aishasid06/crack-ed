"""The case dossier: everything a lawyer needs to decide one case, rendered the same way wherever a case
is opened (review queue, cases table, human-required page)."""

from __future__ import annotations

import streamlit as st

from bh.config import get_config
from bh.decide import DISAGREE_LABEL
from bh.money import money
from bh.rules import get_rules
from mocksites.portal_forms import DECLARATIONS
from ui import theme
from ui.data import CaseBundle


def citation(b: CaseBundle) -> tuple[str, str]:
    if b.case.legal_basis:
        lb = b.case.legal_basis[0]
        return f"{lb.get('statute', '')}, {lb.get('section', '')}", str(lb.get("summary", ""))
    return "No legal basis relied on", f"Rule “{b.case.rule_id}” recommends no enforcement action."


def overlay_text(b: CaseBundle) -> str | None:
    if "refer_trading_standards" not in b.case.secondary_actions:
        return None
    rules = get_rules()
    ov = next((o for o in rules.overlays if o.action == "refer_trading_standards"), None)
    basis = f" ({ov.legal_basis.citation()})" if ov else ""
    return (f"Also recommended: refer to Trading Standards{basis}. The seller shows signs of dealing in "
            f"counterfeit goods at scale.")


def header(b: CaseBundle) -> None:
    c, l = b.case, b.listing
    facts = [("Platform", l.platform.capitalize()), ("Seller", l.seller_name or "unknown"),
             ("Price", money(l.price, l.currency) if l.price is not None else "—"),
             ("Confidence", f"{c.confidence:.0%}"),
             ("Expected value", theme.money(c.expected_value)),
             ("Action", theme.ACTION_LABEL.get(str(c.action), str(c.action)))]
    if c.shopper_reports:
        facts.append(("Shopper reports", c.shopper_reports))
    st.markdown(
        f'<div class="bh-dossier-head"><div class="bh-case-no">Case {c.id} · '
        f'{theme.e(theme.CATEGORY_LABEL.get(str(c.category), str(c.category)))} &nbsp;{theme.pill(str(c.status))}'
        f'</div><div class="bh-case-title">{theme.e(l.title or "(untitled listing)")}</div>'
        f'<div class="bh-facts">{"".join(f"<span>{k} <b>{theme.e(v)}</b></span>" for k, v in facts)}</div></div>',
        unsafe_allow_html=True)


def _photo(path, empty_title: str, empty_body: str, caption: str) -> None:
    if path:
        st.image(str(path), width="stretch")
    else:
        theme.empty(empty_title, empty_body)
    st.markdown(f'<div class="bh-figure">{theme.e(caption)}</div>', unsafe_allow_html=True)


def evidence(b: CaseBundle) -> None:
    """Brand photo and listing photo side by side, with the captured listing facts beside them."""
    l, f = b.listing, b.features
    bits = []
    if f is not None:
        if f.clip_sim is not None:
            bits.append(f"visual similarity {f.clip_sim:.2f}")
        if f.hash_dist is not None:
            bits.append(f"pHash distance {f.hash_dist}")
        if f.own_photo_routes:
            bits.append("own photo via " + ", ".join(f.own_photo_routes))
    g, li, facts = st.columns([1, 1, 1.5], gap="medium")
    with g:
        _photo(b.gold_image, "No brand photo", "No product was matched to this listing.",
               f"Brand photo: {b.product.name if b.product else 'no match'}")
    with li:
        _photo(b.listing_image, "No listing photo", "The capture step stored no images for this listing.",
               "Listing photo" + (f": {', '.join(bits)}" if bits else ""))
    with facts:
        st.markdown(theme.kv_table([
            ("Listing", l.url), ("Seller location", l.seller_location),
            ("Seller age", f"{l.seller_age_days} days" if l.seller_age_days is not None else None),
            ("Rating", l.rating), ("Quantity", l.quantity),
            ("Price vs RRP", f"{f.price_ratio:.0%}" if f and f.price_ratio is not None else None),
            ("Found by query", l.discovered_by_query),
            ("Extracted by", extraction_label(l)),
        ]), unsafe_allow_html=True)


def extraction_label(l: object) -> str:
    """How the listing's details were read: adapter, learned recipe (no LLM), navigator or LLM fallback."""
    method = str(getattr(l, "extraction_method", ""))
    site = get_config().sites.get(getattr(l, "platform", ""))
    if method == "recipe":
        return "learned recipe (replayed, no LLM)"
    if method == "generic_llm":
        return "agentic navigator" if site is not None and site.navigator else "LLM fallback extractor"
    return "site adapter"


def flags(b: CaseBundle) -> None:
    """Banners that must be seen before anything else in the case file."""
    if b.demo_banner:
        st.info(b.demo_banner)
    if b.case.model_disagrees:
        st.warning(f"**{DISAGREE_LABEL.capitalize()}.** {b.case.model_disagreement or ''}")


def statute(b: CaseBundle) -> None:
    cite, summary = citation(b)
    try:
        order = list(get_rules().rule(b.case.rule_id).requires)
    except Exception:
        order = []
    st.markdown(theme.statute_strip(cite, summary, b.case.legal_elements, order, overlay_text(b)),
                unsafe_allow_html=True)


def reasoning(b: CaseBundle) -> None:
    st.markdown("**Reasoning**")
    st.write(b.case.reasoning or "—")
    if b.case.llm_used:
        st.caption("A strong-model vision check reviewed this borderline case.")
    if b.listing.description:
        with st.expander("Listing description"):
            st.write(b.listing.description)
    if b.screenshot:
        with st.expander("Screenshot (uploaded as evidence with the notice)"):
            st.image(str(b.screenshot), width="stretch")


def notice(b: CaseBundle) -> None:
    st.markdown("**Notice to the platform**")
    if b.case.notice_text:
        st.markdown(f'<div class="bh-letter">{theme.e(b.case.notice_text)}</div>', unsafe_allow_html=True)
    else:
        st.caption("No notice prepared — this case is not reported through the platform portal.")


def declarations(signatory: str | None) -> None:
    items = "".join(f"<li>{theme.e(t)}</li>" for t in DECLARATIONS.values())
    who = theme.e(signatory) if signatory else '<span style="color:var(--bh-muted)">enter your name in the sidebar</span>'
    st.markdown(f'<div class="bh-decl"><ol>{items}</ol><div class="bh-sign">Signed: {who}</div></div>',
                unsafe_allow_html=True)


def full(b: CaseBundle) -> None:
    flags(b)
    header(b)
    st.write("")
    evidence(b)
    statute(b)
    if b.case.review_diff:
        st.markdown("**Portal review page did not match the approved case**")
        st.markdown(theme.diff_table(b.case.review_diff), unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        notice(b)
    with right:
        reasoning(b)
