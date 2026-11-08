"""Bounty Hunter for shoppers: search for a product by name or photo, watch the agent search the
marketplaces, and see which listings look genuine and which to avoid.

    streamlit run ui/app.py          # shoppers
    streamlit run ui/staff.py        # the brand's lawyers (review queue, approvals, trace)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from bh.config import get_config  # noqa: E402
from bh.db import RunStatus  # noqa: E402
from ui import theme, workers  # noqa: E402
from ui.data import CaseBundle  # noqa: E402
from ui.shop import data, feed, replay, report, reports  # noqa: E402
from ui.shop.verdicts import CATEGORY_GROUP, GROUP_BY_KEY, GROUPS, LABEL, reason  # noqa: E402

CSS = f"""
<style>
.block-container {{ max-width: 920px; padding-top: 2.4rem; }}
.sh-mark {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.05rem; color: {theme.TEAL}; }}
.sh-hero h1 {{ font-size: 2.7rem !important; line-height: 1.08 !important; margin: .3rem 0 .6rem; max-width: 18ch; }}
.sh-lede {{ color: {theme.MUTED}; font-size: 1.08rem; line-height: 1.55; max-width: 60ch; margin-bottom: 1.6rem; }}
.sh-fine {{ color: {theme.MUTED}; font-size: .82rem; margin-top: 1.4rem; }}
.sh-query {{ display: flex; gap: 1rem; align-items: center; margin: .2rem 0 1.4rem; }}
.sh-query .q {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.7rem; color: {theme.INK}; }}
.sh-steps {{ list-style: none; padding: 0; margin: 0 0 1.4rem; display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .5rem; }}
.sh-steps li {{ border-top: 3px solid {theme.RULE}; padding: .55rem .1rem 0; color: {theme.MUTED}; font-size: .92rem; }}
.sh-steps li b {{ display: block; font-size: .8rem; font-weight: 600; margin-bottom: .1rem; }}
.sh-steps li.done {{ border-color: {theme.GREEN}; color: {theme.INK}; }}
.sh-steps li.active {{ border-color: {theme.TEAL}; color: {theme.INK}; }}
.sh-steps li.active b {{ color: {theme.TEAL}; }}
.sh-now {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .8rem; margin-bottom: 1.2rem; }}
.sh-site {{ background: {theme.SURFACE}; border: 1px solid {theme.RULE}; border-left: 3px solid {theme.TEAL};
  border-radius: 4px; padding: .8rem 1rem; }}
.sh-site .name {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; color: {theme.INK}; }}
.sh-site .doing {{ color: {theme.INK}; margin-top: .15rem; }}
.sh-site .url {{ color: {theme.MUTED}; font-size: .8rem; margin-top: .3rem; word-break: break-all; }}
.sh-live {{ display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; background: {theme.TEAL};
  margin-right: .45rem; animation: shpulse 1.4s ease-in-out infinite; }}
@keyframes shpulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .25; }} }}
@media (prefers-reduced-motion: reduce) {{ .sh-live {{ animation: none; }} }}
.sh-log {{ list-style: none; padding: 0; margin: 0; }}
.sh-log li {{ padding: .4rem 0; border-bottom: 1px solid {theme.RULE}; color: {theme.MUTED}; font-size: .93rem; }}
.sh-log li:first-child {{ color: {theme.INK}; }}
.sh-summary {{ font-size: 1.12rem; line-height: 1.55; margin: .2rem 0 1.6rem; max-width: 62ch; }}
.sh-group {{ display: flex; align-items: baseline; gap: .6rem; margin: 1.6rem 0 .2rem; }}
.sh-group .g {{ font-weight: 700; font-size: 1.1rem; }}
.sh-group h3 {{ margin: 0 !important; }}
.sh-group-blurb {{ color: {theme.MUTED}; margin-bottom: .7rem; font-size: .95rem; }}
[data-testid="stExpander"] summary p {{ font-size: 1.15rem; }}
[data-testid="stExpander"] {{ margin-top: .8rem; }}
.sh-badge {{ display: inline-block; font-size: .78rem; font-weight: 600; padding: .1rem .5rem;
  border-radius: 999px; border: 1px solid currentColor; }}
.sh-title {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.3rem; line-height: 1.2; margin: .35rem 0 .3rem; }}
.sh-title a {{ color: {theme.INK}; text-decoration: none; }}
.sh-title a:hover {{ text-decoration: underline; }}
.sh-price {{ font-size: 1.15rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.sh-price span {{ font-size: .88rem; font-weight: 400; color: {theme.MUTED}; margin-left: .4rem; }}
.sh-where {{ color: {theme.MUTED}; font-size: .88rem; margin: .1rem 0 .45rem; }}
.sh-why {{ line-height: 1.5; margin-bottom: .6rem; }}
.sh-reported {{ font-weight: 600; font-size: .92rem; }}
.sh-reported a {{ color: inherit; }}
.sh-dlg-listing {{ border-left: 3px solid {theme.RED}; padding: .4rem .8rem; margin: .4rem 0 .8rem;
  font-family: 'Newsreader', Georgia, serif; font-size: 1.1rem; }}
.sh-dlg-listing span {{ font-family: 'Public Sans', sans-serif; font-size: .85rem; color: {theme.MUTED}; }}
.sh-side-title {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; margin-bottom: .6rem; }}
.sh-side-empty {{ color: {theme.MUTED}; font-size: .9rem; }}
.sh-side-item {{ border-top: 1px solid {theme.RULE}; padding: .7rem 0 .5rem; }}
.sh-side-head {{ display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }}
.sh-pill {{ font-size: .74rem; font-weight: 600; padding: .05rem .5rem; border: 1px solid currentColor;
  border-radius: 999px; }}
.sh-side-site {{ color: {theme.MUTED}; font-size: .8rem; }}
.sh-side-name {{ font-size: .9rem; margin-top: .3rem; line-height: 1.35; }}
.sh-side-progress {{ color: {theme.MUTED}; font-size: .82rem; margin-top: .3rem; }}
.sh-side-link {{ font-size: .88rem; margin-top: .3rem; font-weight: 600; }}
</style>
"""


def _brand() -> str:
    return data.brand_name()


# ---------------------------------------------------------------- start


def _ensure_marketplaces() -> bool:
    if workers.mock_sites_up():
        return True
    workers.start("mock")
    for _ in range(40):
        time.sleep(0.5)
        if workers.mock_sites_up():
            return True
    return False


def home() -> None:
    st.markdown('<div class="sh-hero"><div class="sh-mark">Bounty Hunter</div>'
                '<h1>Check it\'s genuine before you buy</h1></div>'
                '<p class="sh-lede">Tell us what you\'re looking for, or upload a photo. We search online '
                'marketplaces, compare each listing with the brand\'s official products, sellers and prices, and '
                'show you which look genuine and which to avoid. Spot a fake, and our agent reports it to the '
                'marketplace for you.</p>', unsafe_allow_html=True)
    with st.form("search", border=False):
        text = st.text_input("What are you looking for?", placeholder="Brand and product name")
        photo = st.file_uploader("Add a photo (optional)", type=["jpg", "jpeg", "png"],
                                 help="A photo of the product, or a screenshot of a listing you've seen.")
        go = st.form_submit_button("Search marketplaces", type="primary")
    if go:
        if not text.strip() and photo is None:
            st.warning("Type a product name or add a photo to search.")
            return
        with st.spinner("Connecting to the marketplaces…"):
            ready = _ensure_marketplaces()
        if not ready:
            st.error("The marketplaces aren't reachable right now. Try again in a minute.")
            return
        req = data.new_search(text, photo.name if photo else None, photo.getvalue() if photo else None)
        # a search saved earlier (same words and photo) is played back instead of re-running the pipeline
        saved = (replay.find_source(req.text, req.photo, data.get_engine())
                 if get_config().shopper.replay_saved else None)
        workers.start(f"search-{req.id}", text=req.text, photo=req.photo, replay_of=saved)
        st.query_params["search"] = req.id
        st.rerun()


# ---------------------------------------------------------------- shared pieces


def query_header(req: data.SearchRequest) -> None:
    cols = st.columns([1, 6]) if req.photo and req.photo.exists() else [None, st.container()]
    if cols[0] is not None:
        cols[0].image(str(req.photo), width="stretch")
    label = req.text or "Your photo"
    cols[1].markdown(f'<div class="sh-query"><span class="q">{theme.e(label)}</span></div>', unsafe_allow_html=True)


def new_search_button(key: str) -> None:
    if st.button("New search", key=key, icon=":material/search:"):
        st.query_params.clear()
        st.rerun()


# ---------------------------------------------------------------- searching


def _steps_html(steps: list[tuple[str, str]]) -> str:
    words = {"done": "Done", "active": "In progress", "pending": "Up next"}
    items = "".join(f'<li class="{s}"><b>{i}. {words[s]}</b>{theme.e(label)}</li>'
                    for i, (label, s) in enumerate(steps, 1))
    return f'<ol class="sh-steps">{items}</ol>'


def _now_html(now: list[feed.SiteNow]) -> str:
    if not now:
        return ""
    cards = "".join(
        f'<div class="sh-site"><div class="name"><span class="sh-live" aria-hidden="true"></span>'
        f'{theme.e(feed.site_name(n.site))}</div><div class="doing">{theme.e(n.doing)}</div>'
        f'<div class="url">{theme.e(feed.short_url(n.url))}</div></div>' for n in now)
    return f'<div class="sh-now" aria-live="polite">{cards}</div>'


def searching(req: data.SearchRequest) -> None:
    query_header(req)

    @st.fragment(run_every=1.5)
    def live() -> None:
        w = workers.status(f"search-{req.id}")
        run = data.find_run(req.id)
        finished = run is not None and run.status != RunStatus.running
        if finished or not w.alive:
            st.rerun(scope="app")
        tasks = data.tasks(run.id) if run else []  # type: ignore[arg-type]
        f = feed.build(tasks, _brand(), finished=False)
        st.markdown(_steps_html(f.steps), unsafe_allow_html=True)
        if f.now:
            st.markdown("**Right now**")
            st.markdown(_now_html(f.now), unsafe_allow_html=True)
        elif run is None:
            st.markdown('<p><span class="sh-live" aria-hidden="true"></span>Getting ready…</p>',
                        unsafe_allow_html=True)
        progress = []
        if f.to_open:
            progress.append(f"Opened {f.opened} of {f.to_open} listings")
        if f.to_check:
            progress.append(f"checked {f.checked} of {f.to_check}")
        if progress:
            st.markdown(", ".join(progress) + ".")
        if f.events:
            st.markdown("**What we've done so far**")
            st.markdown('<ul class="sh-log">' + "".join(f"<li>{theme.e(e)}</li>" for e in f.events) + "</ul>",
                        unsafe_allow_html=True)

    live()
    st.write("")
    c1, _ = st.columns([1, 4])
    if c1.button("Cancel search"):
        workers.stop(f"search-{req.id}")
        st.query_params.clear()
        st.rerun()


# ---------------------------------------------------------------- results


REPORT_STATUS = {
    "queued": ("Sending", theme.TEAL), "filing": ("Sending", theme.TEAL), "submitted": ("Submitted", theme.GREEN),
    "waiting_human": ("Needs you", theme.AMBER), "failed": ("Not sent", theme.RED),
}


def _link(url: str, text: str) -> str:
    return f'<a href="{theme.e(url)}" target="_blank" rel="noopener">{theme.e(text)} ↗</a>'


@st.dialog("Report this listing", width="large")
def report_dialog(b: CaseBundle, search_id: str | None) -> None:
    site = reports.site_label(b.listing.platform)
    st.markdown(f"Our agent will fill in **{theme.e(site)}'s report form** with the details below and send it. "
                f"You'll get a link to the submitted report.")
    st.markdown(f'<div class="sh-dlg-listing">{theme.e(b.listing.title)}<br><span>{theme.money(b.listing.price)} '
                f'on {theme.e(site)}, sold by {theme.e(b.listing.seller_name or "unknown")}</span></div>',
                unsafe_allow_html=True)
    keys = list(reports.REASONS)
    default = reports.REASON_FOR_CATEGORY.get(str(b.case.category), "counterfeit")
    reason_key = st.selectbox("What's wrong with this listing?", keys, index=keys.index(default),
                              format_func=reports.REASONS.get)
    details = st.text_area("Describe the problem", value=reports.draft_details(b, _brand()), height=140,
                           help="We've written down what we found. Edit it if you like.")
    rel_keys = list(reports.RELATIONSHIPS)
    relationship = st.radio("How did you come across it?", rel_keys, index=rel_keys.index("shopper"),
                            format_func=reports.RELATIONSHIPS.get)
    c1, c2 = st.columns(2)
    name = c1.text_input("Your name", value=st.session_state.get("reporter_name", ""))
    email = c2.text_input("Your email", value=st.session_state.get("reporter_email", ""),
                          help="The marketplace uses this to tell you what happened.")
    attach = st.checkbox("Attach a screenshot of the listing as evidence", value=b.screenshot is not None,
                         disabled=b.screenshot is None)
    declaration = st.checkbox(reports.DECLARATION)
    tell_brand = st.checkbox("Also let the brand's legal team know", value=False,
                             help="Their lawyers can ask the marketplace to remove it on the brand's behalf.")
    form = reports.ReportForm(reason_key, details, name, email, relationship, attach, declaration)
    s1, s2, _ = st.columns([1.2, 1, 2])
    if s1.button(f"Submit report to {site}", type="primary", width="stretch"):
        errors = form.errors()
        if errors:
            for msg in errors.values():
                st.error(msg)
            return
        st.session_state["reporter_name"], st.session_state["reporter_email"] = name.strip(), email.strip()
        reports.create(b, form, search_id)
        if tell_brand:
            try:
                report.report(b)
            except Exception as e:  # the marketplace report still goes ahead
                st.toast(f"We couldn't notify the brand's legal team: {e}")
        st.toast(f"Sending your report to {site}. Follow it under Your reports.")
        st.rerun()
    if s2.button("Cancel", width="stretch"):
        st.rerun()


def _report_controls(b: CaseBundle, search_id: str | None) -> None:
    current = reports.for_url(b.listing.url)
    site = reports.site_label(b.listing.platform)
    if current is None:
        if st.button(f"Report to {site}", key=f"report_{b.case.id}"):
            report_dialog(b, search_id)
        return
    if current.status in reports.ACTIVE:
        _report_status_live(b.listing.url, site)  # refreshes itself until the agent finishes
    else:
        _report_status(current, site)


def _report_status(current: Any, site: str) -> None:
    word, colour = REPORT_STATUS.get(current.status, ("Sending", theme.TEAL))
    if current.status == "submitted" and current.report_url:
        text = (f"Reported to {theme.e(site)} ({theme.e(current.report_ref)}). "
                + _link(current.report_url, "View the submitted form"))
    elif current.status == "waiting_human":
        text = f"{theme.e(site)} asked for a security check. See Your reports."
    elif current.status == "failed":
        text = "The report wasn't sent. See Your reports to try again."
    else:
        text = f"Sending your report to {theme.e(site)}…"
    st.markdown(f'<div class="sh-reported" style="color:{colour}">{text}</div>', unsafe_allow_html=True)


@st.fragment(run_every=2)
def _report_status_live(url: str, site: str) -> None:
    current = reports.for_url(url)
    if current is not None:
        _report_status(current, site)


def _sidebar_item(r: Any) -> None:
    word, colour = REPORT_STATUS.get(r.status, ("Sending", theme.TEAL))
    site = reports.site_label(r.site)
    extra = ""
    if r.status == "submitted" and r.report_url:
        extra = f'<div class="sh-side-link">{_link(r.report_url, "Open the submitted form")}</div>'
    elif r.status in reports.ACTIVE and r.progress:
        extra = f'<div class="sh-side-progress"><span class="sh-live"></span>{theme.e(r.progress)}</div>'
    elif r.status == "waiting_human":
        extra = (f'<div class="sh-side-progress">{theme.e(site)} wants to check you are human. Complete the check, '
                 f'then press Try again.</div>')
        if r.waiting_url:
            extra += f'<div class="sh-side-link">{_link(r.waiting_url, "Open the check")}</div>'
    elif r.status == "failed":
        extra = f'<div class="sh-side-progress">{theme.e((r.error or "")[:140])}</div>'
    ref = f" {theme.e(r.report_ref)}" if r.report_ref else ""
    st.markdown(
        f'<div class="sh-side-item"><div class="sh-side-head"><span class="sh-pill" style="color:{colour}">'
        f'{word}</span><span class="sh-side-site">{theme.e(site)}{ref}</span></div>'
        f'<div class="sh-side-name">{theme.e(r.listing_title)}</div>{extra}</div>', unsafe_allow_html=True)
    if r.status in ("waiting_human", "failed"):
        if st.button("Try again", key=f"retry_{r.id}"):
            reports.retry(r.id)
            st.rerun()


def sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sh-side-title">Your reports</div>', unsafe_allow_html=True)

        @st.fragment(run_every=2)
        def body() -> None:
            rows = reports.all_reports()
            if not rows:
                st.markdown('<p class="sh-side-empty">Reports you send to a marketplace appear here, with a link '
                            'to each submitted form.</p>', unsafe_allow_html=True)
                return
            for r in rows:
                _sidebar_item(r)

        body()


def _card(b: CaseBundle, brand: str, rrp_limit: float, reportable: bool = False,
          search_id: str | None = None) -> None:
    cat = str(b.case.category)
    group = GROUP_BY_KEY[CATEGORY_GROUP[cat]]
    l = b.listing
    official = f'<span>official price {theme.money(b.product.rrp)}</span>' if b.product else ""
    seller = f", sold by {theme.e(l.seller_name)}" if l.seller_name else ""
    with st.container(border=True):
        img, body = st.columns([1, 3], gap="medium")
        if b.listing_image:
            img.image(str(b.listing_image), width="stretch")
        body.markdown(
            f'<span class="sh-badge" style="color:{group.colour}">{theme.e(LABEL.get(cat, cat))}</span>'
            f'<div class="sh-title"><a href="{theme.e(l.url)}" target="_blank" rel="noopener">'
            f'{theme.e(l.title or "Untitled listing")}</a></div>'
            f'<div class="sh-price">{theme.money(l.price)}{official}</div>'
            f'<div class="sh-where">On {theme.e(feed.site_name(l.platform))}{seller}</div>'
            f'<div class="sh-why">{theme.e(reason(cat, l, b.features, brand, rrp_limit))}</div>',
            unsafe_allow_html=True)
        if reportable:
            with body:
                _report_controls(b, search_id)


def summary(grouped: dict[str, list[CaseBundle]], brand: str) -> str:
    n = sum(len(v) for v in grouped.values())
    g, c, a = (len(grouped[k]) for k in ("genuine", "careful", "avoid"))
    parts = [f"<b>{g}</b> {'looks' if g == 1 else 'look'} genuine" if g else "",
             f"<b>{c}</b> {'is' if c == 1 else 'are'} likely genuine" if c else "",
             f"<b>{a}</b> {'is' if a == 1 else 'are'} best avoided" if a else ""]
    parts = [x for x in parts if x]
    if not parts:
        return f"We checked {n} listings. None of them are the brand you searched for, but some are similar."
    said = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"We checked {n} listings. {said}."


def results(req: data.SearchRequest) -> None:
    brand = _brand()
    run = data.find_run(req.id)
    top, btn = st.columns([5, 1], vertical_alignment="center")
    with top:
        query_header(req)
    with btn:
        new_search_button("new_top")

    items = data.results(run.id) if run else []  # type: ignore[arg-type]
    failed = run is None or run.status == RunStatus.failed
    if failed:
        st.error("The search stopped before it finished." + (" Here's what we found before it stopped."
                                                              if items else " Please try again."))
        if st.button("Try again", type="primary"):
            workers.start(f"search-{req.id}", text=req.text, photo=req.photo)
            st.rerun()
        with st.expander("Details for developers"):
            st.code(workers.tail(f"search-{req.id}", 25) or "No output.", language=None)

    grouped: dict[str, list[CaseBundle]] = {g.key: [] for g in GROUPS}
    for b in items:
        grouped[CATEGORY_GROUP[str(b.case.category)]].append(b)
    for k in grouped:
        grouped[k].sort(key=lambda b: (b.listing.price is None, b.listing.price or 0))

    if not items and not failed:
        theme.empty("No matching listings found",
                    "Try the product's name as the brand writes it, or a clearer photo of the product.")
    elif items:
        st.markdown(f'<p class="sh-summary">{summary(grouped, brand)}</p>', unsafe_allow_html=True)

    limit = get_config().thresholds.counterfeit_price_ratio
    for grp in GROUPS:
        rows = grouped[grp.key]
        if not rows:
            continue
        with st.expander(f":{grp.tone}[**{grp.title}**] ({len(rows)})", expanded=grp.key != "lookalike"):
            st.markdown(f'<div class="sh-group-blurb">{theme.e(grp.blurb)}</div>', unsafe_allow_html=True)
            if grp.key == "avoid":
                st.caption("Spotted a fake? Report it to the marketplace. Our agent fills in their report form for "
                           "you.")
            for b in rows:
                _card(b, brand, limit, reportable=grp.key == "avoid", search_id=req.id)

    if run is not None:
        f = feed.build(data.tasks(run.id), brand, finished=True)  # type: ignore[arg-type]
        if f.events:
            with st.expander("How we searched"):
                st.markdown('<ul class="sh-log">' + "".join(f"<li>{theme.e(e)}</li>" for e in reversed(f.events))
                            + "</ul>", unsafe_allow_html=True)
    st.markdown('<p class="sh-fine">We compare listings with the brand\'s official products, sellers and '
                'prices. It\'s a guide, not a guarantee. If in doubt, buy from an official seller.</p>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- router


def main() -> None:
    st.set_page_config(page_title="Bounty Hunter — find genuine listings", page_icon="🔎", layout="centered",
                       initial_sidebar_state="expanded")
    theme.inject()
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()
    sid = st.query_params.get("search")
    req = data.load_search(sid) if sid else None
    if req is None:
        if sid:
            st.query_params.clear()
        home()
        return
    run = data.find_run(req.id)
    w = workers.status(f"search-{req.id}")
    if (run is not None and run.status != RunStatus.running) or not w.alive:
        results(req)
    else:
        searching(req)


main()
