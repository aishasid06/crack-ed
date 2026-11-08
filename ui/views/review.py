"""Review queue: cases waiting for a lawyer, ordered by expected value. Approve and submit / Reject write to
the DB through bh.enforce.approvals; the submit loop does the rest."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bh.config import get_config
from bh.enforce.approvals import DecisionError, approve_case, confirm_final, reject_case
from ui import theme, workers
from ui.data import CaseBundle, awaiting_final_confirm, review_queue
from ui.views import dossier


def reviewer() -> str:
    return str(st.session_state.get("reviewer", "") or "").strip()


def _queue_table(queue: list[CaseBundle]) -> int | None:
    df = pd.DataFrame([{
        "Rank": i, "Case": b.case.id, "Expected value": b.case.expected_value,
        "Category": theme.CATEGORY_LABEL.get(str(b.case.category), str(b.case.category)),
        "Confidence": b.case.confidence, "Platform": b.listing.platform.capitalize(),
        "Listing": b.listing.title, "Draft": "Ready" if b.case.portal_draft_id else "Not drafted",
        "In budget": b.case.in_review_budget,
        "Model": "disagrees — review" if b.case.model_disagrees else "",
    } for i, b in enumerate(queue, 1)])
    ev = st.dataframe(
        df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        height=min(38 + 35 * len(df), 300), key="queue_table",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Case": st.column_config.NumberColumn(width="small"),
            "Expected value": st.column_config.NumberColumn(format="£%.2f"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Listing": st.column_config.TextColumn(width="large"),
            "In budget": st.column_config.CheckboxColumn(help="Selected within the lawyer review budget"),
        })
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        st.session_state["review_case"] = int(df.iloc[rows[0]]["Case"])
    ids = [b.case.id for b in queue]
    current = st.session_state.get("review_case")
    return current if current in ids else ids[0]


def _decide(b: CaseBundle) -> None:
    who = reviewer()
    st.markdown("**Declarations you sign by approving**")
    st.caption("Approving authorises the agent to tick these declarations in the portal in your name, check the "
               "review page against this case, and submit the notice.")
    dossier.declarations(who or None)
    st.write("")
    c1, c2, _ = st.columns([1.3, 1, 2])
    if not who:
        st.info("Enter your full name under “Reviewing as” in the sidebar to approve or reject.")
    if c1.button("Approve and submit", type="primary", disabled=not who, width="stretch",
                 key=f"approve_{b.case.id}"):
        try:
            approve_case(b.case.id, who)  # type: ignore[arg-type]
            st.session_state.pop("review_case", None)
            st.toast(f"Case {b.case.id} approved. The submit loop will file it.")
            st.rerun()
        except DecisionError as e:
            st.error(str(e))
    if c2.button("Reject", disabled=not who, width="stretch", key=f"reject_{b.case.id}"):
        try:
            reject_case(b.case.id, who)  # type: ignore[arg-type]
            st.session_state.pop("review_case", None)
            st.toast(f"Case {b.case.id} rejected. Nothing will be submitted.")
            st.rerun()
        except DecisionError as e:
            st.error(str(e))


def _final_confirmations() -> None:
    pending = awaiting_final_confirm()
    if not pending:
        return
    st.subheader("Waiting for your final confirmation")
    theme.lede("The review page for these approved cases matched exactly. Confirm to click Submit.")
    for b in pending:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**Case {b.case.id}** — {theme.e(b.listing.title)}  \nApproved by {theme.e(b.case.decided_by)}"
                    f", verified {b.case.review_verified_at:%d %b %H:%M}")
        if c2.button("Confirm and submit", key=f"final_{b.case.id}", type="primary",
                     disabled=reviewer() != (b.case.decided_by or "")):
            try:
                confirm_final(b.case.id, reviewer())  # type: ignore[arg-type]
                st.toast(f"Case {b.case.id} confirmed.")
                st.rerun()
            except DecisionError as e:
                st.error(str(e))
    st.divider()


def render() -> None:
    st.title("Review queue")
    theme.not_legal_advice()
    if get_config().enforce.require_final_confirm:
        _final_confirmations()
    queue = review_queue()
    if not queue:
        theme.empty("Nothing to review", "Cases appear here when a run finds listings that warrant a platform "
                                         "report. Start a run from the Run page.")
        return
    if not workers.status("submit").alive:
        c1, c2 = st.columns([4, 1])
        c1.warning("The submit loop isn't running, so approved cases will wait until it starts.")
        if c2.button("Start submit loop", width="stretch", disabled=not workers.mock_sites_up()):
            workers.start("submit")
            st.rerun()
    theme.lede(f"{len(queue)} cases, highest expected value first. Select a row to open its case file.")
    selected = _queue_table(queue)
    b = next(x for x in queue if x.case.id == selected)
    dossier.full(b)
    st.write("")
    _decide(b)
