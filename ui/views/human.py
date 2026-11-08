"""Human required: tasks paused on a challenge page, and cases whose portal review page didn't match the
approved case. The agent never solves challenges — a person does, then presses Done."""

from __future__ import annotations

import streamlit as st

from bh.db import CaseStatus
from bh.enforce.approvals import DecisionError, reject_case, resume_case
from bh.trace import resume_task
from ui import theme, workers
from ui.data import cases_with_status, waiting_tasks
from ui.views import dossier

NEXT_STEP = {
    "draft": "The case goes back to the queue and is drafted again from scratch.",
    "submit": "The submit loop retries the approved case on its next pass.",
}


def _resume(task_id: int, case_id: int | None) -> str:
    resume_task(task_id)
    if case_id is not None:
        try:
            return f"Case {case_id} is now {resume_case(case_id)}."
        except DecisionError:
            pass
    return "Task released."


def _tasks() -> int:
    tasks = waiting_tasks()
    st.subheader("Paused on a security check")
    if not tasks:
        st.caption("No tasks are paused.")
        return 0
    theme.lede("Open the page in your own browser and complete the check. The portal clears the check for this "
               "machine, so the agent can carry on once you press Done.")
    for t in tasks:
        case_id = t.payload.get("case_id")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1.2])
            what = f"case {case_id}" if case_id is not None else t.dedup_key
            when = f"  \nPaused at {t.finished_at:%d %b %H:%M:%S}" if t.finished_at else ""
            c1.markdown(f"**{t.type.capitalize()}** on {t.site or 'unknown site'} — {theme.e(what)}{when}")
            if t.error:
                c1.caption(t.error)
            c1.caption(NEXT_STEP.get(t.type, "Start a new run to pick this work up again."))
            if t.waiting_url:
                c2.link_button("Open page", t.waiting_url, width="stretch")
            if c2.button("Done", key=f"done_{t.id}", type="primary", width="stretch"):
                try:
                    msg = _resume(t.id, case_id)  # type: ignore[arg-type]
                    if t.type == "draft" and workers.mock_sites_up():
                        workers.start("draft")
                        msg += " Drafting restarted."
                    st.toast(msg)
                    st.rerun()
                except (KeyError, ValueError) as e:
                    st.error(str(e))
    return len(tasks)


def _blocked_cases() -> int:
    waiting = cases_with_status(CaseStatus.waiting_human)
    blocked = [b for b in waiting if b.case.review_diff]
    others = [b for b in waiting if not b.case.review_diff]
    st.subheader("Stopped before Submit")
    if not blocked:
        st.caption("No case has been stopped by a review-page mismatch.")
    else:
        theme.lede("The portal's review page differed from what was approved, so nothing was submitted. Redraft "
                   "to rebuild the portal report from the approved case, or reject it.")
    who = str(st.session_state.get("reviewer", "") or "").strip()
    for b in blocked:
        with st.container(border=True):
            st.markdown(f"**Case {b.case.id}** — {theme.e(b.listing.title)}  \nApproved by "
                        f"{theme.e(b.case.decided_by or '—')}")
            st.markdown(theme.diff_table(b.case.review_diff), unsafe_allow_html=True)
            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("Redraft", key=f"redraft_{b.case.id}", type="primary", width="stretch"):
                st.toast(f"Case {b.case.id} is now {resume_case(b.case.id)}.")  # type: ignore[arg-type]
                st.rerun()
            if c2.button("Reject", key=f"hreject_{b.case.id}", disabled=not who, width="stretch"):
                reject_case(b.case.id, who)  # type: ignore[arg-type]
                st.rerun()
            with st.expander("Case file"):
                dossier.full(b)
    if others:
        st.subheader("Other paused cases")
        for b in others:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**Case {b.case.id}** — {theme.e(b.listing.title)}")
            if c2.button("Resume", key=f"resume_{b.case.id}", width="stretch"):
                st.toast(f"Case {b.case.id} is now {resume_case(b.case.id)}.")  # type: ignore[arg-type]
                st.rerun()
    return len(waiting)


def render() -> None:
    st.title("Human required")
    theme.lede("Work the agent stopped and handed to a person.")
    n = _tasks()
    st.divider()
    m = _blocked_cases()
    if not n and not m:
        st.write("")
        theme.empty("Nothing needs you", "When a site shows a security check, or a review page doesn't match an "
                                         "approved case, it waits here.")
