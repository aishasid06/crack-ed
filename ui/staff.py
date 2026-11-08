"""Bounty Hunter — staff UI for the brand's lawyers (SPEC §8): review queue, approvals, trace.

    streamlit run ui/staff.py

The UI reads the DB and writes human decisions only. Pipeline work runs in separate worker processes
(see ui/workers.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run ui/app.py` puts ui/ (not the project) on sys.path
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui import theme  # noqa: E402
from ui.data import case_counts, review_queue, waiting_tasks  # noqa: E402
from ui.views import cases, evaluation, human, review, run, trace  # noqa: E402

st.set_page_config(page_title="Bounty Hunter — staff", page_icon="⚖️", layout="wide")
theme.inject()


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### Bounty Hunter")
        st.caption("Finds listings that infringe the brand's rights and drafts factual takedown notices for a "
                   "lawyer to approve.")
        st.text_input("Reviewing as", key="reviewer", placeholder="Your full name",
                      help="Approvals and rejections are recorded under this name, and the portal declarations "
                           "are signed in it.")
        counts = case_counts()
        queue = len(review_queue())
        waiting = len(waiting_tasks()) + counts.get("waiting_human", 0)
        st.markdown(f"**{queue}** cases to review  \n**{waiting}** items need a person  \n"
                    f"**{counts.get('submitted', 0)}** notices submitted")


_sidebar()

pages = [
    st.Page(run.render, title="Run", icon=":material/play_circle:", url_path="run", default=True),
    st.Page(review.render, title="Review queue", icon=":material/gavel:", url_path="review"),
    st.Page(human.render, title="Human required", icon=":material/front_hand:", url_path="human"),
    st.Page(cases.render, title="Cases", icon=":material/folder_open:", url_path="cases"),
    st.Page(trace.render, title="Trace", icon=":material/account_tree:", url_path="trace"),
    st.Page(evaluation.render, title="Evaluation", icon=":material/insights:", url_path="evaluation"),
]
st.navigation(pages).run()
