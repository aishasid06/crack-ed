"""Cases: every case and its status, filterable; select one to open its case file."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import theme
from ui.data import all_cases
from ui.views import dossier


def render() -> None:
    st.title("Cases")
    theme.lede("Every listing the pipeline classified, with the action it recommends and where it stands.")
    bundles = all_cases()
    if not bundles:
        theme.empty("No cases yet", "Cases are created when a run classifies captured listings.")
        return
    df = pd.DataFrame([{
        "Case": b.case.id, "Run": b.case.run_id, "Status": str(b.case.status),
        "Category": theme.CATEGORY_LABEL.get(str(b.case.category), str(b.case.category)),
        "Action": theme.ACTION_LABEL.get(str(b.case.action), str(b.case.action)),
        "Rule": b.case.rule_id, "Confidence": b.case.confidence, "Expected value": b.case.expected_value,
        "Platform": b.listing.platform.capitalize(), "Seller": b.listing.seller_name, "Price": b.listing.price,
        "Listing": b.listing.title, "Decided by": b.case.decided_by, "Portal ref": b.case.submission_ref,
        "URL": b.listing.url,
    } for b in bundles])

    f1, f2, f3, f4 = st.columns(4)
    status = f1.multiselect("Status", sorted(df["Status"].unique()),
                            format_func=lambda s: theme.STATUS_STYLE.get(s, (s,))[0])
    category = f2.multiselect("Category", sorted(df["Category"].unique()))
    action = f3.multiselect("Action", sorted(df["Action"].unique()))
    platform = f4.multiselect("Platform", sorted(df["Platform"].unique()))
    query = st.text_input("Search listing title or seller", placeholder="e.g. replica")
    latest_only = st.toggle("Latest case per listing only", value=True)

    view = df
    if latest_only:
        view = view.sort_values("Case").groupby("URL", as_index=False).tail(1)
    for col, chosen in (("Status", status), ("Category", category), ("Action", action), ("Platform", platform)):
        if chosen:
            view = view[view[col].isin(chosen)]
    if query:
        q = query.lower()
        view = view[view["Listing"].str.lower().str.contains(q, regex=False)
                    | view["Seller"].fillna("").str.lower().str.contains(q, regex=False)]
    view = view.sort_values("Case", ascending=False)

    st.caption(f"{len(view)} of {len(df)} cases")
    ev = st.dataframe(
        view.assign(Status=view["Status"].map(lambda s: theme.STATUS_STYLE.get(s, (s,))[0])),
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key="cases_table", height=min(38 + 35 * max(len(view), 1), 460),
        column_config={
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Expected value": st.column_config.NumberColumn(format="£%.2f"),
            "Price": st.column_config.NumberColumn(format="£%.2f"),
            "Listing": st.column_config.TextColumn(width="large"),
            "URL": st.column_config.LinkColumn(display_text="open"),
        })
    st.download_button("Download CSV", view.to_csv(index=False).encode("utf-8"), "cases.csv", "text/csv")
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        cid = int(view.iloc[rows[0]]["Case"])
        b = next(x for x in bundles if x.case.id == cid)
        dossier.full(b)
        if b.case.decided_by:
            st.caption(f"Decided by {b.case.decided_by} on {b.case.decided_at:%d %b %Y %H:%M}"
                       + (f"; declarations signed by {b.case.declared_by}" if b.case.declared_by else "")
                       + (f"; portal reference {b.case.submission_ref}" if b.case.submission_ref else ""))
