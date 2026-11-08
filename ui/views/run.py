"""Run page: start a run (spawns the worker process), live task counts, elapsed time, tokens by model."""

from __future__ import annotations

from datetime import UTC, datetime

import altair as alt
import pandas as pd
import streamlit as st

from bh.config import get_config
from ui import theme, workers
from ui.data import brands, llm_calls_for_runs, runs, stage_of, task_status_counts, tasks_for_runs, \
    tokens_by_model

STATUS_ORDER = ["running", "pending", "succeeded", "waiting_human", "failed"]


def _elapsed(start: datetime | None, end: datetime | None = None) -> str:
    if start is None:
        return "—"
    s = int(((end or datetime.now(UTC)) - (start if start.tzinfo else start.replace(tzinfo=UTC))).total_seconds())
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def _worker_row(w: workers.WorkerState, title: str, detail: str) -> str:
    when = f"for {_elapsed(w.started_at)}" if w.alive else (
        f"last run {w.started_at:%d %b %H:%M}" if w.started_at else "")
    return (f'<div class="bh-worker"><div><span class="name">{theme.e(title)}</span><br>'
            f'<span class="meta">{theme.e(detail)}</span></div><div>{theme.pill(w.label.replace(" ", "_"))}'
            f' <span class="meta">{theme.e(when)}</span></div></div>')


def _start_panel() -> None:
    cfg = get_config()
    pipe = workers.status("pipeline")
    known = sorted({b.slug for b in brands()} | {cfg.brand})
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Start a run")
        brand = st.selectbox("Brand", known, index=known.index(cfg.brand),
                             help="Brands are configured in config.yaml (`brand`, `paths.brand_dir`).")
        use_llm = st.toggle("Use the LLM fallback extractor and borderline checks", value=False,
                            help="Off: deterministic adapters and the calibrated scorer only — no tokens spent.")
        mock_up = workers.mock_sites_up()
        if not mock_up:
            st.warning("The mock marketplaces aren't reachable on "
                       f"{cfg.mocksites.host}:{cfg.mocksites.port}. Start them below before running.")
        c1, c2 = st.columns(2)
        if pipe.alive:
            if c1.button("Stop run", type="secondary", width="stretch"):
                workers.stop("pipeline")
                st.rerun()
        elif c1.button("Start run", type="primary", width="stretch", disabled=not mock_up):
            workers.start("pipeline", extra={"brand": brand, "llm": use_llm}, brand=brand, use_llm=use_llm)
            st.toast("Run started")
            st.rerun()
        if not workers.has_module("bh.orchestrator.run"):
            st.caption("Runs the stage chain (discover → classify → decide → draft). The LangGraph "
                       "orchestrator replaces it once `bh/orchestrator/run.py` exists.")
    with right:
        st.subheader("Workers")
        mock = workers.status("mock")
        sub = workers.status("submit")
        st.markdown(
            _worker_row(pipe, "Pipeline", "Search, capture, classify, rank, draft") +
            _worker_row(sub, "Submit loop", "Signs, verifies and submits approved cases") +
            (_worker_row(mock, "Mock sites", "MockBay, TradeLoft and the report portal") if not mock_up or mock.alive
             else '<div class="bh-worker"><div><span class="name">Mock sites</span><br><span class="meta">'
                  'Running outside the UI</span></div><div>' + theme.pill("running") + '</div></div>'),
            unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if sub.alive:
            if b1.button("Stop submit loop", width="stretch"):
                workers.stop("submit")
                st.rerun()
        elif b1.button("Start submit loop", width="stretch", disabled=not mock_up):
            workers.start("submit")
            st.rerun()
        if mock.alive:
            if b2.button("Stop mock sites", width="stretch"):
                workers.stop("mock")
                st.rerun()
        elif not mock_up and b2.button("Start mock sites", width="stretch"):
            workers.start("mock")
            st.toast("Starting mock sites…")
            st.rerun()


@st.fragment(run_every=2)
def _live() -> None:
    pipe = workers.status("pipeline")
    prog = workers.progress()
    since = pipe.started_at
    run_rows = runs(limit=40, since=since) if since else runs(limit=1)
    run_ids = [r.id for r in run_rows if r.id is not None]
    tasks = tasks_for_runs(run_ids)
    calls = llm_calls_for_runs(run_ids)
    counts = task_status_counts(tasks)
    tokens = tokens_by_model(calls)

    st.subheader("Current run" if pipe.alive else "Latest run")
    if not run_rows:
        theme.empty("No runs yet", "Start the mock sites, then start a run. Task counts and token use "
                                   "appear here while it works.")
        return
    end = None if pipe.alive else max((r.finished_at for r in run_rows if r.finished_at), default=None)
    m = st.columns(5)
    m[0].metric("Elapsed", _elapsed(since or run_rows[-1].started_at, end))
    m[1].metric("Tasks", f"{len(tasks):,}")
    m[2].metric("Running", counts["running"])
    m[3].metric("Need a person", counts["waiting_human"])
    m[4].metric("LLM tokens", f"{sum(t['input_tokens'] + t['output_tokens'] for t in tokens):,}")

    if prog.get("stages") and since and prog.get("started_at", "") >= since.isoformat()[:19]:
        stages = prog["stages"]
        st.markdown(" ".join(
            f'{theme.pill(s["status"] if s["status"] != "skipped" else "classified")} '
            f'<span style="margin-right:1rem">{theme.e(s["label"])}'
            f'{" · " + str(s["seconds"]) + "s" if s.get("seconds") is not None else ""}</span>'
            for s in stages), unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Tasks by status**")
        df = pd.DataFrame([{"status": theme.STATUS_STYLE[s][0], "key": s, "count": counts[s]} for s in STATUS_ORDER])
        colours = [theme.STATUS_STYLE[s][1] for s in STATUS_ORDER]
        chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=18).encode(
            x=alt.X("count:Q", title=None, axis=alt.Axis(grid=True, gridColor=theme.RULE, tickMinStep=1)),
            y=alt.Y("status:N", sort=[theme.STATUS_STYLE[s][0] for s in STATUS_ORDER], title=None),
            color=alt.Color("key:N", scale=alt.Scale(domain=STATUS_ORDER, range=colours), legend=None),
            tooltip=[alt.Tooltip("status:N", title="Status"), alt.Tooltip("count:Q", title="Tasks")],
        ).properties(height=170)
        text = chart.mark_text(align="left", dx=6, color=theme.INK).encode(text="count:Q", color=alt.value(theme.INK))
        st.altair_chart((chart + text).configure_view(stroke=None).configure_axis(
            labelColor=theme.MUTED, domain=False, ticks=False, labelFontSize=12), width="stretch")
    with right:
        st.markdown("**Tokens by model**")
        if tokens:
            st.dataframe(pd.DataFrame(tokens), hide_index=True, width="stretch", column_config={
                "model": "Model", "calls": st.column_config.NumberColumn("Calls"),
                "input_tokens": st.column_config.NumberColumn("Input", format="localized"),
                "output_tokens": st.column_config.NumberColumn("Output", format="localized"),
                "latency_ms": st.column_config.NumberColumn("Total latency (ms)", format="localized")})
        else:
            st.caption("No LLM calls in this run. Deterministic adapters and the scorer handled everything.")

    with st.expander("Stage runs"):
        st.dataframe(pd.DataFrame([{
            "Run": r.id, "Stage": stage_of(r), "Status": str(r.status),
            "Started": r.started_at, "Wall time (s)": (r.totals or {}).get("wall_time_s"),
            "Tasks": (r.totals or {}).get("tasks", {}).get("total"),
        } for r in run_rows]), hide_index=True, width="stretch")
    with st.expander("Worker log"):
        st.code(workers.tail("pipeline", 60) or "No output yet.", language=None)


def render() -> None:
    st.title("Run")
    theme.lede("Search the marketplaces for listings that use the brand, capture the evidence, and prepare "
               "cases for review. Nothing is submitted until a lawyer approves it.")
    _start_panel()
    st.divider()
    _live()
