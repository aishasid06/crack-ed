"""Trace: task tree (parent/child), per-task timing, and every LLM call with its model and tokens."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from bh.trace import WORK_TYPES, compute_run_totals
from ui import theme
from ui.data import llm_calls_for_runs, runs, stage_of, task_tree, tasks_for_runs

STATUSES = ["succeeded", "running", "waiting_human", "failed", "pending"]


def _aware(dt: datetime | None) -> datetime | None:
    return None if dt is None else (dt if dt.tzinfo else dt.replace(tzinfo=UTC))


def detail(t: Any) -> str:
    """One line on what a task did: the navigator's decision and reason, a replayed recipe step, a query."""
    p = t.payload or {}
    if t.type == "agent:model" and p.get("action"):
        why = f": {p['reason']}" if p.get("reason") else ""
        return f"chose {p['action']} {p.get('target') or ''}".strip() + why
    if t.type in ("agent:action", "recipe:step") and p.get("action"):
        tail = " (recipe replay)" if t.type == "recipe:step" else ""
        return f"{p['action']} {p.get('target') or ''}".strip() + tail
    if t.type == "agent:consent":
        return f"cookie banner: clicked {p.get('button', '?')}"
    if t.type in ("agent:task", "recipe:task"):
        bits = [f"{p.get('task', '')} {p.get('subject', '')}".strip()]
        if p.get("mode"):
            bits.append(f"mode {p['mode']}")
        if p.get("recipe_version"):
            bits.append(f"recipe v{p['recipe_version']}")
        return ", ".join(b for b in bits if b)
    if t.type == "discovery" and p.get("query"):
        return f"query “{p['query']}”"
    return ""


def render() -> None:
    st.title("Trace")
    theme.lede("What each run did, how long it took, and which model spent which tokens.")
    run_rows = runs(limit=200)
    if not run_rows:
        theme.empty("No runs yet", "Each pipeline stage records a run with its tasks and LLM calls.")
        return
    labels = {r.id: f"Run {r.id} — {stage_of(r)} — {r.started_at:%d %b %H:%M} — {r.status}" for r in run_rows}
    rid = st.selectbox("Run", list(labels), format_func=labels.get)
    run = next(r for r in run_rows if r.id == rid)
    totals = run.totals if run.finished_at and run.totals else compute_run_totals(rid)  # type: ignore[arg-type]
    tasks = tasks_for_runs([rid])  # type: ignore[list-item]
    calls = llm_calls_for_runs([rid])  # type: ignore[list-item]

    m = st.columns(5)
    m[0].metric("Wall time", f"{totals.get('wall_time_s', 0):,.1f}s")
    m[1].metric("Tasks", totals.get("tasks", {}).get("total", len(tasks)))
    m[2].metric("Max parallel work", totals.get("max_parallelism", "—"),
                help="Most discovery / capture / draft / submit tasks running at once.")
    llm = totals.get("llm", {})
    m[3].metric("LLM calls", llm.get("calls", 0))
    m[4].metric("LLM tokens", f"{llm.get('input_tokens', 0) + llm.get('output_tokens', 0):,}",
                help=f"{llm.get('input_tokens', 0):,} input, {llm.get('output_tokens', 0):,} output")
    if totals.get("sites"):
        st.caption("Sites: " + ", ".join(totals["sites"]))

    if not tasks:
        theme.empty("No tasks", "This run recorded no tasks.")
        return

    now = datetime.now(UTC)
    t0 = min(_aware(t.started_at) for t in tasks if t.started_at) if any(t.started_at for t in tasks) else now
    work = [t for t in tasks if t.started_at and t.type in WORK_TYPES]
    if work:
        st.markdown("**Timeline of work tasks**")
        gantt = pd.DataFrame([{
            "lane": f"{t.type} · {t.site}" if t.site else t.type, "task": t.dedup_key,
            "status": theme.STATUS_STYLE.get(str(t.status), (str(t.status),))[0],
            "start": (_aware(t.started_at) - t0).total_seconds(),  # type: ignore[operator]
            "end": ((_aware(t.finished_at) or now) - t0).total_seconds(),  # type: ignore[operator]
        } for t in work])
        gantt["seconds"] = (gantt["end"] - gantt["start"]).round(2)
        colours = [theme.STATUS_STYLE[s][1] for s in STATUSES]
        chart = alt.Chart(gantt).mark_bar(cornerRadius=2, opacity=0.75).encode(
            x=alt.X("start:Q", title="Seconds since run start", axis=alt.Axis(gridColor=theme.RULE)),
            x2="end:Q",
            y=alt.Y("lane:N", title=None, scale=alt.Scale(paddingInner=0.35)),
            color=alt.Color("status:N", scale=alt.Scale(domain=[theme.STATUS_STYLE[x][0] for x in STATUSES],
                                                         range=colours),
                            legend=alt.Legend(title="Status", orient="top")),
            tooltip=[alt.Tooltip("task:N", title="Task"), alt.Tooltip("status:N", title="Status"),
                     alt.Tooltip("seconds:Q", title="Duration (s)")],
        ).properties(height=alt.Step(34))
        st.altair_chart(chart.configure_view(stroke=None).configure_axis(labelColor=theme.MUTED, domain=False),
                        width="stretch")

    st.markdown("**Task tree**")
    calls_by_task: dict[int, list] = {}
    for c in calls:
        if c.task_id is not None:
            calls_by_task.setdefault(c.task_id, []).append(c)
    rows = []
    for depth, t in task_tree(tasks):
        start, end = _aware(t.started_at), _aware(t.finished_at)
        tc = calls_by_task.get(t.id or -1, [])
        rows.append({
            "Task": ("    " * depth) + ("└ " if depth else "") + t.dedup_key, "Type": t.type, "Site": t.site,
            "Status": theme.STATUS_STYLE.get(str(t.status), (str(t.status),))[0],
            "Start (s)": round((start - t0).total_seconds(), 2) if start else None,
            "Duration (s)": round(((end or now) - start).total_seconds(), 2) if start else None,
            "Detail": detail(t), "Attempts": t.attempts, "LLM calls": len(tc),
            "Tokens": sum(c.input_tokens + c.output_tokens for c in tc), "Error": t.error,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(38 + 35 * len(rows), 520),
                 column_config={
        "Task": st.column_config.TextColumn(width="large"),
        "Detail": st.column_config.TextColumn(width="large"),
        "Tokens": st.column_config.NumberColumn(format="localized"),
    })

    st.markdown("**LLM calls**")
    if not calls:
        st.caption("No LLM calls in this run.")
        return
    task_key = {t.id: t.dedup_key for t in tasks}
    st.dataframe(pd.DataFrame([{
        "Time": c.created_at, "Model": c.model, "Requested": c.requested_model, "Purpose": c.purpose,
        "Task": task_key.get(c.task_id), "Input": c.input_tokens, "Output": c.output_tokens,
        "Cache read": c.cache_read_input_tokens, "Latency (ms)": c.latency_ms, "Stop": c.stop_reason,
        "Error": c.error,
    } for c in calls]), hide_index=True, width="stretch")
