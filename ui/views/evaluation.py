"""Evaluation: renders the reports written by ``eval/evaluate.py`` (SPEC §9 Phase 10), one folder per
evaluated run and split: ``<data_dir>/eval/run_<id>_<split>/metrics.json``. Nothing here is computed by the UI.

Shape used (see eval/metrics.py): ``headline`` and ``by_site`` / ``by_difficulty`` → ``metrics`` map each
metric name to ``{value, ci_low, ci_high, n, num?, den?, wilson_low?, wilson_high?}``; ``classification`` holds
``category_confusion_captured`` / ``action_confusion_end_to_end`` as ``{rows, cols, matrix}``; ``pr_curve`` holds
``points`` ``[{threshold, precision, recall}]`` and the ``act_threshold`` operating point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from bh.config import get_config
from ui import theme, workers

HEADLINE = [
    ("discovery_recall", "Discovery recall", "Actionable listings that were captured at all, before any judging."),
    ("report_precision", "Precision on “should report”",
     "Of the listings the pipeline would report to a platform, the share that should be reported."),
    ("escalate_recall_lenient", "Recall on “escalate”",
     "Of the listings that need a lawyer's judgement, the share that reached a lawyer."),
    ("false_alarm_rate", "False-alarm rate",
     "Listings that should be ignored but were sent to a lawyer. Lower is better."),
]
TABLE_METRICS = [("report_precision", "Report precision"), ("escalate_recall_lenient", "Escalate recall"),
                 ("discovery_recall", "Discovery recall"), ("false_alarm_rate", "False alarms")]


def eval_dir() -> Path:
    cfg = get_config()
    return cfg.paths.resolve(cfg.paths.data_dir) / "eval"


def reports() -> list[Path]:
    """metrics.json files, newest first."""
    d = eval_dir()
    found = list(d.glob("*/metrics.json")) + ([d / "metrics.json"] if (d / "metrics.json").exists() else [])
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pct(v: Any) -> str:
    return "—" if v is None else f"{float(v):.0%}"


def _caption(m: dict[str, Any]) -> str:
    bits = []
    if m.get("ci_low") is not None and m.get("ci_high") is not None:
        bits.append(f"95% CI {_pct(m['ci_low'])}–{_pct(m['ci_high'])}")
    if m.get("den"):
        bits.append(f"{m.get('num', 0)} of {m['den']}")
    return ", ".join(bits)


def _headline(rep: dict[str, Any]) -> None:
    head = rep.get("headline", {})
    cols = st.columns(len(HEADLINE))
    for col, (key, label, help_text) in zip(cols, HEADLINE, strict=True):
        m = head.get(key) or {}
        col.metric(label, _pct(m.get("value")), help=help_text)
        if cap := _caption(m):
            col.caption(cap)


def _confusion(block: dict[str, Any] | None, label_of: Any) -> None:
    if not block or not block.get("matrix"):
        st.caption("No rows to show.")
        return
    rows = [label_of(r) for r in block["rows"]]
    cols = [label_of(c) for c in block["cols"]]
    df = pd.DataFrame([{"Truth": rows[i], "Predicted": cols[j], "Count": v}
                       for i, row in enumerate(block["matrix"]) for j, v in enumerate(row)])
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=cols, axis=alt.Axis(labelAngle=-35, title="Predicted")),
        y=alt.Y("Truth:N", sort=rows, title="Truth"),
        tooltip=["Truth", "Predicted", "Count"])
    heat = base.mark_rect(stroke=theme.PAPER, strokeWidth=2).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(range=[theme.SURFACE, theme.PRIMARY]), legend=None))
    vmax = max(int(df["Count"].max()), 1)
    text = base.mark_text(fontSize=12).encode(
        text=alt.condition(alt.datum.Count > 0, "Count:Q", alt.value("")),
        color=alt.condition(alt.datum.Count > vmax / 2, alt.value("#FFFFFF"), alt.value(theme.INK)))
    st.altair_chart((heat + text).properties(height=max(220, 34 * len(rows))).configure_view(stroke=None)
                    .configure_axis(labelColor=theme.MUTED, titleColor=theme.MUTED, domain=False, ticks=False),
                    width="stretch")


def _pr(pr: dict[str, Any] | None) -> None:
    if not pr or not pr.get("points"):
        st.caption((pr or {}).get("note", "No precision–recall curve in this report."))
        return
    df = pd.DataFrame(pr["points"]).rename(columns={"precision": "Precision", "recall": "Recall",
                                                    "threshold": "Threshold"})
    line = alt.Chart(df).mark_line(strokeWidth=2, color=theme.TEAL, interpolate="step-after").encode(
        x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", gridColor=theme.RULE)),
        y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1.02]), axis=alt.Axis(format="%", gridColor=theme.RULE)))
    hover = alt.Chart(df).mark_point(size=90, opacity=0).encode(
        x="Recall:Q", y="Precision:Q",
        tooltip=[alt.Tooltip("Threshold:Q", format=".2f"), alt.Tooltip("Recall:Q", format=".0%"),
                 alt.Tooltip("Precision:Q", format=".0%")])
    layers: list[Any] = [line, hover]
    at = pr.get("act_threshold") or {}
    if at.get("precision") is not None and at.get("recall") is not None:
        pt = pd.DataFrame([{"Recall": at["recall"], "Precision": at["precision"]}])
        layers.append(alt.Chart(pt).mark_point(size=120, filled=True, color=theme.RED, stroke=theme.PAPER,
                                               strokeWidth=2).encode(x="Recall:Q", y="Precision:Q"))
        layers.append(alt.Chart(pt).mark_text(align="left", dx=10, dy=-10, color=theme.INK).encode(
            x="Recall:Q", y="Precision:Q", text=alt.value(f"chosen threshold {at.get('threshold', 0):.2f}")))
    st.altair_chart(alt.layer(*layers).properties(height=360).configure_view(stroke=None)
                    .configure_axis(labelColor=theme.MUTED, titleColor=theme.MUTED, domain=False), width="stretch")
    if pr.get("average_precision") is not None:
        st.caption(f"Average precision {pr['average_precision']:.2f}")


def _breakdown(block: dict[str, Any] | None, label: str) -> None:
    if not block:
        return
    rows = []
    for key, v in block.items():
        m, c = v.get("metrics", {}), v.get("counts", {})
        rows.append({label: key, "Listings": c.get("listings"),
                     **{name: (m.get(k) or {}).get("value") for k, name in TABLE_METRICS}})
    cfg = {name: st.column_config.ProgressColumn(name, min_value=0, max_value=1, format="percent")
           for _, name in TABLE_METRICS}
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=cfg)


def render() -> None:
    st.title("Evaluation")
    theme.lede("How well the pipeline finds and judges listings, measured against ground-truth labels the agent "
               "never sees. Thresholds and the scorer are fitted on the dev split; the test split is held out.")
    ev = workers.status("eval")
    found = reports()
    c1, _ = st.columns([1, 3])
    label = "Evaluating…" if ev.alive else ("Evaluate the latest run" if found else "Run the evaluation")
    if c1.button(label, disabled=ev.alive or not workers.has_module("eval.evaluate"),
                 type="secondary" if found else "primary"):
        workers.start("eval")
        st.rerun()
    if not found:
        theme.empty("No evaluation report yet", "Run the evaluation to score the latest run against the test "
                                                "split. Reports are saved in data/eval/.")
        return
    names = {p: p.parent.name if p.parent != eval_dir() else "metrics.json" for p in found}
    path = st.selectbox("Report", found, format_func=names.get)
    rep = load(path)
    if rep is None:
        st.error(f"{path} isn't a readable evaluation report.")
        return
    run = rep.get("run") or {}
    st.caption(f"Run {run.get('run_id', '?')}, {rep.get('split_note') or rep.get('split', 'test')}, generated "
               f"{str(rep.get('generated_at', ''))[:16].replace('T', ' ')}")
    _headline(rep)

    left, right = st.columns(2, gap="large")
    cls = rep.get("classification") or {}
    with left:
        which = st.radio("Confusion matrix", ["Category (captured listings)", "Action (end to end)"],
                         horizontal=True, label_visibility="collapsed")
        if which.startswith("Category"):
            _confusion(cls.get("category_confusion_captured"), lambda c: theme.CATEGORY_LABEL.get(c, c))
        else:
            _confusion(cls.get("action_confusion_end_to_end"),
                       lambda a: theme.ACTION_LABEL.get(a, a.replace("_", " ").capitalize()))
    with right:
        st.markdown("**Precision–recall for “should report”**")
        _pr(rep.get("pr_curve"))

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**By site**")
        _breakdown(rep.get("by_site"), "Site")
    with b:
        st.markdown("**By difficulty**")
        _breakdown(rep.get("by_difficulty"), "Difficulty")
    md = path.parent / "report.md"
    if md.exists():
        with st.expander("Full written report"):
            st.markdown(md.read_text(encoding="utf-8"))
