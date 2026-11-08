"""Evaluation: renders the report written by ``eval/evaluate.py`` (SPEC §9 Phase 10).

The page reads ``<data_dir>/eval/report.json``. Expected shape (every block optional):

    {
      "generated_at": "...", "split": "test",
      "discovery":      {"recall": 0.97, "ci": [0.9, 1.0], "captured": 31, "actionable": 32},
      "should_report":  {"precision": 0.94, "ci": [0.85, 1.0], "recall": 0.9, "n": 18},
      "escalate":       {"recall": 0.86, "ci": [0.7, 0.97], "n": 14},
      "end_to_end":     {"should_report_precision": ..., "escalate_recall": ...},
      "confusion":      {"labels": ["counterfeit_identical", ...], "matrix": [[...], ...]},   # rows = truth
      "pr_curve":       {"precision": [...], "recall": [...], "thresholds": [...], "chosen_threshold": 0.7},
      "by_site":        [{"site": "mockbay", "n": 20, "discovery_recall": ..., "should_report_precision": ...,
                          "escalate_recall": ...}, ...],
      "by_difficulty":  [{"difficulty": "hard", ...same keys...}, ...]
    }

CIs are bootstrap 95% intervals. Nothing on this page is computed by the UI.
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


def report_path() -> Path:
    cfg = get_config()
    return cfg.paths.resolve(cfg.paths.data_dir) / "eval" / "report.json"


def load_report() -> dict[str, Any] | None:
    try:
        return json.loads(report_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pct(v: Any) -> str:
    return "—" if v is None else f"{float(v):.0%}"


def _ci(block: dict[str, Any]) -> str | None:
    ci = block.get("ci")
    return f"95% CI {_pct(ci[0])}–{_pct(ci[1])}" if ci and len(ci) == 2 else None


def _headline(rep: dict[str, Any]) -> None:
    d, sr, esc = rep.get("discovery", {}), rep.get("should_report", {}), rep.get("escalate", {})
    m = st.columns(3)
    m[0].metric("Discovery recall", _pct(d.get("recall")),
                help="Actionable listings that were captured at all, before any classification.")
    m[0].caption(" · ".join(x for x in [_ci(d), f"{d.get('captured')} of {d.get('actionable')} captured"
                                        if d.get("actionable") is not None else None] if x))
    m[1].metric("Precision on “should report”", _pct(sr.get("precision")),
                help="Of the cases we would report to a platform, the share that should be reported.")
    m[1].caption(" · ".join(x for x in [_ci(sr), f"n = {sr['n']}" if sr.get("n") is not None else None] if x))
    m[2].metric("Recall on “escalate”", _pct(esc.get("recall")),
                help="Of the listings that need a lawyer's judgement, the share we escalated.")
    m[2].caption(" · ".join(x for x in [_ci(esc), f"n = {esc['n']}" if esc.get("n") is not None else None] if x))


def _confusion(rep: dict[str, Any]) -> None:
    cm = rep.get("confusion")
    if not cm or not cm.get("matrix"):
        return
    labels = [theme.CATEGORY_LABEL.get(l, l) for l in cm["labels"]]
    df = pd.DataFrame([{"Truth": labels[i], "Predicted": labels[j], "Count": v}
                       for i, row in enumerate(cm["matrix"]) for j, v in enumerate(row)])
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=-35, title="Predicted")),
        y=alt.Y("Truth:N", sort=labels, title="Truth"),
        tooltip=["Truth", "Predicted", "Count"])
    heat = base.mark_rect(stroke=theme.SURFACE, strokeWidth=2).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(range=[theme.SURFACE, theme.PRIMARY]), legend=None))
    vmax = max(df["Count"].max(), 1)
    text = base.mark_text(fontSize=12).encode(
        text=alt.condition(alt.datum.Count > 0, "Count:Q", alt.value("")),
        color=alt.condition(alt.datum.Count > vmax / 2, alt.value("#FFFFFF"), alt.value(theme.INK)))
    st.altair_chart((heat + text).properties(height=380).configure_view(stroke=None)
                    .configure_axis(labelColor=theme.MUTED, titleColor=theme.MUTED, domain=False, ticks=False),
                    width="stretch")


def _pr(rep: dict[str, Any]) -> None:
    pr = rep.get("pr_curve")
    if not pr or not pr.get("precision"):
        return
    df = pd.DataFrame({"Recall": pr["recall"], "Precision": pr["precision"]})
    if pr.get("thresholds"):
        th = list(pr["thresholds"]) + [None] * (len(df) - len(pr["thresholds"]))
        df["Threshold"] = th[: len(df)]
    line = alt.Chart(df).mark_line(strokeWidth=2, color=theme.TEAL, interpolate="step-after").encode(
        x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", gridColor=theme.RULE)),
        y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", gridColor=theme.RULE)))
    hover = alt.Chart(df).mark_point(size=80, opacity=0).encode(
        x="Recall:Q", y="Precision:Q",
        tooltip=[alt.Tooltip("Recall:Q", format=".0%"), alt.Tooltip("Precision:Q", format=".0%")]
        + ([alt.Tooltip("Threshold:Q", format=".2f")] if "Threshold" in df else []))
    layers = [line, hover]
    chosen = pr.get("chosen_threshold")
    if chosen is not None and "Threshold" in df and df["Threshold"].notna().any():
        idx = (df["Threshold"].dropna() - chosen).abs().idxmin()
        pt = df.loc[[idx]]
        layers.append(alt.Chart(pt).mark_point(size=110, filled=True, color=theme.RED, stroke=theme.SURFACE,
                                               strokeWidth=2).encode(x="Recall:Q", y="Precision:Q"))
        layers.append(alt.Chart(pt).mark_text(align="left", dx=10, dy=-10, color=theme.INK).encode(
            x="Recall:Q", y="Precision:Q", text=alt.value(f"chosen threshold {chosen:.2f}")))
    st.altair_chart(alt.layer(*layers).properties(height=380).configure_view(stroke=None)
                    .configure_axis(labelColor=theme.MUTED, titleColor=theme.MUTED, domain=False),
                    width="stretch")


def _breakdown(rows: list[dict[str, Any]], key: str, title: str) -> None:
    if not rows:
        return
    st.markdown(f"**{title}**")
    df = pd.DataFrame(rows)
    cfg = {c: st.column_config.ProgressColumn(c.replace("_", " ").capitalize(), min_value=0, max_value=1,
                                              format="percent")
           for c in df.columns if c.endswith(("recall", "precision"))}
    cfg[key] = st.column_config.TextColumn(key.capitalize())
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg)


def render() -> None:
    st.title("Evaluation")
    theme.lede("How well the pipeline finds and classifies listings, measured on the held-out test split "
               "against ground-truth labels the agent never sees.")
    rep = load_report()
    has_eval = workers.has_module("eval.evaluate")
    ev = workers.status("eval")
    if rep is None:
        if has_eval:
            theme.empty("No evaluation report yet", "Run the evaluation to score the latest run against the "
                                                    "test split.")
        else:
            theme.empty("Evaluation isn't available yet",
                        "eval/evaluate.py arrives in Phase 10. Once it writes data/eval/report.json, the metrics "
                        "appear here.")
    if has_eval:
        label = "Running evaluation…" if ev.alive else ("Run evaluation again" if rep else "Run evaluation")
        if st.button(label, disabled=ev.alive, type="secondary" if rep else "primary"):
            workers.start("eval")
            st.rerun()
    if rep is None:
        return
    st.caption(f"Split: {rep.get('split', 'test')} · generated {rep.get('generated_at', 'unknown')}")
    _headline(rep)
    if e2e := rep.get("end_to_end"):
        st.caption("End to end (discovery misses count as errors): " + ", ".join(
            f"{k.replace('_', ' ')} {_pct(v)}" for k, v in e2e.items() if isinstance(v, (int, float))))
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Confusion matrix** (test split, captured listings)")
        _confusion(rep)
    with right:
        st.markdown("**Precision–recall for “should report”**")
        _pr(rep)
    a, b = st.columns(2, gap="large")
    with a:
        _breakdown(rep.get("by_site", []), "site", "By site")
    with b:
        _breakdown(rep.get("by_difficulty", []), "difficulty", "By difficulty")
