"""Visual system for the Streamlit UI: tokens, CSS, and small HTML components.

Direction: an evidence dossier for a brand lawyer. Cool archive paper, ink text, a registry teal for
primary actions, a stamp red reserved for infringement / rejection. Newsreader (serif) carries case titles,
statute citations and the notice letter; Public Sans carries the interface. Status colours always come
with a glyph and a word, never colour alone.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

PAPER = "#121B20"
SURFACE = "#1A252B"
INK = "#E2E8E6"
MUTED = "#93A2A9"
RULE = "#2C3B42"
TEAL = "#5CC0B5"
PRIMARY = "#267C74"  # button fill: white text stays readable
RED = "#E58479"
AMBER = "#DDA84A"
GREEN = "#74C48F"
BLUE = "#86AEDB"

# Element status → (glyph, word, colour)
ELEMENT_STATUS = {
    "met": ("✓", "met", GREEN),
    "unmet": ("✗", "not met", RED),
    "unknown": ("?", "unknown", AMBER),
}

# Case / task status → (word, colour)
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "classified": ("Classified", MUTED),
    "awaiting_approval": ("Awaiting approval", TEAL),
    "drafted": ("Drafted", TEAL),
    "approved": ("Approved", BLUE),
    "submitted": ("Submitted", GREEN),
    "rejected": ("Rejected", RED),
    "waiting_human": ("Needs a person", AMBER),
    "failed": ("Failed", RED),
    "pending": ("Pending", MUTED),
    "running": ("Running", BLUE),
    "succeeded": ("Succeeded", GREEN),
    "finished": ("Finished", GREEN),
    "not_started": ("Not started", MUTED),
    "superseded": ("Replaced after hand-off", MUTED),
}

CATEGORY_LABEL = {
    "authorised": "Authorised seller",
    "genuine_resale": "Genuine resale",
    "counterfeit_identical": "Counterfeit (identical mark)",
    "counterfeit_similar": "Counterfeit (similar mark)",
    "copied_photos": "Copied photos",
    "grey_import": "Grey import",
    "modified_goods": "Modified goods",
    "lookalike": "Look-alike",
    "unrelated": "Unrelated",
}

ACTION_LABEL = {
    "ignore": "Ignore",
    "monitor": "Monitor",
    "platform_report": "Report to platform",
    "escalate": "Escalate to lawyer",
    "refer_trading_standards": "Refer to Trading Standards",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {{
  --bh-paper: {PAPER}; --bh-surface: {SURFACE}; --bh-ink: {INK}; --bh-muted: {MUTED};
  --bh-rule: {RULE}; --bh-teal: {TEAL}; --bh-red: {RED}; --bh-amber: {AMBER}; --bh-green: {GREEN};
}}
.stApp {{ background: var(--bh-paper); color: var(--bh-ink); }}
[data-testid="stSidebar"] {{ background: #162127; border-right: 1px solid var(--bh-rule); }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
h1, h2, h3 {{ font-family: 'Newsreader', Georgia, serif !important; font-weight: 500 !important;
  color: var(--bh-ink); letter-spacing: -0.01em; }}
h1 {{ font-size: 2.35rem !important; line-height: 1.1 !important; }}
h2 {{ font-size: 1.6rem !important; }}
h3 {{ font-size: 1.25rem !important; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
[data-testid="stMetricLabel"] p {{ color: var(--bh-muted); font-size: .85rem; }}
code {{ color: var(--bh-ink); background: #24323A; }}
a {{ color: var(--bh-teal); }}
:focus-visible {{ outline: 2px solid var(--bh-teal) !important; outline-offset: 2px; }}

.bh-lede {{ color: var(--bh-muted); max-width: 68ch; margin: -.4rem 0 1.2rem; font-size: 1rem; }}
.bh-banner {{ border-left: 3px solid var(--bh-amber); background: rgba(221,168,74,.10); padding: .55rem .9rem;
  font-size: .9rem; color: #EBD3A4; margin-bottom: 1.2rem; }}

.bh-pill {{ display: inline-flex; align-items: center; gap: .35rem; font-size: .8rem; font-weight: 600;
  padding: .12rem .55rem; border-radius: 999px; border: 1px solid currentColor; white-space: nowrap; }}
.bh-pill::before {{ content: ""; width: .45rem; height: .45rem; border-radius: 50%; background: currentColor; }}

.bh-dossier-head {{ border-top: 2px solid var(--bh-ink); padding-top: .9rem; margin-top: .4rem; }}
.bh-case-no {{ font-size: .85rem; color: var(--bh-muted); font-variant-numeric: tabular-nums; }}
.bh-case-title {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.9rem; line-height: 1.15;
  margin: .15rem 0 .5rem; color: var(--bh-ink); }}
.bh-facts {{ display: flex; flex-wrap: wrap; gap: .4rem 1.6rem; font-size: .92rem; color: var(--bh-muted); }}
.bh-facts b {{ color: var(--bh-ink); font-weight: 600; font-variant-numeric: tabular-nums; }}

/* The statute strip: the one loud element. */
.bh-statute {{ background: var(--bh-surface); border: 1px solid var(--bh-rule); margin: 1.2rem 0;
  border-radius: 4px; overflow: hidden; }}
.bh-statute-cite {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.7rem; line-height: 1.2;
  padding: 1rem 1.2rem .3rem; color: var(--bh-ink); }}
.bh-statute-sum {{ padding: 0 1.2rem 1rem; color: var(--bh-muted); max-width: 72ch; font-size: .95rem; }}
.bh-elements {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  border-top: 1px solid var(--bh-rule); }}
.bh-el {{ padding: .75rem 1.2rem .85rem; border-right: 1px solid var(--bh-rule);
  border-bottom: 1px solid var(--bh-rule); border-top: 3px solid var(--el); margin-top: -1px; }}
.bh-el-name {{ font-weight: 600; font-size: .92rem; color: var(--bh-ink); display: flex; gap: .45rem; }}
.bh-el-glyph {{ color: var(--el); font-weight: 700; }}
.bh-el-state {{ font-size: .78rem; color: var(--el); font-weight: 600; }}
.bh-el-ev {{ font-size: .85rem; color: var(--bh-muted); margin-top: .25rem; line-height: 1.4; }}
.bh-overlay {{ padding: .7rem 1.2rem; background: rgba(229,132,121,.10); color: #F0C2BC; font-size: .9rem;
  border-top: 1px solid var(--bh-rule); }}

.bh-figure {{ font-size: .82rem; color: var(--bh-muted); margin-top: .3rem; }}
.bh-letter {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.04rem; line-height: 1.6;
  background: var(--bh-surface); border: 1px solid var(--bh-rule); padding: 1.2rem 1.5rem;
  white-space: pre-wrap; max-width: 76ch; color: var(--bh-ink); }}
.bh-decl {{ background: var(--bh-surface); border: 1px solid var(--bh-rule); border-left: 3px solid var(--bh-teal);
  padding: .9rem 1.2rem; max-width: 76ch; }}
.bh-decl ol {{ margin: 0 0 .6rem 1.1rem; padding: 0; }}
.bh-decl li {{ margin-bottom: .35rem; }}
.bh-sign {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.15rem; border-top: 1px dashed var(--bh-rule);
  padding-top: .5rem; }}
.bh-kv {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
.bh-kv th {{ text-align: left; color: var(--bh-muted); font-weight: 500; padding: .3rem .8rem .3rem 0;
  width: 38%; vertical-align: top; }}
.bh-kv td {{ padding: .3rem 0; border-bottom: 1px solid var(--bh-rule); color: var(--bh-ink);
  font-variant-numeric: tabular-nums; word-break: break-word; }}
.bh-diff td, .bh-diff th {{ padding: .35rem .6rem; border-bottom: 1px solid var(--bh-rule); font-size: .88rem;
  text-align: left; vertical-align: top; }}
.bh-diff .exp {{ color: {GREEN}; }} .bh-diff .act {{ color: {RED}; }}
.bh-empty {{ border: 1px dashed var(--bh-rule); padding: 1.6rem; color: var(--bh-muted); text-align: left;
  background: rgba(255,255,255,.02); max-width: 70ch; }}
.bh-empty b {{ color: var(--bh-ink); }}
.bh-worker {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
  border-bottom: 1px solid var(--bh-rule); padding: .45rem 0; font-size: .92rem; }}
.bh-worker span.name {{ font-weight: 600; }}
.bh-worker span.meta {{ color: var(--bh-muted); font-size: .85rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def pill(status: str) -> str:
    word, colour = STATUS_STYLE.get(status, (status.replace("_", " ").capitalize(), MUTED))
    return f'<span class="bh-pill" style="color:{colour}">{e(word)}</span>'


def lede(text: str) -> None:
    st.markdown(f'<p class="bh-lede">{e(text)}</p>', unsafe_allow_html=True)


def not_legal_advice() -> None:
    st.markdown('<div class="bh-banner">Decision support for a lawyer — not legal advice.</div>',
                unsafe_allow_html=True)


def empty(title: str, body: str) -> None:
    st.markdown(f'<div class="bh-empty"><b>{e(title)}</b><br>{e(body)}</div>', unsafe_allow_html=True)


def kv_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(f"<tr><th>{e(k)}</th><td>{e(v) if v not in (None, '') else '—'}</td></tr>" for k, v in rows)
    return f'<table class="bh-kv">{body}</table>'


def element_name(name: str) -> str:
    return name.replace("_", " ").replace(" ge ", " ≥ ").replace("uk eea", "UK/EEA").capitalize()


def statute_strip(citation: str, summary: str, elements: dict[str, Any], order: list[str],
                  overlay: str | None = None) -> str:
    names = [n for n in order if n in elements] + [n for n in elements if n not in order]
    cells = []
    for n in names:
        el = elements.get(n) or {}
        glyph, word, colour = ELEMENT_STATUS.get(str(el.get("status")), ("?", "unknown", AMBER))
        cells.append(
            f'<div class="bh-el" style="--el:{colour}"><div class="bh-el-name"><span class="bh-el-glyph" '
            f'aria-hidden="true">{glyph}</span>{e(element_name(n))}</div><div class="bh-el-state">{e(word)}</div>'
            f'<div class="bh-el-ev">{e(el.get("evidence") or "")}</div></div>')
    grid = f'<div class="bh-elements">{"".join(cells)}</div>' if cells else ""
    ov = f'<div class="bh-overlay">{e(overlay)}</div>' if overlay else ""
    return (f'<section class="bh-statute" aria-label="Legal basis and evidence"><div class="bh-statute-cite">'
            f'{e(citation)}</div><div class="bh-statute-sum">{e(summary)}</div>{grid}{ov}</section>')


def diff_table(diff: dict[str, Any]) -> str:
    rows = "".join(
        f'<tr><th>{e(k.replace("_", " ").capitalize())}</th><td class="exp">{e(v.get("expected"))}</td>'
        f'<td class="act">{e(v.get("actual"))}</td></tr>' for k, v in diff.items())
    return (f'<table class="bh-diff"><thead><tr><th>Field</th><th>Approved case</th><th>Portal review page</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')


def money(v: float | None) -> str:
    return "—" if v is None else f"£{v:,.2f}"
