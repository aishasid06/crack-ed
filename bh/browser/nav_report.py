"""Navigator report: first visit (agent loop) vs recipe replay, from two traced runs.

    python -m bh.browser.nav_report <first_run_id> <replay_run_id> [--site bazaarly] [--out data/eval/...md]

Reads only the trace (Task spans and LLMCall rows). Steps are model decisions for the agent and code steps for
a replay. Cost is an estimate from token counts at the list prices below (USD per million tokens).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from bh.config import get_config
from bh.db import LLMCall, Task, get_engine

# USD per million input / output tokens (prompts ≤ 100k tokens). Estimate only; check current pricing.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (5.0, 25.0)}


def _secs(t: Task) -> float:
    return (t.finished_at - t.started_at).total_seconds() if t.finished_at and t.started_at else 0.0


def summarise(run_id: int, site: str = "bazaarly", engine: Any = None) -> dict[str, Any]:
    with Session(engine or get_engine()) as s:
        tasks = list(s.exec(select(Task).where(Task.run_id == run_id)))
        calls = list(s.exec(select(LLMCall).where(LLMCall.run_id == run_id)))
    nav = [t for t in tasks if t.site == site and t.type in ("agent:task", "recipe:task")]
    rows = []
    found_by_task: dict[str, int] = defaultdict(int)
    for t in tasks:
        if t.type == "agent:action" and t.site == site:
            parts = t.dedup_key.split(":")
            found_by_task[":".join(parts[1:4])] += int(t.payload.get("extracted") or 0)
    for t in sorted(nav, key=lambda t: t.id or 0):
        p = t.payload
        listings = p.get("listings_found")
        if listings is None and p.get("task") == "search":
            listings = sum(v for k, v in found_by_task.items() if k.startswith("search:"))
        rows.append({"type": t.type, "task": p.get("task"), "subject": p.get("subject"), "mode": p.get("mode"),
                     "recipe_version": p.get("recipe_version"), "status": str(t.status),
                     "success": p.get("success"), "reason": p.get("reason") or t.error, "steps": p.get("steps") or 0,
                     "tokens": (p.get("input_tokens") or 0) + (p.get("output_tokens") or 0),
                     "seconds": round(_secs(t), 1), "listings_found": listings if p.get("task") == "search" else None})
    cost = 0.0
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    for c in calls:
        m = by_model[c.model]
        m["calls"] += 1
        m["in"] += c.input_tokens
        m["out"] += c.output_tokens
        pin, pout = PRICES.get(c.model, (0.0, 0.0))
        cost += c.input_tokens / 1e6 * pin + c.output_tokens / 1e6 * pout
    search = [r for r in rows if r["task"] == "search"]
    listing = [r for r in rows if r["task"] == "listing"]

    def agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        # finished = the task completed (data returned); the model's own "success" flag can be false when a
        # page simply has no seller details, so it is not used here
        return {"tasks": len(rs), "succeeded": sum(r["status"] == "succeeded" for r in rs),
                "modes": sorted({str(r["mode"]) for r in rs}), "steps": sum(r["steps"] for r in rs),
                "tokens": sum(r["tokens"] for r in rs), "seconds": round(sum(r["seconds"] for r in rs), 1)}

    return {"run_id": run_id, "site": site, "search": agg(search), "listing": agg(listing),
            "listings_found": sum(r["listings_found"] or 0 for r in search),
            "llm_calls": sum(m["calls"] for m in by_model.values()), "llm_by_model": dict(by_model),
            "tokens": sum(m["in"] + m["out"] for m in by_model.values()), "cost_usd": round(cost, 5),
            "recipe_versions": sorted({r["recipe_version"] for r in rows if r["recipe_version"] is not None}),
            "failures": [r for r in rows if r["status"] != "succeeded"], "tasks": rows}


def markdown(a: dict[str, Any], b: dict[str, Any]) -> str:
    def line(label: str, x: Any, y: Any) -> str:
        return f"| {label} | {x} | {y} |"

    out = [f"# Navigator on {a['site']}: first visit vs recipe replay", "",
           f"First visit: run {a['run_id']} (agent loop, recipe learned). Replay: run {b['run_id']}.", "",
           "| | First visit | Replay |", "|---|---|---|",
           line("Modes", ", ".join(a["search"]["modes"] + a["listing"]["modes"]),
                ", ".join(b["search"]["modes"] + b["listing"]["modes"])),
           line("Recipe version", a["recipe_versions"] or "-", b["recipe_versions"] or "-"),
           line("Listings found (search)", a["listings_found"], b["listings_found"]),
           line("Search: steps", a["search"]["steps"], b["search"]["steps"]),
           line("Search: seconds", a["search"]["seconds"], b["search"]["seconds"]),
           line("Listings read (ok / total)", f"{a['listing']['succeeded']} / {a['listing']['tasks']}",
                f"{b['listing']['succeeded']} / {b['listing']['tasks']}"),
           line("Listings: steps", a["listing"]["steps"], b["listing"]["steps"]),
           line("Listings: seconds", a["listing"]["seconds"], b["listing"]["seconds"]),
           line("LLM calls", a["llm_calls"], b["llm_calls"]),
           line("Tokens", f"{a['tokens']:,}", f"{b['tokens']:,}"),
           line("Est. cost (USD)", f"{a['cost_usd']:.4f}", f"{b['cost_usd']:.4f}"),
           line("Failures", len(a["failures"]), len(b["failures"])), "",
           "Steps: model decisions (agent) or code steps (replay). Seconds: sum of task durations. Cost: token "
           "counts at list prices (estimate).", ""]
    for name, r in (("First visit", a), ("Replay", b)):
        if r["failures"]:
            out.append(f"{name} failures:")
            out += [f"- {f['type']} {f['subject']}: {f['reason']}" for f in r["failures"]]
            out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first_run", type=int)
    ap.add_argument("replay_run", type=int)
    ap.add_argument("--site", default="bazaarly")
    ap.add_argument("--out", type=Path, default=None, help="markdown file (default data/eval/navigator_<a>_vs_<b>.md)")
    args = ap.parse_args()
    a, b = summarise(args.first_run, args.site), summarise(args.replay_run, args.site)
    cfg = get_config()
    out = args.out or cfg.paths.resolve(cfg.paths.data_dir) / "eval" / f"navigator_run{a['run_id']}_vs_run{b['run_id']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(a, b), encoding="utf-8")
    print(json.dumps({"first_visit": {k: v for k, v in a.items() if k != "tasks"},
                      "replay": {k: v for k, v in b.items() if k != "tasks"}, "report": str(out)}, indent=2,
                     default=str))


if __name__ == "__main__":
    main()
