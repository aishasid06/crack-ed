"""Show a run's classified listings: category, rule, confidence and the evidence for each legal element.

    python -m bh.show_cases --run <run_id> [--json]

Read-only. For real-brand demo runs the banner is printed first.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.config import DEMO_BANNER
from bh.db import Case, Listing, Run, get_engine
from bh.decide import DISAGREE_LABEL
from bh.money import money


def cases_for_run(run_id: int, engine: Engine | None = None) -> dict[str, Any]:
    with Session(engine or get_engine()) as s:
        run = s.get(Run, run_id)
        if run is None:
            raise KeyError(f"run {run_id} not found")
        rows = []
        for c in sorted(s.exec(select(Case).where(Case.run_id == run_id)), key=lambda c: c.id or 0):
            l = s.get(Listing, c.listing_id)
            rows.append({
                "case_id": c.id, "listing": l.title if l else None, "url": l.url if l else None,
                "site": l.platform if l else None, "price": l.price if l else None,
                "currency": l.currency if l else None, "seller": l.seller_name if l else None,
                "category": str(c.category), "rule": c.rule_id, "confidence": round(c.confidence, 3),
                "status": str(c.status), "llm_used": c.llm_used,
                "model_disagrees": c.model_disagrees, "model_disagreement": c.model_disagreement,
                "elements": {k: {"status": (v or {}).get("status"), "evidence": (v or {}).get("evidence")}
                             for k, v in (c.legal_elements or {}).items()},
                "reasoning": c.reasoning,
            })
    demo = bool((run.config_snapshot or {}).get("real_brand_demo"))
    return {"run_id": run_id, "banner": DEMO_BANNER if demo else None, "cases": rows}


def _text(out: dict[str, Any]) -> str:
    lines = []
    if out["banner"]:
        lines += [f"*** {out['banner']} ***", ""]
    lines.append(f"Run {out['run_id']}: {len(out['cases'])} classified listing(s)")
    for c in out["cases"]:
        lines += ["", f"- {c['listing']}  ({c['site']}, {money(c['price'], c['currency'])})",
                  f"  {c['url']}",
                  f"  category: {c['category']}   rule: {c['rule']}   confidence: {c['confidence']}"
                  f"   status: {c['status']}{'   (vision check used)' if c['llm_used'] else ''}"]
        if c["model_disagrees"]:
            lines.append(f"  {DISAGREE_LABEL}: {c['model_disagreement']}")
        if c["elements"]:
            lines.append("  elements:")
            lines += [f"    {k}: {v['status']} — {v['evidence']}" for k, v in c["elements"].items()]
        else:
            lines.append("  elements: none (rule requires none)")
        if c["reasoning"]:
            lines.append(f"  reasoning: {c['reasoning']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        out = cases_for_run(args.run)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, default=str) if args.json else _text(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
