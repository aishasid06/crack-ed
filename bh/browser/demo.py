"""Phase 4 check: search each marketplace and parse cards and a few listings.

    python -m bh.browser.demo [--query Q] [--sites mockbay tradeloft] [--pages 2] [--listings 3]

The default query is the configured brand slug. Runs inside a traced run so every navigation
is recorded. Requires the mock sites to be running (``make mock``).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from bh.browser.ops import capture_listing, search_site
from bh.browser.session import BrowserManager
from bh.config import get_config
from bh.trace import run_span, task_span


async def main_async(query: str, sites: list[str], pages: int, listings: int, use_llm: bool) -> dict[str, object]:
    cfg = get_config()
    llm = None
    if use_llm:
        from bh.llm import LLMClient

        llm = LLMClient(cfg)
    out: dict[str, object] = {}
    async with run_span(None, {"demo": "phase4", "query": query}) as run, BrowserManager(cfg) as browser:
        for site in sites:
            async with task_span(run.id, "discovery", site, f"demo:{site}:{query}"):
                cards = await search_site(browser, site, query, pages, llm)
            parsed = []
            for c in cards[:listings]:
                async with task_span(run.id, "capture", site, f"capture:{c.url}"):
                    parsed.append((await capture_listing(browser, site, c.url, llm)).model_dump(mode="json"))
            out[site] = {"cards": len(cards), "sample_cards": [c.model_dump(mode="json") for c in cards[:3]],
                         "listings": parsed, "max_parallel_pages": browser.max_active[site]}
        out["run_id"] = run.id
    return out


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=cfg.brand)
    ap.add_argument("--sites", nargs="+", default=[s for s, c in cfg.sites.items() if c.adapter])
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--listings", type=int, default=2)
    ap.add_argument("--llm", action="store_true", help="allow the generic LLM fallback")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(main_async(args.query, args.sites, args.pages, args.listings, args.llm)), indent=2))


if __name__ == "__main__":
    main()
