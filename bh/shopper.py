"""Shopper search: "I want to buy this product — which listings are genuine?"

A shopper gives a product name and/or a photo. The agent:

    load the brand's official products → (photo) work out which product it shows →
    plan a few searches → search each marketplace (and search by photo where a site supports it) →
    keep the results that look like the brand's products → open them and capture the details →
    check each listing against the legal rules (same rules as enforcement)

Every step is a traced task, so the shopper UI can show what the agent is doing and which site it is on.
Searches run in their own database (``config.shopper.db_url``) so they never mix with the enforcement
pipeline's runs, cases and review queue.

With an API key (``shopper.use_llm``: auto), the same LLM features as the pipeline's ``--llm`` apply: the
generic extractor fallback, the borderline vision check, and the navigator on adapter-less sites (which learns a
recipe). Without a key, adapter-less sites with a learned recipe are still searched by replaying it (no tokens).

    python -m bh.shopper --search-id ID [--text "product name"] [--photo path/to/photo.jpg]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bh.config import AppConfig, get_config
from bh.discovery.queries import Query, dedupe

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhotoMatch:
    product_id: int
    product_name: str
    similarity: float


def identify_product(ctx: Any, vec: Any, min_similarity: float) -> PhotoMatch | None:
    """The official product a shopper's photo most resembles, if it is close enough to say so."""
    sim, pid = ctx.nearest_clip(vec)
    product = ctx.product(pid)
    if product is None or sim < min_similarity:
        return None
    return PhotoMatch(product.id, product.name, round(sim, 3))


def shopper_queries(text: str | None, match: PhotoMatch | None, ctx: Any, max_queries: int) -> list[Query]:
    """The shopper's own words first, then the product their photo shows, then the brand's mark."""
    qs: list[Query] = []
    if text and text.strip():
        qs.append(Query(text.strip()[:60], "shopper"))
    if match is not None:
        product = ctx.product(match.product_id)
        mark = (ctx.marks or [ctx.name])[0]
        short = product.short_name if product else match.product_name
        qs.append(Query(f"{mark} {short}", "product"))
    qs.append(Query((ctx.marks or [ctx.name])[0], "mark"))
    return dedupe(qs)[:max_queries]


def search_config(cfg: AppConfig, with_navigator: bool = False) -> AppConfig:
    """Shopper searches page less deeply and open fewer listings than an enforcement run."""
    sh = cfg.shopper
    sites = {k: v.model_copy(update={"max_pages": min(v.max_pages, sh.max_pages)}) for k, v in cfg.sites.items()}
    budgets = cfg.budgets.model_copy(update={"max_listings_per_run": sh.max_listings})
    out = cfg.model_copy(update={"sites": sites, "budgets": budgets})
    if with_navigator:
        out = out.model_copy(update={"navigator": out.navigator.model_copy(update={"enabled": True})})
    return out


def make_llm(cfg: AppConfig) -> Any | None:
    """The LLM client for shopper searches, per ``shopper.use_llm`` (auto: only when an API key is set)."""
    from bh.config import get_secrets

    mode = cfg.shopper.use_llm
    if mode == "never" or (mode == "auto" and get_secrets().anthropic_api_key is None):
        return None
    from bh.llm import LLMClient

    return LLMClient(cfg)


def search_sites(cfg: AppConfig, llm: Any | None) -> list[str]:
    """Marketplaces with an adapter, plus adapter-less sites the navigator can handle: with an LLM it can
    browse any of them; without one, only sites with a learned search recipe (replayed with no tokens)."""
    from bh.discovery.worker import marketplace_sites

    sites = marketplace_sites(cfg)
    if not cfg.shopper.navigator_sites:
        return sites
    # real (read_only) marketplaces are never searched from the shopper app: only by name, from the command line
    navigable = [s for s, c in cfg.sites.items() if c.navigator and c.mode != "read_only" and s not in sites]
    if llm is not None:
        return sites + navigable
    from bh.browser.recipes import RecipeStore

    store = RecipeStore(cfg)
    return sites + [s for s in navigable if (r := store.load(s)) is not None and r.search is not None]


BUDGET_REASON = "over max_listings_per_run"


def pick_fairly(scored: list[Any], limit: int) -> list[Any]:
    """Which matches to open: every site gets its turn, best match first, until ``limit``. A shopper wants
    results from every marketplace, not only the largest one (the prefilter's own cut is by score alone)."""
    eligible = [s for s in scored if s.keep or BUDGET_REASON in s.reasons]
    by_site: dict[str, list[Any]] = {}
    for s in sorted(eligible, key=lambda s: -s.priority):
        by_site.setdefault(s.card.site, []).append(s)
    picked: list[Any] = []
    while len(picked) < limit and any(by_site.values()):
        for site in sorted(by_site):
            if by_site[site] and len(picked) < limit:
                picked.append(by_site[site].pop(0))
    chosen = {id(s) for s in picked}
    for s in eligible:
        s.keep = id(s) in chosen
    return picked


class RulesOnlyScorer:
    """Used when the calibrated scorer hasn't been fitted: shoppers see the rule outcome, not a score."""

    meta: dict[str, Any] = {"rules_only": True}

    def predict(self, feats: list[Any], counterfeit_price_ratio: float) -> list[float]:
        return [0.5] * len(feats)


def _scorer(cfg: AppConfig) -> Any:
    from bh.classify import Scorer

    path = cfg.paths.resolve(cfg.models.scorer_path)
    return Scorer(path) if path.exists() else RulesOnlyScorer()


async def photo_search(run_id: int, browser: Any, prefilter: Any, photo: Path, sites: list[str]) -> int:
    """Search by photo on every site whose adapter supports it. Returns the number of results seen."""
    from bh.trace import task_span

    seen = 0
    for site in sites:
        adapter = browser.adapter(site)
        if adapter is None or not hasattr(adapter, "visual_search"):
            continue
        try:
            async with task_span(run_id, "photo_search", site, f"photo_search:{site}") as t:
                async with browser.page(site) as page:
                    cards = await adapter.visual_search(page, photo)
                scored = await prefilter.score(cards, "photo")
                t.payload = {"cards": len(cards), "passing": sum(s.keep for s in scored)}
                seen += len(cards)
        except Exception as e:  # recorded on the task; text search results still count
            log.warning("photo search on %s failed: %s", site, e)
    return seen


_AUTO: Any = object()


async def run_search(search_id: str, text: str | None, photo: Path | None, cfg: AppConfig | None = None,
                     encoder: Any | None = None, browser: Any | None = None, llm: Any = _AUTO) -> int:
    import httpx

    from bh.brand_index import load_brand
    from bh.browser.session import BrowserManager
    from bh.classify import classify_listings
    from bh.discovery.worker import Prefilter, capture_all, fan_out_discovery, finalize_prefilter
    from bh.trace import run_span, task_span
    from bh.vision import ClipEncoder, open_rgb

    base = cfg or get_config()
    llm = make_llm(base) if llm is _AUTO else llm
    scfg = search_config(base, with_navigator=llm is not None and base.shopper.navigator_sites)
    encoder = encoder or ClipEncoder(scfg.vision.clip_model)
    sites = search_sites(scfg, llm)
    snapshot = {"stage": "shopper", "search_id": search_id, "query": text, "has_photo": photo is not None,
                "llm": llm is not None, "sites": sites}
    async with run_span(None, snapshot) as run:
        rid = run.id
        async with task_span(rid, "load_brand", None, "load_brand"):
            ctx = await load_brand(scfg, encoder)
        match = None
        if photo is not None:
            async with task_span(rid, "identify_photo", None, "identify_photo") as t:
                vec = (await asyncio.to_thread(encoder.encode, [open_rgb(photo)]))[0]
                match = identify_product(ctx, vec, scfg.shopper.photo_match_min)
                t.payload = {"product": match.product_name if match else None,
                             "similarity": match.similarity if match else None}
        queries = shopper_queries(text, match, ctx, scfg.shopper.max_queries)
        async with task_span(rid, "plan_search", None, "plan_search") as t:
            t.payload = {"queries": [q.text for q in queries], "brand": ctx.name, "sites": sites}

        async def work(b: Any) -> list[int]:
            async with httpx.AsyncClient(timeout=20) as client:
                prefilter = Prefilter(rid, ctx, scfg, encoder, client)
                await fan_out_discovery(rid, sites, queries, b, prefilter, llm)
                if photo is not None:
                    await photo_search(rid, b, prefilter, photo, sites)
                scored = await finalize_prefilter(rid, prefilter)
            kept = pick_fairly(scored, scfg.shopper.max_listings)
            async with task_span(rid, "capture_all", None, "capture_all", payload={"n": len(kept)}):
                ids, _, _ = await capture_all(rid, kept, b, llm, scfg)
            return ids

        if browser is not None:
            ids = await work(browser)
        else:
            async with BrowserManager(scfg) as b:
                ids = await work(b)
        await classify_listings(rid, ids, scfg, encoder, llm, scorer=_scorer(scfg))  # llm: borderline vision check
    return rid


def use_shopper_db(cfg: AppConfig) -> None:
    """Point every DB write in this process (runs, tasks, listings, cases) at the shopper database."""
    from bh.db import init_db, make_engine, set_engine

    url = cfg.shopper.db_url
    prefix = "sqlite:///"
    if url.startswith(prefix) and not Path(url[len(prefix):]).is_absolute():
        url = prefix + str(cfg.paths.resolve(url[len(prefix):]))
    engine = make_engine(url)
    init_db(engine)
    set_engine(engine)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-id", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--photo", type=Path, default=None)
    args = ap.parse_args()
    if not args.text and not args.photo:
        ap.error("give --text, --photo or both")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = get_config()
    use_shopper_db(cfg)
    code = 0
    try:
        rid = asyncio.run(run_search(args.search_id, args.text, args.photo, cfg))
        print(f"[bh] search {args.search_id} finished as run {rid}", flush=True)
    except Exception:
        log.exception("search %s failed", args.search_id)
        code = 1
    print(f"[bh] exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
