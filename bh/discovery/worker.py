"""Discovery pipeline (SPEC §7.1 nodes 1-5, Phase 5):

    load_brand → generate_queries → fan-out search per (site, query) → dedup by URL →
    thumbnail prefilter (CLIP / pHash / brand fuzzy match) → parallel capture
    (full listing, screenshot, images) → Listing rows

Each stage is an async function the LangGraph orchestrator (Phase 8) calls as a node; ``run_discovery``
chains them for the CLI. Every stage, search, capture, download and browser step is a trace span.

    python -m bh.discovery.worker [--sites echobazaar tradeloft] [--max-queries N] [--llm]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from sqlalchemy.engine import Engine
from sqlmodel import select

from bh.brand_index import BrandContext, load_brand
from bh.browser.adapters.base import Card, ListingData
from bh.browser.ops import capture_listing, iter_search_pages
from bh.browser.session import BrowserManager
from bh.config import AppConfig, get_config
from bh.db import Listing, get_engine, session_scope, utcnow
from bh.discovery.matching import BrandMatcher
from bh.discovery.queries import Query, generate_queries
from bh.errors import HumanRequired
from bh.rules import get_rules
from bh.trace import run_span, task_span
from bh.vision import ClipEncoder, ImageEncoder, open_rgb

log = logging.getLogger(__name__)
_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


@dataclass
class Found:
    card: Card
    queries: list[str]


@dataclass
class Scored:
    card: Card
    queries: list[str]
    clip_sim: float | None = None
    hash_dist: int | None = None
    brand_score: float = 0.0
    matched_product_id: int | None = None
    thumb_path: Path | None = None
    keep: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def priority(self) -> float:
        return max(self.clip_sim or 0.0, 1.0 - (self.hash_dist or 64) / 64.0, self.brand_score / 100.0)

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe record of a kept card, stored on its discovery task so a resumed run can restore it."""
        return {"card": self.card.model_dump(mode="json"), "queries": list(self.queries), "clip_sim": self.clip_sim,
                "hash_dist": self.hash_dist, "brand_score": self.brand_score,
                "matched_product_id": self.matched_product_id, "keep": self.keep, "reasons": list(self.reasons)}

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> "Scored":
        return cls(Card.model_validate(d["card"]), list(d.get("queries", [])), d.get("clip_sim"), d.get("hash_dist"),
                   d.get("brand_score", 0.0), d.get("matched_product_id"), keep=bool(d.get("keep")),
                   reasons=list(d.get("reasons", [])))


@dataclass
class DiscoveryResult:
    run_id: int
    queries: list[str]
    cards_found: int
    unique_cards: int
    kept: int
    captured: int
    failed: int
    waiting_human: int
    listing_ids: list[int]
    scored: list[Scored] = field(repr=False, default_factory=list)
    pages_read: int = 0


def navigator_active(cfg: AppConfig, llm: Any | None) -> bool:
    """Navigator sites are opt-in (navigator.enabled / --navigator) and need an LLM."""
    return cfg.navigator.enabled and llm is not None


def enable_navigator(cfg: AppConfig) -> AppConfig:
    return cfg.model_copy(update={"navigator": cfg.navigator.model_copy(update={"enabled": True})})


def marketplace_sites(cfg: AppConfig, only: list[str] | None = None, with_navigator: bool = False) -> list[str]:
    """Searchable marketplaces. Navigator sites are included only with ``with_navigator`` (see navigator_active)."""
    sites = [s for s, c in cfg.sites.items() if c.adapter or c.search_url or (c.navigator and with_navigator)]
    return [s for s in sites if not only or s in only]


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:60] or "x"


def listing_dir(cfg: AppConfig, site: str, data: ListingData) -> Path:
    key = data.external_id or hashlib.sha1(data.url.encode()).hexdigest()[:16]
    return cfg.paths.resolve(cfg.paths.data_dir) / "listings" / _safe_id(site) / _safe_id(key)


# ---------------------------------------------------------------- downloads


async def download_image(client: httpx.AsyncClient, url: str, dest_stem: Path, max_bytes: int) -> Path | None:
    if urlparse(url).scheme not in ("http", "https"):
        return None
    try:
        async with client.stream("GET", url) as r:
            if r.status_code != 200:
                return None
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            ext = _IMAGE_TYPES.get(ctype)
            if ext is None:
                return None
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf += chunk
                if len(buf) > max_bytes:
                    return None
    except httpx.HTTPError as e:
        log.warning("image download failed %s: %s", url, e)
        return None
    dest = dest_stem.with_suffix(ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(dest.write_bytes, bytes(buf))
    return dest


# ---------------------------------------------------------------- prefilter (scores result cards page by page)


class Prefilter:
    """Scores search-result cards against the gold assets as pages arrive. Thumbnails are downloaded
    and encoded once per URL; the keep rule is: CLIP ≥ prefilter_clip OR pHash ≤ prefilter_hash OR
    brand fuzzy score ≥ prefilter_brand_fuzzy (or no thumbnail at all → keep, cannot judge)."""

    def __init__(self, run_id: int, ctx: BrandContext, cfg: AppConfig, encoder: ImageEncoder,
                 client: httpx.AsyncClient) -> None:
        self.run_id, self.ctx, self.cfg, self.encoder, self.client = run_id, ctx, cfg, encoder, client
        self.cache: dict[str, Scored] = {}
        self.sem = asyncio.Semaphore(cfg.discovery.thumb_concurrency)
        self.matcher = BrandMatcher([*ctx.marks, ctx.name])
        self.thumbs_dir = cfg.paths.resolve(cfg.paths.data_dir) / "runs" / str(run_id) / "thumbs"
        self._batch = 0

    async def score(self, cards: list[Card], query: str) -> list[Scored]:
        new = [Scored(c, []) for c in {c.url: c for c in cards if c.url not in self.cache}.values()]
        for s in new:
            self.cache[s.card.url] = s
        if new:
            await self._score_new(new)
        out = []
        for c in cards:
            s = self.cache[c.url]
            if query not in s.queries:
                s.queries.append(query)
            out.append(s)
        return out

    async def _score_new(self, new: list[Scored]) -> None:
        max_bytes = int(self.cfg.discovery.max_image_mb * 1024 * 1024)
        self._batch += 1

        async def fetch(s: Scored) -> None:
            if s.card.thumbnail_url:
                async with self.sem:
                    s.thumb_path = await download_image(
                        self.client, s.card.thumbnail_url,
                        self.thumbs_dir / hashlib.sha1(s.card.url.encode()).hexdigest(), max_bytes)

        async with task_span(self.run_id, "download:thumbnails", None, f"thumbs:{self._batch}:{id(new)}",
                             payload={"n": len(new)}):
            await asyncio.gather(*(fetch(s) for s in new))
        await asyncio.to_thread(self._apply, new)

    def _apply(self, new: list[Scored]) -> None:
        th = self.cfg.thresholds
        have = [s for s in new if s.thumb_path]
        imgs = [open_rgb(s.thumb_path) for s in have]  # type: ignore[arg-type]
        vecs = self.encoder.encode(imgs) if imgs else []
        for s in new:
            s.brand_score = self.matcher.score(s.card.title)
        for s, im, v in zip(have, imgs, vecs, strict=True):
            s.hash_dist, hp = self.ctx.nearest_hash(im)
            s.clip_sim, cp = self.ctx.nearest_clip(v)
            s.matched_product_id = cp if s.clip_sim >= th.prefilter_clip else hp
        for s in new:
            if s.clip_sim is not None and s.clip_sim >= th.prefilter_clip:
                s.reasons.append(f"clip {s.clip_sim:.2f}")
            if s.hash_dist is not None and s.hash_dist <= th.prefilter_hash:
                s.reasons.append(f"phash {s.hash_dist}")
            if s.brand_score >= th.prefilter_brand_fuzzy:
                s.reasons.append(f"brand text {s.brand_score:.0f}")
            if s.thumb_path is None and s.card.thumbnail_url is None:
                s.reasons.append("no thumbnail")
            s.keep = bool(s.reasons)

    def kept_for(self, site: str, query: str) -> list[Scored]:
        """Prefilter-passing cards this (site, query) search saw."""
        return [s for s in self.cache.values() if s.keep and s.card.site == site and query in s.queries]

    def restore(self, items: list[dict[str, Any]]) -> None:
        """Re-add kept cards recorded by an already-finished discovery task (resumed runs)."""
        for d in items:
            s = Scored.from_payload(d)
            have = self.cache.get(s.card.url)
            if have is None:
                self.cache[s.card.url] = s
            else:
                have.queries.extend(q for q in s.queries if q not in have.queries)

    def finalize(self) -> list[Scored]:
        """All scored cards; kept ones beyond the listing budget are dropped (highest priority first)."""
        scored = list(self.cache.values())
        kept = sorted((s for s in scored if s.keep), key=lambda s: -s.priority)
        for s in kept[self.cfg.budgets.max_listings_per_run:]:
            s.keep = False
            s.reasons.append("over max_listings_per_run")
        return scored


# ---------------------------------------------------------------- stage: adaptive search fan-out


async def search_adaptive(browser: BrowserManager, site: str, query: str, prefilter: Prefilter,
                          llm: Any | None) -> dict[str, Any]:
    """Page through results until ``stop_after_empty_pages`` consecutive pages contain no
    prefilter-passing card, or the site's hard ``max_pages`` cap is reached."""
    cap = browser.cfg.site(site).max_pages
    patience = browser.cfg.discovery.stop_after_empty_pages
    pages = empty = cards = passing = 0
    stop = "exhausted"
    async with aclosing(iter_search_pages(browser, site, query, cap, llm)) as it:
        async for batch in it:
            pages += 1
            scored = await prefilter.score(batch, query)
            cards += len(batch)
            hits = sum(s.keep for s in scored)
            passing += hits
            empty = 0 if hits else empty + 1
            if empty >= patience:
                stop = "no_relevant_results"
                break
            if pages >= cap:
                stop = "max_pages"
                break
    return {"pages": pages, "cards": cards, "passing": passing, "stop": stop}


async def fan_out_discovery(run_id: int, sites: list[str], queries: list[Query], browser: BrowserManager,
                            prefilter: Prefilter, llm: Any | None) -> tuple[int, int, int]:
    """One traced task per (site, query), at most ``max_concurrency`` running per site. The slot is taken
    before the span opens, so task timings and run parallelism reflect real work, not queueing.
    Returns (cards seen, pages read, tasks waiting for a human)."""
    sems = {s: asyncio.Semaphore(browser.cfg.site(s).max_concurrency) for s in sites}

    async def one(site: str, q: Query) -> dict[str, Any]:
        async with sems[site], task_span(run_id, "discovery", site, f"discovery:{site}:{q.text}",
                                         payload={"query": q.text, "kind": q.kind}) as t:
            if t.already_done:
                return {"pages": 0, "cards": 0}
            stats = await search_adaptive(browser, site, q.text, prefilter, llm)
            t.payload = {**t.payload, **stats}
            return stats

    results = await asyncio.gather(*(one(s, q) for s in sites for q in queries), return_exceptions=True)
    cards = pages = waiting = 0
    for r in results:
        if isinstance(r, HumanRequired):
            waiting += 1
        elif isinstance(r, BaseException):
            log.warning("discovery task failed: %s", r)  # already recorded on the task
        else:
            cards += r["cards"]
            pages += r["pages"]
    return cards, pages, waiting


def dedup(found: list[tuple[Card, str]]) -> list[Found]:
    by_url: dict[str, Found] = {}
    for card, q in found:
        f = by_url.get(card.url)
        if f is None:
            by_url[card.url] = Found(card, [q])
        elif q not in f.queries:
            f.queries.append(q)
    return list(by_url.values())


async def finalize_prefilter(run_id: int, prefilter: Prefilter) -> list[Scored]:
    async with task_span(run_id, "prefilter", None, "prefilter") as t:
        scored = prefilter.finalize()
        t.payload = {"unique": len(scored), "kept": sum(s.keep for s in scored)}
    return scored


async def dedup_and_prefilter(run_id: int, found: list[tuple[Card, str]], ctx: BrandContext, cfg: AppConfig,
                              encoder: ImageEncoder) -> list[Scored]:
    """Score an already-collected card list (used outside adaptive search, e.g. tests and re-scoring)."""
    async with httpx.AsyncClient(timeout=20) as client:
        pf = Prefilter(run_id, ctx, cfg, encoder, client)
        for f in dedup(found):
            for q in f.queries:
                await pf.score([f.card], q)
        return await finalize_prefilter(run_id, pf)


# ---------------------------------------------------------------- stage: capture


def upsert_listing(engine: Engine, data: ListingData, run_id: int, query: str | None, screenshot: Path | None,
                   images: list[Path]) -> int:
    with session_scope(engine) as s:
        row = s.exec(select(Listing).where(Listing.url == data.url)).first()
        if row is None:
            row = Listing(platform=data.site, url=data.url, run_id=run_id, first_seen_at=utcnow(),
                          discovered_by_query=query)
            s.add(row)
        row.external_id = data.external_id
        row.title, row.description = data.title, data.description
        row.price, row.currency = data.price, data.currency
        row.seller_name, row.seller_location = data.seller_name, data.seller_location
        row.seller_age_days, row.rating, row.quantity = data.seller_age_days, data.rating, data.quantity
        row.image_urls = list(data.image_urls)
        row.local_image_paths = [str(p) for p in images]
        row.screenshot_path = str(screenshot) if screenshot else None
        row.extraction_method = data.extraction_method
        s.flush()
        assert row.id is not None
        return row.id


async def capture_one(run_id: int, s: Scored, browser: BrowserManager, llm: Any | None, cfg: AppConfig,
                      client: httpx.AsyncClient, img_sem: asyncio.Semaphore, engine: Engine,
                      site_sem: asyncio.Semaphore) -> int | None:
    site = s.card.site
    async with site_sem, task_span(run_id, "capture", site, f"capture:{s.card.url}",
                                   payload={"url": s.card.url, "queries": s.queries, "prefilter": s.reasons}) as t:
        if t.already_done:  # captured earlier in this run (resume): report the stored listing
            lid = t.payload.get("listing_id")
            return int(lid) if lid is not None else None
        shot_tmp = cfg.paths.resolve(cfg.paths.data_dir) / "runs" / str(run_id) / "shots" / \
            f"{hashlib.sha1(s.card.url.encode()).hexdigest()}.png"
        data = await capture_listing(browser, site, s.card.url, llm, screenshot_path=shot_tmp)
        folder = listing_dir(cfg, site, data)
        folder.mkdir(parents=True, exist_ok=True)
        shot = folder / "screenshot.png"
        await asyncio.to_thread(shot_tmp.replace, shot)
        urls = data.image_urls[: cfg.discovery.max_images_per_listing]
        max_bytes = int(cfg.discovery.max_image_mb * 1024 * 1024)
        async with task_span(run_id, "download:images", site, f"images:{s.card.url}", payload={"n": len(urls)}):
            async def fetch(i: int, u: str) -> Path | None:
                async with img_sem:
                    return await download_image(client, u, folder / f"img_{i + 1}", max_bytes)

            paths = [p for p in await asyncio.gather(*(fetch(i, u) for i, u in enumerate(urls))) if p]
        listing_id = await asyncio.to_thread(upsert_listing, engine, data, run_id,
                                             s.queries[0] if s.queries else None, shot, paths)
        t.payload = {**t.payload, "listing_id": listing_id, "images": len(paths),
                     "extraction_method": str(data.extraction_method)}
        return listing_id


async def capture_all(run_id: int, kept: list[Scored], browser: BrowserManager, llm: Any | None, cfg: AppConfig,
                      engine: Engine | None = None) -> tuple[list[int], int, int]:
    eng = engine or get_engine()
    img_sem = asyncio.Semaphore(cfg.discovery.image_concurrency)
    site_sems = {site: asyncio.Semaphore(cfg.site(site).max_concurrency) for site in {s.card.site for s in kept}}
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(*(capture_one(run_id, s, browser, llm, cfg, client, img_sem, eng,
                                                     site_sems[s.card.site]) for s in kept),
                                       return_exceptions=True)
    ids = [r for r in results if isinstance(r, int)]
    waiting = sum(isinstance(r, HumanRequired) for r in results)
    failed = sum(isinstance(r, BaseException) and not isinstance(r, HumanRequired) for r in results)
    for r in results:
        if isinstance(r, BaseException) and not isinstance(r, HumanRequired):
            log.warning("capture failed: %s", r)
    return ids, failed, waiting


# ---------------------------------------------------------------- end to end


async def discover(run_id: int, cfg: AppConfig, browser: BrowserManager, encoder: ImageEncoder,
                   llm: Any | None = None, sites: list[str] | None = None,
                   engine: Engine | None = None) -> DiscoveryResult:
    async with task_span(run_id, "load_brand", None, "load_brand"):
        ctx = await load_brand(cfg, encoder, engine)
    async with task_span(run_id, "generate_queries", None, "generate_queries") as t:
        queries = await generate_queries(ctx, cfg, get_rules().vocabulary, llm)
        t.payload = {"queries": [q.text for q in queries]}
    site_list = marketplace_sites(cfg, sites, with_navigator=navigator_active(cfg, llm))
    async with httpx.AsyncClient(timeout=20) as client:
        prefilter = Prefilter(run_id, ctx, cfg, encoder, client)
        cards_seen, pages, waiting = await fan_out_discovery(run_id, site_list, queries, browser, prefilter, llm)
        scored = await finalize_prefilter(run_id, prefilter)
    kept = [s for s in scored if s.keep]
    async with task_span(run_id, "capture_all", None, "capture_all", payload={"n": len(kept)}):
        ids, failed, waiting2 = await capture_all(run_id, kept, browser, llm, cfg, engine)
    return DiscoveryResult(run_id, [q.text for q in queries], cards_seen, len(scored), len(kept), len(ids), failed,
                           waiting + waiting2, ids, scored, pages)


async def run_discovery(cfg: AppConfig | None = None, encoder: ImageEncoder | None = None, llm: Any | None = None,
                        sites: list[str] | None = None, browser: BrowserManager | None = None,
                        engine: Engine | None = None) -> DiscoveryResult:
    cfg = cfg or get_config()
    encoder = encoder or ClipEncoder(cfg.vision.clip_model)
    async with run_span(None, cfg.snapshot(), engine=engine) as run:
        if browser is not None:
            return await discover(run.id, cfg, browser, encoder, llm, sites, engine)
        async with BrowserManager(cfg) as b:
            return await discover(run.id, cfg, b, encoder, llm, sites, engine)


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--llm", action="store_true", help="enable the generic LLM fallback / query extension")
    ap.add_argument("--navigator", action="store_true",
                    help="also search navigator sites (e.g. bazaarly) with the agentic navigator; needs --llm")
    args = ap.parse_args()
    if args.navigator:
        if not args.llm:
            ap.error("--navigator needs --llm (the agentic navigator is driven by the small model)")
        cfg = enable_navigator(cfg)
    if args.max_queries:
        cfg = cfg.model_copy(update={"discovery": cfg.discovery.model_copy(update={"max_queries": args.max_queries})})
    llm = None
    if args.llm:
        from bh.llm import LLMClient

        llm = LLMClient(cfg)
    res = asyncio.run(run_discovery(cfg, llm=llm, sites=args.sites))
    print(json.dumps({k: v for k, v in res.__dict__.items() if k not in ("scored", "listing_ids")}, indent=2))


if __name__ == "__main__":
    main()
