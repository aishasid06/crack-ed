"""LangGraph pipeline (SPEC §7.1, Phase 8).

    load_brand → generate_queries ─Send per (site, query)→ discover ─→ dedup_and_prefilter
        ─Send per kept card→ capture ─→ classify → decide → draft → report

Graph state holds only JSON-safe values (ids, query strings, per-task summaries). Live, run-scoped resources
(browser, CLIP encoder, LLM client, HTTP client, prefilter cache, per-site semaphores) live in
``PipelineDeps``, which LangGraph hands every node as ``runtime.context``.

Each node reuses the stage functions from discovery / classify / decide / enforce, so every unit of work is
a traced Task. The per-site semaphores are taken *before* a task's span opens, so a queued search or capture
never looks like a running one.

Resume: re-running the graph with the same run id (``run_pipeline(resume_run_id=...)``) skips tasks that
already succeeded. Discovery tasks store the cards they kept and capture tasks store their listing id, so a
resumed run rebuilds its inputs without searching again. Classification is skipped when every captured
listing already has a case in the run. Decide and draft are idempotent and always run, which is how a
draft paused by a portal challenge (``waiting_human``) gets retried.

The orchestrator never approves or submits anything. Submission stays in ``bh.enforce.submit``, a separate
loop that acts only on cases a lawyer approved.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.brand_index import BrandContext, load_brand
from bh.browser.session import BrowserManager
from bh.classify import Scorer, classify_listings
from bh.config import AppConfig
from bh.db import Case, Run, session_scope
from bh.decide import decide_cases
from bh.discovery.queries import Query, generate_queries
from bh.discovery.worker import (Prefilter, capture_one, finalize_prefilter, marketplace_sites, navigator_active,
                                 search_adaptive)
from bh.enforce.portal import draft_cases
from bh.errors import HumanRequired
from bh.rules import get_rules
from bh.trace import compute_run_totals, task_span, type_name
from bh.security import redact
from bh.vision import ImageEncoder

log = logging.getLogger(__name__)

Stage = Literal["discovery", "classify", "decide", "draft"]
STAGES: tuple[Stage, ...] = ("discovery", "classify", "decide", "draft")


# ---------------------------------------------------------------- state + dependencies


class DiscoveryJob(TypedDict):
    site: str
    query: str
    kind: str


class CaptureJob(TypedDict):
    url: str


class PipelineState(TypedDict, total=False):
    run_id: int
    brand_id: int
    sites: list[str]
    queries: list[dict[str, str]]                          # [{text, kind}]
    discovery: Annotated[list[dict[str, Any]], operator.add]  # one entry per (site, query) task
    prefilter: dict[str, Any]
    kept_urls: list[str]
    captures: Annotated[list[dict[str, Any]], operator.add]   # one entry per capture task
    listing_ids: list[int]
    case_ids: list[int]
    classification: dict[str, Any]
    decision: dict[str, Any]
    drafting: dict[str, Any]
    totals: dict[str, Any]


@dataclass
class PipelineDeps:
    """Run-scoped resources shared by all nodes (passed as LangGraph runtime context, never put in state)."""

    run_id: int
    cfg: AppConfig
    engine: Engine
    browser: BrowserManager
    encoder: ImageEncoder
    http: httpx.AsyncClient
    llm: Any | None = None
    scorer: Scorer | None = None
    sites: list[str] | None = None  # None → every configured marketplace
    # created by load_brand
    brand: BrandContext | None = None
    prefilter: Prefilter | None = None
    site_sems: dict[str, asyncio.Semaphore] = field(default_factory=dict)
    img_sem: asyncio.Semaphore | None = None


Rt = Runtime[PipelineDeps]


def _err(e: BaseException) -> str:
    return redact(f"{type_name(e)}: {e}")[:500]


# ---------------------------------------------------------------- nodes: discovery


async def node_load_brand(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    d = runtime.context
    async with task_span(d.run_id, "load_brand", None, "load_brand") as t:
        d.brand = await load_brand(d.cfg, d.encoder, d.engine)  # always rebuilt: it is in-memory only
        t.payload = {**t.payload, "brand": d.brand.slug, "products": len(d.brand.products),
                     "gold_images": len(d.brand.gold)}
    await asyncio.to_thread(_set_run_brand, d.engine, d.run_id, d.brand.brand_id)
    d.prefilter = Prefilter(d.run_id, d.brand, d.cfg, d.encoder, d.http)
    sites = marketplace_sites(d.cfg, d.sites, with_navigator=navigator_active(d.cfg, d.llm))
    d.site_sems = {s: asyncio.Semaphore(d.cfg.site(s).max_concurrency) for s in sites}
    d.img_sem = asyncio.Semaphore(d.cfg.discovery.image_concurrency)
    return {"brand_id": d.brand.brand_id, "sites": sites}


def _set_run_brand(engine: Engine, run_id: int, brand_id: int) -> None:
    with session_scope(engine) as s:
        run = s.get(Run, run_id)
        if run is not None:
            run.brand_id = brand_id


async def node_generate_queries(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    d = runtime.context
    assert d.brand is not None
    async with task_span(d.run_id, "generate_queries", None, "generate_queries") as t:
        if t.already_done and t.payload.get("kinds"):  # resumed: reuse the exact queries (no second LLM call)
            queries = [Query(q, k) for q, k in zip(t.payload["queries"], t.payload["kinds"], strict=True)]
        else:
            queries = await generate_queries(d.brand, d.cfg, get_rules().vocabulary, d.llm)
            t.payload = {**t.payload, "queries": [q.text for q in queries], "kinds": [q.kind for q in queries]}
    return {"queries": [{"text": q.text, "kind": q.kind} for q in queries]}


def route_discovery(state: PipelineState) -> list[Send] | str:
    """Fan out one discovery task per (site, query)."""
    jobs = [Send("discover", DiscoveryJob(site=s, query=q["text"], kind=q["kind"]))
            for s in state.get("sites", []) for q in state.get("queries", [])]
    return jobs or "dedup_and_prefilter"


async def node_discover(job: DiscoveryJob, runtime: Rt) -> dict[str, Any]:
    """Adaptive search of one (site, query). Prefilter-passing cards land in the shared prefilter cache and
    are also recorded on the task, so a resumed run can restore them without searching again."""
    d = runtime.context
    assert d.prefilter is not None
    site, q = job["site"], job["query"]
    entry: dict[str, Any] = {"site": site, "query": q}
    try:
        async with d.site_sems[site], task_span(d.run_id, "discovery", site, f"discovery:{site}:{q}",
                                                payload={"query": q, "kind": job["kind"]}) as t:
            if t.already_done:
                kept = t.payload.get("kept", [])
                d.prefilter.restore(kept)
                return {"discovery": [{**entry, "status": "skipped", "pages": t.payload.get("pages", 0),
                                       "cards": t.payload.get("cards", 0), "passing": len(kept)}]}
            stats = await search_adaptive(d.browser, site, q, d.prefilter, d.llm)
            t.payload = {**t.payload, **stats, "kept": [s.to_payload() for s in d.prefilter.kept_for(site, q)]}
        return {"discovery": [{**entry, "status": "succeeded", **stats}]}
    except HumanRequired as e:  # recorded on the task as waiting_human
        return {"discovery": [{**entry, "status": "waiting_human", "url": e.url}]}
    except Exception as e:  # recorded on the task as failed; other searches carry on
        log.warning("discovery %s %r failed: %s", site, q, e)
        return {"discovery": [{**entry, "status": "failed", "error": _err(e)}]}


async def node_dedup_and_prefilter(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    """Cards were deduplicated by URL and scored page by page during search; this applies the run's
    listing budget and picks what to capture."""
    d = runtime.context
    assert d.prefilter is not None
    scored = await finalize_prefilter(d.run_id, d.prefilter)
    kept = sorted((s for s in scored if s.keep), key=lambda s: -s.priority)
    runs = state.get("discovery", [])
    summary = {"cards_seen": sum(r.get("cards", 0) for r in runs), "pages": sum(r.get("pages", 0) for r in runs),
               "unique": len(scored), "kept": len(kept),
               "over_budget": sum("over max_listings_per_run" in s.reasons for s in scored)}
    return {"prefilter": summary, "kept_urls": [s.card.url for s in kept]}


def route_capture(state: PipelineState) -> list[Send] | str:
    """Fan out one capture task per kept card."""
    return [Send("capture", CaptureJob(url=u)) for u in state.get("kept_urls", [])] or "collect"


async def node_capture(job: CaptureJob, runtime: Rt) -> dict[str, Any]:
    d = runtime.context
    assert d.prefilter is not None and d.img_sem is not None
    s = d.prefilter.cache[job["url"]]
    entry: dict[str, Any] = {"url": job["url"], "site": s.card.site}
    try:
        lid = await capture_one(d.run_id, s, d.browser, d.llm, d.cfg, d.http, d.img_sem, d.engine,
                                d.site_sems[s.card.site])
        return {"captures": [{**entry, "status": "succeeded" if lid is not None else "skipped", "listing_id": lid}]}
    except HumanRequired as e:
        return {"captures": [{**entry, "status": "waiting_human", "url_waiting": e.url}]}
    except Exception as e:
        log.warning("capture %s failed: %s", job["url"], e)
        return {"captures": [{**entry, "status": "failed", "error": _err(e)}]}


async def node_collect(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    ids = sorted({c["listing_id"] for c in state.get("captures", []) if c.get("listing_id") is not None})
    return {"listing_ids": ids}


# ---------------------------------------------------------------- nodes: classify / decide / draft / report


def _cases_of_run(engine: Engine, run_id: int) -> dict[int, int]:
    """listing_id → case_id for this run's cases."""
    with Session(engine) as s:
        return {c.listing_id: c.id for c in s.exec(select(Case).where(Case.run_id == run_id))}  # type: ignore[misc]


async def node_classify(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    d = runtime.context
    ids = state.get("listing_ids", [])
    existing = await asyncio.to_thread(_cases_of_run, d.engine, d.run_id)
    if ids and not set(ids) <= set(existing):
        # all listings together: seller clustering (Trading Standards overlay) needs the whole set
        res = await classify_listings(d.run_id, ids, d.cfg, d.encoder, d.llm, d.engine, d.scorer)
        summary = {"classified": len(res.case_ids), "by_rule": res.by_rule, "llm_calls": res.llm_calls}
        existing = await asyncio.to_thread(_cases_of_run, d.engine, d.run_id)
    else:
        summary = {"classified": 0, "note": "every captured listing already has a case in this run" if ids
                   else "no listings captured"}
    return {"case_ids": sorted(existing[i] for i in ids if i in existing), "classification": summary}


async def node_decide(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    d = runtime.context
    ids = state.get("case_ids", [])
    async with task_span(d.run_id, "decide", None, "decide", payload={"cases": len(ids)}) as t:
        summary = await asyncio.to_thread(decide_cases, ids, d.cfg, d.engine)
        t.payload = {**t.payload, **summary}
    return {"decision": summary}


async def node_draft(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    """Portal steps 1-4 only for this run's awaiting_approval cases. Declarations and Submit stay with the
    submit-on-approval loop."""
    d = runtime.context
    ids = state.get("case_ids", [])
    async with task_span(d.run_id, "draft_all", "portal", "draft_all", payload={"cases": len(ids)}) as t:
        out = await draft_cases(d.run_id, d.cfg, d.browser, d.engine, case_ids=ids) if ids else \
            {"drafted": [], "failed": [], "waiting_human": None}
        summary = {"drafted": [{"case_id": c, "draft_id": dr} for c, dr in out["drafted"]],
                   "failed": out["failed"], "waiting_human": out["waiting_human"]}
        if not t.already_done:
            t.payload = {**t.payload, **summary}
    return {"drafting": summary}


async def node_report(state: PipelineState, runtime: Rt) -> dict[str, Any]:
    d = runtime.context

    def _write() -> dict[str, Any]:
        totals = compute_run_totals(d.run_id, d.engine)
        with session_scope(d.engine) as s:
            run = s.get(Run, d.run_id)
            assert run is not None
            run.totals = totals
        return totals

    async with task_span(d.run_id, "report", None, "report"):
        totals = await asyncio.to_thread(_write)
    return {"totals": totals}


# ---------------------------------------------------------------- graph


def build_graph(until: Stage = "draft") -> CompiledStateGraph:
    """Compile the pipeline up to and including ``until``, always finishing with ``report``.

    ``until="discovery"`` ends after capture. Use it to collect listings before a scorer exists, e.g. the
    first run on a new brand, followed by ``python -m eval.fit_scorer``.
    """
    if until not in STAGES:
        raise ValueError(f"until must be one of {STAGES}, got {until!r}")
    upto = STAGES[: STAGES.index(until) + 1]
    g: StateGraph = StateGraph(PipelineState, context_schema=PipelineDeps)
    g.add_node("load_brand", node_load_brand)
    g.add_node("generate_queries", node_generate_queries)
    g.add_node("discover", node_discover)
    g.add_node("dedup_and_prefilter", node_dedup_and_prefilter)
    g.add_node("capture", node_capture)
    g.add_node("collect", node_collect)
    g.add_node("report", node_report)
    g.add_edge(START, "load_brand")
    g.add_edge("load_brand", "generate_queries")
    g.add_conditional_edges("generate_queries", route_discovery, ["discover", "dedup_and_prefilter"])
    g.add_edge("discover", "dedup_and_prefilter")
    g.add_conditional_edges("dedup_and_prefilter", route_capture, ["capture", "collect"])
    g.add_edge("capture", "collect")
    prev = "collect"
    for stage, fn in (("classify", node_classify), ("decide", node_decide), ("draft", node_draft)):
        if stage in upto:
            g.add_node(stage, fn)
            g.add_edge(prev, stage)
            prev = stage
    g.add_edge(prev, "report")
    g.add_edge("report", END)
    return g.compile()
