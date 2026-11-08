"""CLI entry for the full pipeline (SPEC §7.1, Phase 8): one command, one Run, one trace.

    python -m bh.orchestrator.run --brand vaelo                 # discovery → classify → decide → draft → report
    python -m bh.orchestrator.run --brand vaelo --llm           # + generic extractor fallback / borderline vision check
    python -m bh.orchestrator.run --brand vaelo --until discovery   # e.g. collect listings before fitting the scorer
    python -m bh.orchestrator.run --brand vaelo --resume 12     # after a human cleared a challenge: same run, skip done work

Needs the mock sites running (``make mock``). The run never approves or submits: lawyers approve in the UI
and ``python -m bh.enforce.submit --poll 10`` submits the approved cases.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from bh.browser.session import BrowserManager
from bh.classify import Scorer
from bh.config import AppConfig, Secrets, get_config, get_secrets, load_config
from bh.db import Case, CaseStatus, Run, RunStatus, get_engine
from bh.discovery.worker import marketplace_sites
from bh.enforce.approvals import resume_case
from bh.orchestrator.graph import STAGES, PipelineDeps, Stage, build_graph
from bh.trace import describe_task_statuses, run_span
from bh.vision import ClipEncoder, ImageEncoder

log = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """The pipeline cannot start (fail before any browsing, with what to do about it)."""


def _includes(until: Stage, stage: Stage) -> bool:
    return STAGES.index(stage) <= STAGES.index(until)


async def preflight(cfg: AppConfig, until: Stage, secrets: Secrets, sites: list[str] | None,
                    brand: str | None, scorer: Scorer | None = None) -> Scorer | None:
    """Check brand, scorer, portal credentials and site reachability. Returns the scorer to use (None
    when classification is not part of this run)."""
    if brand is not None and brand != cfg.brand:
        raise PreflightError(f"--brand {brand!r} but config.yaml protects {cfg.brand!r}; "
                             "use a config for that brand (--config / BH_CONFIG)")
    chosen = marketplace_sites(cfg, sites)
    if not chosen:
        raise PreflightError(f"no marketplace sites selected (configured: {marketplace_sites(cfg)})")
    if not _includes(until, "classify"):
        scorer = None
    elif scorer is None:
        try:
            scorer = Scorer(cfg.paths.resolve(cfg.models.scorer_path))
        except FileNotFoundError as e:
            raise PreflightError(f"{e}. First run: `python -m bh.orchestrator.run --until discovery`, then "
                                 "`python -m eval.fit_scorer`.") from None
    targets = list(chosen)
    if _includes(until, "draft"):
        secrets.require_portal()
        targets.append("portal")
    async with httpx.AsyncClient(timeout=5) as h:
        for site in targets:
            url = cfg.site(site).base_url
            try:
                await h.get(url)
            except httpx.HTTPError as e:
                raise PreflightError(f"{site} not reachable at {url} ({type(e).__name__}); start it with `make mock`"
                                     ) from None
    return scorer


def _check_resumable(engine: Engine, run_id: int, force: bool) -> None:
    with Session(engine) as s:
        run = s.get(Run, run_id)
    if run is None:
        raise PreflightError(f"run {run_id} not found")
    if run.status == RunStatus.running and not force:
        raise PreflightError(f"run {run_id} is still marked running; if its worker died, pass --force")


def release_paused_drafts(engine: Engine, run_id: int) -> list[int]:
    """Resuming means the human finished the challenge: drafts of this run paused by it go back to the
    queue (``resume_case``). Approved cases paused in the submit loop are left to that loop and the UI."""
    with Session(engine) as s:
        paused = [c.id for c in s.exec(select(Case).where(Case.run_id == run_id,
                                                          Case.status == CaseStatus.waiting_human))
                  if not c.decided_by]
    for cid in paused:
        resume_case(cid, engine)  # type: ignore[arg-type]
    return paused  # type: ignore[return-value]


async def run_pipeline(cfg: AppConfig | None = None, *, until: Stage = "draft", llm: Any | None = None,
                       encoder: ImageEncoder | None = None, browser: BrowserManager | None = None,
                       engine: Engine | None = None, secrets: Secrets | None = None, scorer: Scorer | None = None,
                       sites: list[str] | None = None, brand: str | None = None,
                       resume_run_id: int | None = None, force: bool = False) -> dict[str, Any]:
    """Run the LangGraph pipeline as one traced Run. Returns the final graph state, with ``totals`` read
    from the Run after it closed (so they include the whole wall time)."""
    cfg = cfg or get_config()
    eng = engine or get_engine()
    secrets = secrets or (browser.secrets if browser is not None else get_secrets())
    scorer = await preflight(cfg, until, secrets, sites, brand, scorer)
    if resume_run_id is not None:
        _check_resumable(eng, resume_run_id, force)
        released = release_paused_drafts(eng, resume_run_id)
        if released:
            log.info("released paused drafts: %s", released)
    encoder = encoder or ClipEncoder(cfg.vision.clip_model)
    graph = build_graph(until)
    snapshot = {**cfg.snapshot(), "pipeline": {"until": until, "sites": sites, "llm": llm is not None}}

    async with run_span(None, snapshot, engine=eng, resume_run_id=resume_run_id) as run:
        async with httpx.AsyncClient(timeout=30) as http:
            async def go(b: BrowserManager) -> dict[str, Any]:
                deps = PipelineDeps(run.id, cfg, eng, b, encoder, http, llm, scorer, sites)
                return await graph.ainvoke({"run_id": run.id}, context=deps)

            if browser is not None:
                state = await go(browser)
            else:
                async with BrowserManager(cfg, secrets) as b:
                    state = await go(b)
    with Session(eng) as s:
        final = s.get(Run, run.id)
        assert final is not None
        state["totals"], state["run_status"] = final.totals, str(final.status)
    return state


def summarise(state: dict[str, Any]) -> dict[str, Any]:
    disc = state.get("discovery", [])
    caps = state.get("captures", [])
    count = lambda xs: {k: sum(x["status"] == k for x in xs) for k in sorted({x["status"] for x in xs})}  # noqa: E731
    totals = state.get("totals", {})
    out = {
        "run_id": state.get("run_id"), "status": state.get("run_status"),
        "queries": len(state.get("queries", [])), "sites": state.get("sites"),
        "discovery_tasks": count(disc), "prefilter": state.get("prefilter"),
        "captures": count(caps), "listings": len(state.get("listing_ids", [])),
        "classification": state.get("classification"), "decision": state.get("decision"),
        "drafting": state.get("drafting"),
        "tokens_by_model": totals.get("llm", {}).get("by_model", {}),
        "tasks_by_status": totals.get("tasks", {}).get("by_status", {}),
        "tasks_summary": describe_task_statuses(totals.get("tasks", {}).get("by_status", {})),
        "max_parallelism": totals.get("max_parallelism"), "wall_time_s": totals.get("wall_time_s"),
    }
    waiting = [x for x in disc + caps if x["status"] == "waiting_human"]
    if waiting or (state.get("drafting") or {}).get("waiting_human"):
        out["next"] = (f"a human is needed (see the 'Human required' page); then resume with "
                       f"`python -m bh.orchestrator.run --resume {state.get('run_id')}`")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", default=None, help="brand slug; must match the config's brand")
    ap.add_argument("--config", type=Path, default=None, help="alternative config.yaml")
    ap.add_argument("--until", choices=STAGES, default="draft", help="last stage to run (report always runs)")
    ap.add_argument("--sites", nargs="+", default=None, help="marketplaces to search (default: all configured)")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--llm", action="store_true",
                    help="enable LLM calls: generic extractor fallback, borderline vision check, query extension "
                         "(if enabled in config)")
    ap.add_argument("--resume", type=int, default=None, metavar="RUN_ID", help="continue an earlier run")
    ap.add_argument("--force", action="store_true", help="with --resume: reopen a run still marked running")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config) if args.config else get_config()
    if args.max_queries:
        cfg = cfg.model_copy(update={"discovery": cfg.discovery.model_copy(update={"max_queries": args.max_queries})})
    llm = None
    if args.llm:
        from bh.llm import LLMClient

        llm = LLMClient(cfg)
    try:
        state = asyncio.run(run_pipeline(cfg, until=args.until, llm=llm, sites=args.sites, brand=args.brand,
                                         resume_run_id=args.resume, force=args.force))
    except PreflightError as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 2
    print(json.dumps(summarise(state), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
